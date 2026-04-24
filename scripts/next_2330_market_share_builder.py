#!/usr/bin/env python3
"""
next_2330_market_share_builder.py  v1.1
全球市佔率資料建立與自動更新器

Schema（3 張獨立表，完全不修改 next_2330_features）：
  ① company_product_market_share   — 每產品市佔詳細資料（時序式，is_active=TRUE 為最新）
  ② company_market_power_score     — 彙總市佔評分 (0~1)，預測器直接讀取
  ③ company_market_share_changelog — 異動記錄，保留完整歷史

使用方法：
  python next_2330_market_share_builder.py --init           # 建表 + 載入基線
  python next_2330_market_share_builder.py --update-auto    # 全自動季度更新（需 duckduckgo_search）
  python next_2330_market_share_builder.py --update 2059    # 互動式手動更新
  python next_2330_market_share_builder.py --recompute      # 重算所有市佔評分
  python next_2330_market_share_builder.py --report         # 列出現有評分報告
  python next_2330_market_share_builder.py --missing        # 列出資料不足的候選股
  python next_2330_market_share_builder.py --cron-setup     # 輸出 crontab 季度自動更新範本
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import date, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════
# 連線設定（與 predictor 相同）
# ══════════════════════════════════════
DB_CONFIG = {
    "dbname":   "stock",
    "user":     "stock",
    "password": "stock",
    "host": "localhost",
    "port":     "5432",
}

# ══════════════════════════════════════
# 評分參數
# ══════════════════════════════════════
RANK_SCORE = {1: 1.00, 2: 0.85, 3: 0.73, 4: 0.63, 5: 0.55}
SCOPE_MULT = {"global": 1.00, "asia": 0.60, "taiwan": 0.30, "unknown": 0.40}
AUTO_UPDATE_DAYS = 90    # 距上次更新超過此天數才觸發自動搜尋
MIN_AUTO_CONFIDENCE = 0.28  # 自動解析結果低於此可信度不寫入


# ══════════════════════════════════════
# DDL — 3 張獨立表
# ══════════════════════════════════════
_DDL_STATEMENTS = [
    # ① 產品市佔詳細表
    """
    CREATE TABLE IF NOT EXISTS company_product_market_share (
        id                  SERIAL          PRIMARY KEY,
        stock_id            VARCHAR(10)     NOT NULL,
        stock_name          VARCHAR(100),
        product_name        VARCHAR(200)    NOT NULL DEFAULT '',
        market_segment      VARCHAR(300)    NOT NULL,
        market_scope        VARCHAR(20)     NOT NULL DEFAULT 'global',
        market_rank         INTEGER,
        market_share_pct    DECIMAL(6,4),
        market_share_min    DECIMAL(6,4),
        market_share_max    DECIMAL(6,4),
        market_size_usd_bn  DECIMAL(10,2),
        top_competitors     JSONB,
        data_source         VARCHAR(300),
        source_url          TEXT,
        reference_date      DATE,
        confidence_score    DECIMAL(3,2)    NOT NULL DEFAULT 0.50,
        analyst_note        TEXT,
        is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
        created_at          TIMESTAMP       NOT NULL DEFAULT NOW(),
        updated_at          TIMESTAMP       NOT NULL DEFAULT NOW()
    )
    """,
    # index
    "CREATE INDEX IF NOT EXISTS idx_cpms_stock  ON company_product_market_share(stock_id)",
    "CREATE INDEX IF NOT EXISTS idx_cpms_active ON company_product_market_share(stock_id, is_active)",

    # ② 彙總評分表（預測器讀取）
    """
    CREATE TABLE IF NOT EXISTS company_market_power_score (
        stock_id            VARCHAR(10)     PRIMARY KEY,
        stock_name          VARCHAR(100),
        best_market_rank    INTEGER,
        top_product_name    VARCHAR(300),
        top_product_ms_pct  DECIMAL(6,4),
        weighted_ms_score   DECIMAL(5,4)    NOT NULL DEFAULT 0.1500,
        market_position     VARCHAR(30),
        global_coverage     BOOLEAN         NOT NULL DEFAULT FALSE,
        ai_server_exposure  BOOLEAN         NOT NULL DEFAULT FALSE,
        key_product_summary TEXT,
        last_updated        DATE            NOT NULL DEFAULT CURRENT_DATE,
        avg_confidence      DECIMAL(3,2)
    )
    """,

    # ③ 異動記錄表
    """
    CREATE TABLE IF NOT EXISTS company_market_share_changelog (
        id              SERIAL          PRIMARY KEY,
        stock_id        VARCHAR(10)     NOT NULL,
        changed_at      TIMESTAMP       NOT NULL DEFAULT NOW(),
        changed_by      VARCHAR(100)    NOT NULL DEFAULT 'system',
        old_score       DECIMAL(5,4),
        new_score       DECIMAL(5,4),
        change_reason   TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_changelog_stock ON company_market_share_changelog(stock_id)",
]


# ══════════════════════════════════════
# 基線資料集（台灣上市公司，研究報告估計值）
# 資料來源：公開研究報告 / 券商研究 / 公司法說會整理
# ══════════════════════════════════════
BASELINE_DATA: dict[str, list] = {

    # ─── 基準訓練股（非候選） ───
    "2330": [{"product_name": "Wafer Foundry", "market_segment": "Global Advanced Process Foundry (<7nm)",
               "market_scope": "global", "market_rank": 1, "market_share_pct": 0.90,
               "market_share_min": 0.85, "market_share_max": 0.95, "market_size_usd_bn": 80.0,
               "top_competitors": ["Samsung", "Intel Foundry", "SMIC"],
               "data_source": "TrendForce / IDC 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.98, "analyst_note": "先進製程幾乎壟斷，3nm以下市佔超過90%"}],
    "2454": [{"product_name": "Mobile AP / SoC", "market_segment": "Global Smartphone AP Market",
               "market_scope": "global", "market_rank": 1, "market_share_pct": 0.40,
               "market_share_min": 0.35, "market_share_max": 0.45, "market_size_usd_bn": 25.0,
               "top_competitors": ["Qualcomm", "Apple", "Samsung LSI"],
               "data_source": "Counterpoint Research 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.92, "analyst_note": "全球手機AP出貨量第一，中低階旗艦雙龍頭"}],
    "3034": [{"product_name": "Display Driver IC", "market_segment": "Global AMOLED/LCD Driver IC",
               "market_scope": "global", "market_rank": 2, "market_share_pct": 0.18,
               "market_share_min": 0.15, "market_share_max": 0.22, "market_size_usd_bn": 10.0,
               "top_competitors": ["Samsung LSI", "Raydium", "Himax"],
               "data_source": "IHS Markit / 券商整理 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.85, "analyst_note": "顯示驅動IC全球前三，iPhone DDI重要供應商"}],

    # ─── AI 伺服器生態鏈 ───
    "2059": [
        {"product_name": "Server Rack Rails", "market_segment": "AI Server High-end Rack Rails (Global)",
         "market_scope": "global", "market_rank": 1, "market_share_pct": 0.80,
         "market_share_min": 0.70, "market_share_max": 0.92, "market_size_usd_bn": 1.5,
         "top_competitors": ["Accuride International", "Sugatsune", "Jonathan Engineering"],
         "data_source": "券商研究 / 法說會 2025Q4", "reference_date": "2025-12-31",
         "confidence_score": 0.88, "analyst_note": "高階AI伺服器滑軌市佔80%+，壟斷NVIDIA GB200/GB300供應鏈"},
        {"product_name": "Server Rack Rails", "market_segment": "Traditional Server Rack Rails (Global)",
         "market_scope": "global", "market_rank": 1, "market_share_pct": 0.32,
         "market_share_min": 0.28, "market_share_max": 0.38, "market_size_usd_bn": 3.0,
         "top_competitors": ["Accuride International", "Sugatsune"],
         "data_source": "公司年報 / 法說會 2025", "reference_date": "2025-06-30",
         "confidence_score": 0.82, "analyst_note": "傳統伺服器滑軌全球第一，30%+市佔"},
    ],
    "3533": [
        {"product_name": "CPU LGA Socket Connector", "market_segment": "AMD EPYC Server CPU Socket (Global)",
         "market_scope": "global", "market_rank": 1, "market_share_pct": 0.50,
         "market_share_min": 0.45, "market_share_max": 0.55, "market_size_usd_bn": 0.8,
         "top_competitors": ["Enplas", "3M Electronic"],
         "data_source": "券商研究 / AMD供應鏈調查 2025", "reference_date": "2025-12-31",
         "confidence_score": 0.87, "analyst_note": "AMD伺服器CPU socket市佔約50%，Intel平台亦有供貨"},
        {"product_name": "Liquid Cooling Quick Disconnect", "market_segment": "AI Server Liquid Cooling Connectors",
         "market_scope": "global", "market_rank": 2, "market_share_pct": 0.20,
         "market_size_usd_bn": 0.5, "top_competitors": ["Colder Products", "Parker Hannifin"],
         "data_source": "公司法說會 2025Q4", "reference_date": "2025-12-31",
         "confidence_score": 0.70, "analyst_note": "水冷快接頭已通過NVIDIA驗證，2025年起量產"},
    ],

    # ─── 半導體 IC ───
    "5269": [
        {"product_name": "USB4/USB3 Host Controller IC", "market_segment": "Global USB4 Host Controller IC",
         "market_scope": "global", "market_rank": 1, "market_share_pct": 0.45,
         "market_share_min": 0.35, "market_share_max": 0.60, "market_size_usd_bn": 2.0,
         "top_competitors": ["Intel (Thunderbolt)", "Texas Instruments", "Renesas"],
         "data_source": "產業研調 / 報告綜合 2025", "reference_date": "2025-06-30",
         "confidence_score": 0.80, "analyst_note": "USB4/Thunderbolt全球少數獨立認證廠，AMD平台指定供應商"},
        {"product_name": "PCIe Bridge/Switch IC", "market_segment": "Global PCIe Gen4/5 Bridge Controller",
         "market_scope": "global", "market_rank": 3, "market_share_pct": 0.15,
         "market_size_usd_bn": 1.2, "top_competitors": ["Broadcom", "Marvell PLX", "Microchip"],
         "data_source": "研究報告估計 2025", "reference_date": "2025-06-30",
         "confidence_score": 0.65, "analyst_note": "PCIe bridge IC全球前三，AI伺服器高速互連受益"},
    ],
    "4919": [
        {"product_name": "TPM Security Chip", "market_segment": "Global Discrete TPM Module",
         "market_scope": "global", "market_rank": 2, "market_share_pct": 0.28,
         "market_share_min": 0.20, "market_share_max": 0.35, "market_size_usd_bn": 0.6,
         "top_competitors": ["Infineon Technologies", "STMicroelectronics"],
         "data_source": "Mordor Intelligence / 研究報告 2025", "reference_date": "2025-06-30",
         "confidence_score": 0.82, "analyst_note": "TPM 2.0晶片全球三大之一，Win11強制需求"},
        {"product_name": "Nuvoton MCU", "market_segment": "Global Industrial/IoT MCU",
         "market_scope": "global", "market_rank": 6, "market_share_pct": 0.04,
         "market_size_usd_bn": 22.0, "top_competitors": ["Renesas", "STMicro", "NXP", "Microchip"],
         "data_source": "IHS Markit MCU Report 2025", "reference_date": "2025-06-30",
         "confidence_score": 0.75, "analyst_note": "全球MCU約第六，工控/安防/IoT利基，取得松下半導體業務後規模擴大"},
    ],
    "3443": [{"product_name": "ASIC Design Services", "market_segment": "Taiwan ASIC Design House",
               "market_scope": "global", "market_rank": 2, "market_share_pct": 0.12,
               "market_size_usd_bn": 5.0, "top_competitors": ["GUC", "Alchip", "Faraday"],
               "data_source": "研究報告 / 公司年報 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.78, "analyst_note": "台灣ASIC設計三大廠，台積電生態深度綁定"}],
    "6526": [{"product_name": "WiFi 6E/7 SoC", "market_segment": "Global Home/Enterprise WiFi SoC",
               "market_scope": "global", "market_rank": 4, "market_share_pct": 0.12,
               "market_share_min": 0.08, "market_share_max": 0.15, "market_size_usd_bn": 8.0,
               "top_competitors": ["Qualcomm", "Broadcom", "MediaTek (parent)", "Realtek"],
               "data_source": "ABI Research 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.72, "analyst_note": "聯發科子公司，WiFi SoC全球前五，家用/工業路由器市場"}],

    # ─── 被動元件 ───
    "2327": [
        {"product_name": "Chip Resistors", "market_segment": "Global Chip Resistor Market",
         "market_scope": "global", "market_rank": 1, "market_share_pct": 0.25,
         "market_share_min": 0.22, "market_share_max": 0.30, "market_size_usd_bn": 3.5,
         "top_competitors": ["Vishay", "KOA Speer", "Rohm"],
         "data_source": "公司年報 / 法說會 2025", "reference_date": "2025-12-31",
         "confidence_score": 0.95, "analyst_note": "晶片電阻全球第一，市佔約25%"},
        {"product_name": "Tantalum Capacitors", "market_segment": "Global Tantalum Capacitor Market",
         "market_scope": "global", "market_rank": 1, "market_share_pct": 0.30,
         "market_size_usd_bn": 1.5, "top_competitors": ["AVX (Kyocera)", "Vishay", "Rohm"],
         "data_source": "公司年報 2025", "reference_date": "2025-12-31",
         "confidence_score": 0.93, "analyst_note": "鉭質電容全球第一（KEMET收購後）"},
        {"product_name": "MLCC", "market_segment": "Global MLCC Market",
         "market_scope": "global", "market_rank": 3, "market_share_pct": 0.10,
         "market_share_min": 0.08, "market_share_max": 0.13, "market_size_usd_bn": 15.0,
         "top_competitors": ["Murata", "Samsung Electro-Mechanics"],
         "data_source": "IHS Markit 2025", "reference_date": "2025-12-31",
         "confidence_score": 0.90, "analyst_note": "MLCC全球第三，AI/EV高階需求受益"},
    ],

    # ─── 設備 ───
    "3563": [
        {"product_name": "PCB AOI Equipment", "market_segment": "Taiwan PCB AOI/Electrical Test",
         "market_scope": "asia", "market_rank": 1, "market_share_pct": 0.35,
         "market_share_min": 0.25, "market_share_max": 0.45, "market_size_usd_bn": 0.6,
         "top_competitors": ["Orbotech (KLA)", "Camtek", "Saki Corporation"],
         "data_source": "公司年報 / 法說會 2025", "reference_date": "2025-06-30",
         "confidence_score": 0.78, "analyst_note": "台灣PCB AOI第一，全球5-15%，切入半導體先進封裝AOI"},
        {"product_name": "Semiconductor Wafer AOI", "market_segment": "Advanced Packaging/Wafer Inspection Equipment",
         "market_scope": "global", "market_rank": 5, "market_share_pct": 0.05,
         "market_size_usd_bn": 3.0, "top_competitors": ["KLA", "Camtek", "Onto Innovation"],
         "data_source": "研究報告估計 2025", "reference_date": "2025-06-30",
         "confidence_score": 0.60, "analyst_note": "半導體AOI新進者，HBM/Chiplet先進封裝成長機會"},
    ],
    "8114": [{"product_name": "Cash Recycling Equipment", "market_segment": "Asia Pacific Cash Handling Machines",
               "market_scope": "asia", "market_rank": 3, "market_share_pct": 0.12,
               "market_size_usd_bn": 2.0, "top_competitors": ["Hitachi-Omron", "GRG Banking", "Nautilus Hyosung"],
               "data_source": "研究報告估計 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.60, "analyst_note": "亞太現金收付設備前三，台灣金融機構主要供應商"}],

    # ─── 環保 ───
    "6581": [{"product_name": "Industrial Waste Treatment Systems",
               "market_segment": "Taiwan Industrial Environmental Services",
               "market_scope": "taiwan", "market_rank": 2, "market_share_pct": 0.18,
               "market_size_usd_bn": 0.3, "top_competitors": ["龍德造船", "群益環保"],
               "data_source": "公司年報 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.65, "analyst_note": "台灣工業廢氣廢液處理前三，技術進入門檻高"}],

    # ─── 生技醫療 ───
    "1795": [{"product_name": "Generic Pharmaceuticals", "market_segment": "Asia Pacific Specialty Generic Drugs",
               "market_scope": "asia", "market_rank": 4, "market_share_pct": 0.06,
               "market_size_usd_bn": 8.0, "top_competitors": ["Sun Pharma", "Teva", "Cipla"],
               "data_source": "研究報告 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.60, "analyst_note": "亞太學名藥前5，特殊劑型毛利率54%"}],
    "6491": [{"product_name": "Contact Lenses", "market_segment": "Taiwan Daily/Extended Wear Contact Lenses",
               "market_scope": "taiwan", "market_rank": 1, "market_share_pct": 0.35,
               "market_size_usd_bn": 0.4, "top_competitors": ["J&J Vision", "Alcon", "Bausch+Lomb"],
               "data_source": "公司年報 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.72, "analyst_note": "台灣隱形眼鏡市佔第一，亞洲市場擴張中"}],
    "6782": [{"product_name": "Specialty Contact Lenses", "market_segment": "Asia Specialty Contact Lenses",
               "market_scope": "asia", "market_rank": 5, "market_share_pct": 0.04,
               "market_size_usd_bn": 2.0, "top_competitors": ["J&J Vision", "Alcon", "CooperVision"],
               "data_source": "研究報告估計 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.55, "analyst_note": "亞洲隱形眼鏡利基，特殊矯視型設計"}],
    "1786": [{"product_name": "HA Medical Aesthetics Filler",
               "market_segment": "Taiwan Medical Aesthetics HA Filler",
               "market_scope": "taiwan", "market_rank": 1, "market_share_pct": 0.40,
               "market_size_usd_bn": 0.2, "top_competitors": ["Allergan (AbbVie)", "Galderma", "Merz"],
               "data_source": "公司年報 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.70, "analyst_note": "台灣醫美玻尿酸市佔第一，小市值高爆發彈性"}],

    # ─── 其他電子 ───
    "6257": [{"product_name": "IC Packaging and Testing", "market_segment": "Global OSAT",
               "market_scope": "global", "market_rank": 8, "market_share_pct": 0.02,
               "market_size_usd_bn": 45.0, "top_competitors": ["ASE", "Amkor", "JCET", "SPIL"],
               "data_source": "TrendForce OSAT 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.75, "analyst_note": "全球OSAT市佔約2%，利基封裝類型"}],
    "3653": [{"product_name": "Server Heatsink/Thermal Module",
               "market_segment": "AI Server Thermal Management Components",
               "market_scope": "global", "market_rank": 3, "market_share_pct": 0.10,
               "market_size_usd_bn": 1.5, "top_competitors": ["Furukawa", "AVC", "Delta Electronics"],
               "data_source": "研究報告估計 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.60, "analyst_note": "AI伺服器散熱片前三"}],
    "6937": [{"product_name": "IC Testing Services", "market_segment": "Taiwan Advanced IC Test (HBM/SoC)",
               "market_scope": "asia", "market_rank": 2, "market_share_pct": 0.15,
               "market_size_usd_bn": 2.0, "top_competitors": ["ASE Testing", "Powertech"],
               "data_source": "公司年報 / 券商研究 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.68, "analyst_note": "台灣獨立測試廠第二大，HBM/先進封裝成長期"}],

    # ─── 其他候選股（保守估計） ───
    "7722": [{"product_name": "Mobile Payment Platform", "market_segment": "Taiwan Mobile Payment",
               "market_scope": "taiwan", "market_rank": 2, "market_share_pct": 0.25,
               "data_source": "金融研究院 2025", "reference_date": "2025-06-30",
               "confidence_score": 0.70, "analyst_note": "台灣行動支付第二，僅限台灣市場"}],
    "2548": [{"product_name": "Luxury Residential Real Estate", "market_segment": "Taiwan Luxury Construction",
               "market_scope": "taiwan", "market_rank": 3, "market_share_pct": 0.05,
               "data_source": "估計", "reference_date": "2025-06-30",
               "confidence_score": 0.40, "analyst_note": "台灣豪宅建商，本地市場"}],
    "6534": [{"product_name": "Specialty Biotech Chemicals", "market_segment": "Taiwan Specialty Biotech Chemicals",
               "market_scope": "taiwan", "market_rank": None, "market_share_pct": 0.10,
               "data_source": "估計", "reference_date": "2025-06-30",
               "confidence_score": 0.40, "analyst_note": "創新板特殊化學/生技，規模小"}],
    "6799": [{"product_name": "Electronic Components Niche", "market_segment": "Taiwan Electronic Components Niche",
               "market_scope": "taiwan", "market_rank": None, "market_share_pct": 0.05,
               "data_source": "估計", "reference_date": "2025-06-30",
               "confidence_score": 0.35, "analyst_note": "利基電子零件，主要內銷"}],
}

STOCK_NAME_MAP = {
    "2330": "台積電", "2454": "聯發科", "3034": "聯詠",
    "2059": "川湖",   "3533": "嘉澤",   "5269": "祥碩",
    "2327": "國巨",   "4919": "新唐",   "3443": "創意",
    "3563": "牧德",   "6526": "達發",   "8114": "振樺電",
    "6581": "鋼聯",   "1795": "美時",   "6491": "晶碩",
    "6782": "視陽",   "1786": "科妍",   "6257": "矽格",
    "3653": "健策",   "6937": "天虹",   "7722": "LINEPAY",
    "2548": "華固",   "6799": "來頡",   "6534": "正瀚-創",
}

# 自動搜尋關鍵字（每季更新）
WATCHLIST_SEARCH = {
    "2059": {"name_en": "King Slide",         "queries": ["King Slide server rack rail global market share 2025", "King Slide AI server rack slides market"]},
    "3533": {"name_en": "Chant Sincere",       "queries": ["Chant Sincere CPU socket market share AMD server 2025"]},
    "5269": {"name_en": "ASMedia Technology",  "queries": ["ASMedia Technology USB controller market share global 2025", "ASMedia USB4 market share"]},
    "2327": {"name_en": "Yageo",               "queries": ["Yageo global market share resistor capacitor 2025", "Yageo passive components ranking"]},
    "4919": {"name_en": "Nuvoton Technology",  "queries": ["Nuvoton TPM market share global 2025", "Nuvoton Technology security chip market share"]},
    "3443": {"name_en": "Global Unichip GUC",  "queries": ["Global Unichip ASIC design market share 2025"]},
    "3563": {"name_en": "Machvision MVI",      "queries": ["Machvision MVI PCB AOI inspection market share 2025"]},
    "6526": {"name_en": "Airoha Technology",   "queries": ["Airoha Technology WiFi SoC market share global 2025"]},
    "6581": {"name_en": "Steel Allied",        "queries": ["鋼聯科技 industrial environmental Taiwan market share"]},
    "6937": {"name_en": "Tian Hong IC Test",   "queries": ["天虹科技 IC testing HBM Taiwan market share 2025"]},
}

# 搜尋解析正規表達式
PERCENT_RE = [
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:global\s+)?(?:market\s+share|share\s+of(?:\s+the)?\s+market)", re.I),
    re.compile(r"(?:market\s+share)\s+of\s+(\d+(?:\.\d+)?)\s*%", re.I),
    re.compile(r"(?:accounts?\s+for|commands?|holds?|captures?|claims?|maintains?)\s+(?:about|approximately|around|nearly|over|more\s+than|roughly)?\s*(\d+(?:\.\d+)?)\s*%", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s+(?:share|of\s+(?:the\s+)?(?:global|world(?:wide)?)\s+market)", re.I),
    re.compile(r"(?:approximately|about|nearly|over|more\s+than|roughly)\s+(\d+(?:\.\d+)?)\s*%", re.I),
]
RANK_RE = [
    re.compile(r"(?:world'?s?|global|worldwide)\s+(?:largest|biggest|leading|number\s+one|#1|no\.?\s*1)\s+(?:supplier|maker|manufacturer|provider|vendor|company)", re.I),
    re.compile(r"rank(?:ed)?\s+(?:as\s+)?#?([1-5])\s+(?:globally|worldwide|in\s+the\s+world|in\s+(?:the\s+)?global)", re.I),
    re.compile(r"(?:one\s+of\s+the|among\s+the)\s+top\s+([2-5])\s+(?:global|world(?:wide)?|largest)", re.I),
    re.compile(r"(?:leading|dominant|top)\s+(?:supplier|manufacturer|provider|vendor|player)\s+(?:globally|worldwide|in\s+the\s+world)", re.I),
]
SOURCE_QUALITY = {
    "mordor": 0.80, "gartner": 0.85, "idc": 0.85, "ihs": 0.80,
    "trendforce": 0.82, "counterpoint": 0.82, "omdia": 0.80,
    "annual report": 0.88, "investor": 0.85, "earnings": 0.80,
    "reuters": 0.65, "bloomberg": 0.65,
}


# ══════════════════════════════════════
# DB 工具
# ══════════════════════════════════════
def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def exec_sql(sql: str, params=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


# ══════════════════════════════════════
# Schema 初始化
# ══════════════════════════════════════
def init_schema():
    log.info("建立 3 張市佔率資料表（若已存在則跳過）…")
    with get_conn() as conn:
        with conn.cursor() as cur:
            for stmt in _DDL_STATEMENTS:
                cur.execute(stmt)
        conn.commit()
    log.info("✓ Schema 建立完成")


# ══════════════════════════════════════
# 評分計算
# ══════════════════════════════════════
def compute_product_score(rank: Optional[int], share_pct: Optional[float],
                           scope: str, confidence: float) -> float:
    if rank and rank <= 5:
        base = RANK_SCORE.get(rank, 0.45)
    elif share_pct:
        base = min(1.0, float(share_pct) / 0.25)   # 25% 市佔 = 滿分 1.0
    else:
        base = 0.20
    mult = SCOPE_MULT.get(scope, 0.40)
    return round(min(1.0, base * mult * float(confidence)), 4)


def determine_position(rank: Optional[int], score: float) -> str:
    if rank == 1:           return "world_no1"
    if rank in (2, 3):      return "top3"
    if rank in (4, 5):      return "top5"
    if score >= 0.50:       return "regional"
    return "local"


# ══════════════════════════════════════
# 資料寫入
# ══════════════════════════════════════
def upsert_product(conn, stock_id: str, stock_name: str, rec: dict):
    """將舊的 active 記錄設 inactive，再 INSERT 新紀錄"""
    seg = rec["market_segment"]
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE company_product_market_share "
            "SET is_active=FALSE, updated_at=NOW() "
            "WHERE stock_id=%s AND market_segment=%s AND is_active=TRUE",
            (stock_id, seg)
        )
        cur.execute("""
            INSERT INTO company_product_market_share (
                stock_id, stock_name, product_name, market_segment, market_scope,
                market_rank, market_share_pct, market_share_min, market_share_max,
                market_size_usd_bn, top_competitors, data_source, source_url,
                reference_date, confidence_score, analyst_note,
                is_active, created_at, updated_at
            ) VALUES (
                %s,%s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s,
                TRUE, NOW(), NOW()
            )""",
            (
                stock_id, stock_name,
                rec.get("product_name", ""),
                seg,
                rec.get("market_scope", "global"),
                rec.get("market_rank"),
                rec.get("market_share_pct"),
                rec.get("market_share_min"),
                rec.get("market_share_max"),
                rec.get("market_size_usd_bn"),
                json.dumps(rec["top_competitors"]) if rec.get("top_competitors") else None,
                rec.get("data_source", "auto-search"),
                rec.get("source_url"),
                rec.get("reference_date", str(date.today())),
                rec.get("confidence_score", 0.50),
                rec.get("analyst_note"),
            )
        )


def recompute_score(stock_id: str, conn=None) -> float:
    own_conn = conn is None
    if own_conn:
        conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT stock_id, stock_name, product_name, market_segment, market_scope,
                       market_rank, market_share_pct, confidence_score, analyst_note
                FROM company_product_market_share
                WHERE stock_id=%s AND is_active=TRUE
            """, (stock_id,))
            rows = cur.fetchall()

        if not rows:
            return 0.15

        scored = [(compute_product_score(r["market_rank"], r["market_share_pct"],
                                         r["market_scope"] or "unknown",
                                         r["confidence_score"] or 0.5), r)
                  for r in rows]
        best_s, best_r = max(scored, key=lambda x: x[0])
        bonus   = sum(s for s, _ in sorted(scored, key=lambda x: x[0], reverse=True)[1:]) * 0.05
        final   = min(1.0, round(best_s + bonus, 4))
        pos     = determine_position(best_r["market_rank"], best_s)
        glb     = any(r["market_scope"] == "global" for r in rows)
        ai_exp  = any("AI" in (r.get("analyst_note") or "") or
                      "server" in (r.get("analyst_note") or "").lower() for r in rows)
        avg_c   = round(sum(r["confidence_score"] or 0.5 for r in rows) / len(rows), 2)

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO company_market_power_score (
                    stock_id, stock_name, best_market_rank, top_product_name,
                    top_product_ms_pct, weighted_ms_score, market_position,
                    global_coverage, ai_server_exposure, key_product_summary,
                    last_updated, avg_confidence
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (stock_id) DO UPDATE SET
                    stock_name          = EXCLUDED.stock_name,
                    best_market_rank    = EXCLUDED.best_market_rank,
                    top_product_name    = EXCLUDED.top_product_name,
                    top_product_ms_pct  = EXCLUDED.top_product_ms_pct,
                    weighted_ms_score   = EXCLUDED.weighted_ms_score,
                    market_position     = EXCLUDED.market_position,
                    global_coverage     = EXCLUDED.global_coverage,
                    ai_server_exposure  = EXCLUDED.ai_server_exposure,
                    key_product_summary = EXCLUDED.key_product_summary,
                    last_updated        = EXCLUDED.last_updated,
                    avg_confidence      = EXCLUDED.avg_confidence
            """, (
                stock_id, best_r["stock_name"],
                best_r["market_rank"], best_r["product_name"],
                best_r["market_share_pct"], final, pos,
                glb, ai_exp,
                (best_r.get("analyst_note") or "")[:200],
                date.today(), avg_c,
            ))
        conn.commit()
        return final
    finally:
        if own_conn:
            conn.close()


# ══════════════════════════════════════
# 載入基線資料
# ══════════════════════════════════════
def load_baseline():
    log.info("載入基線市佔資料…")
    total = 0
    with get_conn() as conn:
        for sid, records in BASELINE_DATA.items():
            if not records:
                continue
            sname = STOCK_NAME_MAP.get(sid, sid)
            for rec in records:
                try:
                    upsert_product(conn, sid, sname, rec)
                    total += 1
                except Exception as e:
                    log.warning(f"  {sid} [{rec.get('market_segment','')}] 失敗：{e}")
            conn.commit()

        for sid in BASELINE_DATA:
            try:
                score = recompute_score(sid, conn)
                log.info(f"  ✓ {sid:6} {STOCK_NAME_MAP.get(sid,''):8} → ms_score={score:.4f}")
            except Exception as e:
                log.warning(f"  {sid} 評分失敗：{e}")

    log.info(f"✓ 基線載入：{total} 筆產品 / {len(BASELINE_DATA)} 支公司")


# ══════════════════════════════════════
# 自動搜尋更新
# ══════════════════════════════════════
def _parse_snippet(text: str) -> dict:
    result = {"share_pct": None, "rank": None}
    for pat in PERCENT_RE:
        m = pat.search(text)
        if m:
            val = float(m.group(1))
            if 0.5 <= val <= 100:
                result["share_pct"] = round(val / 100, 4)
                break
    for pat in RANK_RE:
        m = pat.search(text)
        if m:
            if any(k in text.lower() for k in ["number one", "#1", "no. 1", "largest", "biggest"]):
                result["rank"] = 1
            else:
                try:
                    result["rank"] = int(m.group(1))
                except Exception:
                    pass
            break
    return result


def _estimate_conf(text: str, url: str) -> float:
    base = 0.38
    combo = (text + " " + url).lower()
    for kw, bonus in SOURCE_QUALITY.items():
        if kw in combo:
            base = max(base, bonus)
            break
    if "2025" in text or "2024" in text:
        base = min(1.0, base + 0.05)
    return round(base, 2)


def ddg_search(query: str, max_results: int = 6) -> list:
    """使用 ddgs（新版）或 duckduckgo_search（舊版）搜尋，失敗時改用 HTML 備援"""
    # 優先 ddgs（新套件名）
    try:
        from ddgs import DDGS as DDGS_NEW
        with DDGS_NEW() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except ImportError:
        pass
    # 次選：舊套件名
    try:
        from duckduckgo_search import DDGS
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
    except ImportError:
        pass
    # 備援：直接抓 DuckDuckGo HTML
    try:
        from bs4 import BeautifulSoup
        from urllib.parse import quote_plus
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        r = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for div in soup.select(".result")[:max_results]:
            t = div.select_one(".result__title")
            b = div.select_one(".result__snippet")
            a = div.select_one("a.result__url")
            if t and b:
                out.append({"title": t.get_text(strip=True),
                            "body":  b.get_text(strip=True),
                            "href":  a["href"] if a else ""})
        return out
    except Exception as e:
        log.debug(f"搜尋備援失敗：{e}")
        return []


def auto_update_stock(stock_id: str) -> bool:
    cfg = WATCHLIST_SEARCH.get(stock_id)
    if not cfg:
        return False
    log.info(f"  搜尋 {stock_id} {cfg['name_en']}…")
    snippets = []
    for q in cfg["queries"]:
        for r in ddg_search(q, max_results=4):
            text = f"{r.get('title','')} {r.get('body','')}".strip()
            snippets.append((text, r.get("href", "")))
        time.sleep(1.5)

    if not snippets:
        log.info(f"  {stock_id} 無搜尋結果")
        return False

    combined = " ".join(s for s, _ in snippets)
    parsed   = _parse_snippet(combined)
    if not parsed["share_pct"] and not parsed["rank"]:
        log.info(f"  {stock_id} 未解析到市佔數值")
        return False

    conf = _estimate_conf(combined, snippets[0][1])
    if conf < MIN_AUTO_CONFIDENCE:
        log.info(f"  {stock_id} 可信度 {conf:.2f} 偏低，跳過")
        return False

    sname = STOCK_NAME_MAP.get(stock_id, stock_id)
    rec = {
        "product_name":    f"Auto-detected: {cfg['name_en']}",
        "market_segment":  f"Global Market Auto-Search {date.today().strftime('%Y-Q%q')[:8]}",
        "market_scope":    "global",
        "market_rank":     parsed["rank"],
        "market_share_pct": parsed["share_pct"],
        "data_source":     f"DuckDuckGo auto-search {date.today()}",
        "source_url":      snippets[0][1],
        "reference_date":  str(date.today()),
        "confidence_score": conf,
        "analyst_note":    f"自動解析 rank={parsed['rank']} share={parsed['share_pct']}",
    }
    with get_conn() as conn:
        # 取舊分數
        with conn.cursor() as cur:
            cur.execute("SELECT weighted_ms_score FROM company_market_power_score WHERE stock_id=%s", (stock_id,))
            row = cur.fetchone()
            old_s = float(row[0]) if row else None

        upsert_product(conn, stock_id, sname, rec)
        conn.commit()
        new_s = recompute_score(stock_id, conn)

        with conn.cursor() as cur:
            cur.execute("""INSERT INTO company_market_share_changelog
                (stock_id, changed_by, old_score, new_score, change_reason)
                VALUES (%s,'auto-search',%s,%s,%s)""",
                (stock_id, old_s, new_s,
                 f"auto rank={parsed['rank']} share={parsed['share_pct']} conf={conf}"))
        conn.commit()

    log.info(f"  ✓ {stock_id} 更新：ms_score {old_s} → {new_s:.4f}")
    return True


def run_auto_update(force: bool = False):
    cutoff = date.today() - timedelta(days=AUTO_UPDATE_DAYS)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT stock_id, last_updated FROM company_market_power_score")
            existing = {r[0]: r[1] for r in cur.fetchall()}

    to_update = [
        sid for sid in WATCHLIST_SEARCH
        if force or sid not in existing or (existing[sid] and existing[sid] < cutoff)
    ]
    if not to_update:
        log.info("所有公司均在更新期限內（90天），無需更新")
        return
    log.info(f"待更新 {len(to_update)} 支：{to_update}")
    ok = 0
    for sid in to_update:
        ok += auto_update_stock(sid)
        time.sleep(2)
    log.info(f"自動更新完成：{ok}/{len(to_update)} 成功")


# ══════════════════════════════════════
# 手動更新
# ══════════════════════════════════════
def manual_update(stock_id: str, product: str, segment: str,
                  rank: Optional[int], share: Optional[float],
                  scope: str, source: str, note: str):
    sname = STOCK_NAME_MAP.get(stock_id, stock_id)
    rec = {
        "product_name":    product,
        "market_segment":  segment or product,
        "market_scope":    scope,
        "market_rank":     rank,
        "market_share_pct": share,
        "data_source":     source,
        "reference_date":  str(date.today()),
        "confidence_score": 0.85,
        "analyst_note":    note,
    }
    with get_conn() as conn:
        upsert_product(conn, stock_id, sname, rec)
        conn.commit()
        new_s = recompute_score(stock_id, conn)
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO company_market_share_changelog
                (stock_id, changed_by, new_score, change_reason)
                VALUES (%s,'manual',%s,%s)""",
                (stock_id, new_s, f"手動更新：{product} scope={scope} rank={rank} share={share}"))
        conn.commit()
    log.info(f"✓ {stock_id} {sname} 手動更新完成，ms_score={new_s:.4f}")


# ══════════════════════════════════════
# 報告 & 工具
# ══════════════════════════════════════
def print_report(top_n: int = 30):
    with get_conn() as conn:
        df = pd.read_sql("""
            SELECT stock_id, stock_name, weighted_ms_score,
                   best_market_rank, top_product_name, top_product_ms_pct,
                   market_position, global_coverage, ai_server_exposure,
                   last_updated, avg_confidence
            FROM company_market_power_score
            ORDER BY weighted_ms_score DESC
            LIMIT %s
        """, conn, params=(top_n,))
    if df.empty:
        print("  無資料，請先執行 --init"); return
    print(f"\n{'='*92}")
    print(f"  全球市佔率評分報告  Top {min(top_n, len(df))}  (ms_score: 0=本地  1=全球絕對壟斷)")
    print(f"{'='*92}")
    print(f"  {'代號':6} {'名稱':10} {'ms分':6} {'排名':4} {'地位':12} {'AI':3} {'信度':5} {'更新日':12} 核心產品")
    print("  " + "─" * 88)
    for _, r in df.iterrows():
        rnk  = str(r["best_market_rank"]) if r["best_market_rank"] else "N/A"
        ai   = "✓" if r["ai_server_exposure"] else " "
        prod = (r["top_product_name"] or "")[:28]
        print(f"  {str(r['stock_id']):6} {str(r['stock_name'])[:8]:10} "
              f"{r['weighted_ms_score']:.4f}  {rnk:4} "
              f"{str(r['market_position'])[:12]:12} {ai:3} "
              f"{r['avg_confidence']:.2f}   {str(r['last_updated']):12} {prod}")


def print_missing():
    with get_conn() as conn:
        try:
            cands = pd.read_sql(
                "SELECT DISTINCT stock_id, stock_name FROM next_2330_features "
                "WHERE date=(SELECT MAX(date) FROM next_2330_features)",
                conn)
        except Exception:
            print("  無法讀取 next_2330_features"); return
        have = pd.read_sql(
            "SELECT stock_id FROM company_market_power_score WHERE weighted_ms_score > 0.18",
            conn)
    miss = cands[~cands["stock_id"].isin(have["stock_id"])]
    print(f"\n  候選池中缺乏市佔資料：{len(miss)} 支")
    for _, r in miss.iterrows():
        print(f"    {r['stock_id']:10} {r.get('stock_name','')}")


def print_cron_setup():
    import os
    script = os.path.abspath(__file__)
    py = sys.executable
    print("""
  ═══ 季度自動更新 crontab 設定 ═══

  執行以下指令編輯 crontab：
    crontab -e

  加入下列行（每季第一天 02:00 執行）：
  ─────────────────────────────────────────────────────────
  # 1月/4月/7月/10月 1日 02:00 執行市佔率自動更新
  0 2 1 1,4,7,10 * {py} {script} --update-auto >> /var/log/market_share_update.log 2>&1
  ─────────────────────────────────────────────────────────

  或使用 systemd timer（推薦）：
    sudo tee /etc/systemd/system/market-share-update.service << 'EOF'
  [Unit]
  Description=Stock Global Market Share Auto Update

  [Service]
  Type=oneshot
  ExecStart={py} {script} --update-auto
  StandardOutput=append:/var/log/market_share_update.log
  StandardError=append:/var/log/market_share_update.log
  EOF

    sudo tee /etc/systemd/system/market-share-update.timer << 'EOF'
  [Unit]
  Description=Quarterly market share update

  [Timer]
  OnCalendar=*-01,04,07,10-01 02:00:00
  Persistent=true

  [Install]
  WantedBy=timers.target
  EOF

    sudo systemctl enable --now market-share-update.timer
    sudo systemctl list-timers market-share-update.timer
""".format(py=py, script=script))


# ══════════════════════════════════════
# CLI
# ══════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description="next_2330_market_share_builder v1.1 — 全球市佔率資料建立與自動更新器"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init",        action="store_true", help="建表 + 載入基線資料")
    g.add_argument("--update-auto", action="store_true", help="全自動季度更新（90天未更新才觸發）")
    g.add_argument("--update",      metavar="STOCK_ID",  help="手動更新指定公司")
    g.add_argument("--recompute",   action="store_true", help="重新計算所有公司市佔評分")
    g.add_argument("--report",      action="store_true", help="輸出現有評分報告")
    g.add_argument("--missing",     action="store_true", help="列出資料不足的候選股")
    g.add_argument("--cron-setup",  action="store_true", help="顯示 crontab/systemd 季度排程設定")

    p.add_argument("--product",  default="", help="產品名稱（--update 用）")
    p.add_argument("--segment",  default="", help="市場細分描述（不填同 --product）")
    p.add_argument("--rank",     type=int,   help="全球排名整數")
    p.add_argument("--share",    type=float, help="市佔率 0~1（如 0.30 = 30%%）")
    p.add_argument("--scope", default="global", choices=["global","asia","taiwan"], help="市場範圍")
    p.add_argument("--source",   default="手動更新", help="資料來源說明")
    p.add_argument("--note",     default="",        help="分析師備註")
    p.add_argument("--force",    action="store_true", help="強制更新（忽略 90 天限制）")
    p.add_argument("--top",      type=int, default=30, help="報告顯示筆數")

    args = p.parse_args()

    # DB 連線測試
    try:
        with get_conn() as conn:
            conn.cursor().execute("SELECT 1")
        log.info("✓ PostgreSQL 連線成功")
    except Exception as e:
        log.error(f"✗ 無法連線 PostgreSQL：{e}")
        sys.exit(1)

    if args.init:
        init_schema()
        load_baseline()
        print_report(top_n=args.top)

    elif args.update_auto:
        run_auto_update(force=args.force)
        print_report(top_n=args.top)

    elif args.update:
        if not args.product:
            print("錯誤：--update 需要 --product 指定產品名稱")
            sys.exit(1)
        manual_update(args.update, args.product,
                      args.segment or args.product,
                      args.rank, args.share, args.scope,
                      args.source, args.note)

    elif args.recompute:
        log.info("重算所有公司市佔評分…")
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT stock_id FROM company_product_market_share WHERE is_active=TRUE")
                ids = [r[0] for r in cur.fetchall()]
        for sid in ids:
            s = recompute_score(sid)
            log.info(f"  {sid:6} {STOCK_NAME_MAP.get(sid,''):8} → {s:.4f}")
        print_report(top_n=args.top)

    elif args.report:
        print_report(top_n=args.top)

    elif args.missing:
        print_missing()

    elif args.cron_setup:
        print_cron_setup()


if __name__ == "__main__":
    main()
