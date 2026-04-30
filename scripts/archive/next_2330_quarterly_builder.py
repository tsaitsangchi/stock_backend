#!/usr/bin/env python3
"""
next_2330_quarterly_builder.py - 批量每季執行 feature builder（已修正時間切片）
使用: python next_2330_quarterly_builder.py [--start 1990-01-01] [--end 2026-03-31]

修改重點：
  ① 每季自動計算正確的 --start-date（該 calc-date 往前 10 年）
     例：calc-date=2010-12-31 → --start-date=2000-12-31
  ② 強制加上 --force，確保每一季重新計算（避免舊資料殘留）
  ③ 保留 --resume 斷點續算與重試機制
  ④ dry-run 模式同步顯示每季對應的 --start-date
  ⑤ 每季間隔 3 秒，避免 PostgreSQL 過載
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta       # timedelta 供 generate_quarterly_dates 使用
from dateutil.relativedelta import relativedelta
import logging

# ══════════════════════════════════════════════
# Log 設定（同時輸出到檔案與終端機）
# ══════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quarterly_builder.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 日期工具
# ══════════════════════════════════════════════
def generate_quarterly_dates(start_date: str, end_date: str) -> list:
    """生成每季末日期清單（3/6/9/12 月底）"""
    dates = []
    current = datetime.strptime(start_date, '%Y-%m-%d').replace(day=1)
    end_dt  = datetime.strptime(end_date,   '%Y-%m-%d')

    while current <= end_dt:
        if current.month in [3, 6, 9, 12]:
            # 計算該月最後一天
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)
            quarter_end = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
            dates.append(quarter_end)
        current += relativedelta(months=1)

    return dates


def get_start_date_for_calc(calc_date_str: str) -> str:
    """
    自動計算該季的 --start-date（calc-date 往前 10 年）。

    feature_builder 需要長達 10 年的歷史資料來計算：
      - gross_margin_10y_stability（10 年毛利率穩定度）
      - rev_cagr_5y（5 年營收 CAGR）
      - rev_acceleration（近 3 年 vs 前 3 年 CAGR）
    所以往前 10 年可覆蓋所有特徵的資料窗需求。
    """
    calc_dt  = datetime.strptime(calc_date_str, '%Y-%m-%d')
    start_dt = calc_dt - relativedelta(years=10)
    return start_dt.strftime('%Y-%m-%d')


# ══════════════════════════════════════════════
# 執行單一季
# ══════════════════════════════════════════════
def run_feature_builder(calc_date: str, max_retries: int = 3) -> bool:
    """
    執行單一季的 feature builder。

    自動帶入：
      --calc-date  → 該季末日期（作為資料上界）
      --start-date → calc-date 往前 10 年（資料下界）
      --force      → 強制重算，避免殘留舊快照值
      --resume     → 批次內若有部分股票已完成可跳過（斷點續算）
    """
    start_date = get_start_date_for_calc(calc_date)
    cmd = [
        'python', 'next_2330_feature_builder.py',
        '--calc-date',  calc_date,
        '--start-date', start_date,
        '--force',
        '--resume',
    ]

    logger.info(f"執行 {calc_date}  (資料起始: {start_date})")

    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,          # 15 分鐘上限（全市場約 1,800 支股票）
                cwd='.'
            )
            if result.returncode == 0:
                logger.info(f"✓ {calc_date} 完成（資料起始 {start_date}）")
                return True
            else:
                logger.warning(
                    f"⚠ {calc_date} 失敗（嘗試 {attempt + 1}）：{result.stderr[:400]}..."
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠ {calc_date} 超時（嘗試 {attempt + 1}）")
        except Exception as e:
            logger.warning(f"⚠ {calc_date} 異常（嘗試 {attempt + 1}）：{e}")

        if attempt < max_retries:
            logger.info(f"  重試第 {attempt + 2} 次...")
            time.sleep(8)

    logger.error(f"✗ {calc_date} 最終失敗")
    return False


# ══════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='批量每季執行 2330 feature builder（自動時間切片）'
    )
    parser.add_argument('--start',   default='1990-01-01',
                        help='批次起始日期（預設 1990-01-01）')
    parser.add_argument('--end',     default='2026-03-31',
                        help='批次結束日期（預設 2026-03-31）')
    parser.add_argument('--dry-run', action='store_true',
                        help='僅列出待執行清單，不實際執行')
    parser.add_argument('--resume',  type=str,
                        help='從指定日期繼續（例：2015-12-31）')
    args = parser.parse_args()

    # ── 生成季末日期清單 ───────────────────────
    dates = generate_quarterly_dates(args.start, args.end)
    if not dates:
        logger.error("找不到任何符合條件的季末日期，請確認 --start / --end 參數")
        sys.exit(1)

    # ── 斷點續算 ───────────────────────────────
    if args.resume:
        try:
            resume_idx = dates.index(args.resume)
            dates = dates[resume_idx:]
            logger.info(f"從 {args.resume} 繼續執行（剩餘 {len(dates)} 季）")
        except ValueError:
            logger.warning(f"找不到 {args.resume}，將從頭開始執行")

    logger.info(f"總共 {len(dates)} 季（{dates[0]} → {dates[-1]}）")

    # ── Dry-run：列出清單後離開 ────────────────
    if args.dry_run:
        print("\nDry-run 模式，預定執行清單：")
        for i, d in enumerate(dates, 1):
            s = get_start_date_for_calc(d)
            print(f"  {i:3d}. calc-date={d}  →  start-date={s}")
        print(f"\n共 {len(dates)} 季，加 --dry-run 移除此旗標後即可正式執行。")
        return

    # ── 正式執行 ──────────────────────────────
    success = failed = 0
    start_time = time.time()

    for i, calc_date in enumerate(dates, 1):
        logger.info(f"\n[{i:3d}/{len(dates)}] {calc_date}")

        if run_feature_builder(calc_date):
            success += 1
        else:
            failed += 1

        # 每 4 季（1 年）報告一次進度
        if i % 4 == 0:
            elapsed = (time.time() - start_time) / 60
            logger.info(
                f"進度 {i}/{len(dates)} ({i / len(dates) * 100:.1f}%)  "
                f"成功:{success}  失敗:{failed}  已用 {elapsed:.1f} 分鐘"
            )

        time.sleep(3)   # 每季間隔 3 秒，避免 PostgreSQL 過載

    # ── 最終摘要 ──────────────────────────────
    total_min = (time.time() - start_time) / 60
    logger.info(f"\n{'=' * 55}")
    logger.info(f"=== 全部完成 ===")
    logger.info(f"成功 {success}/{len(dates)} 季（{success / len(dates) * 100:.1f}%）")
    logger.info(f"失敗 {failed} 季")
    logger.info(f"總耗時 {total_min:.1f} 分鐘（{total_min / 60:.2f} 小時）")
    logger.info(f"完整 log：quarterly_builder.log")
    logger.info(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
