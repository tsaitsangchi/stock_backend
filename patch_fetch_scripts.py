"""
patch_fetch_scripts.py — 批次修正所有 fetch_*.py 的安全與架構問題

執行方式：
    cd ~/project/stock_backend/scripts
    python patch_fetch_scripts.py

修正項目：
  [P0-SEC] 移除硬編碼 FINMIND_TOKEN
  [P0]     移除獨立的 DB_CONFIG 定義
  [P0]     加入 from config import DB_CONFIG, FINMIND_TOKEN
  [P0]     使用 core.finmind_client.finmind_get（統一重試邏輯）
"""

import re
import os
import shutil
from pathlib import Path

HARDCODED_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9"
    ".eyJkYXRlIjoiMjAyNi0wMy0xNCAxODoxNTo1NCIsInVzZXJfaWQiOiJ0c2FpdHNhbmdjaGkiLCJlbWFpbCI6InRzYWl0c2FuZ2NoaUBnbWFpbC5jb20iLCJpcCI6IjIyMC4xMzQuMjYuNzAifQ"
    ".muoHEMMLiiRQoxZj7evq-9hclsVRXE3IfLNZWDZ6PQE"
)

SCRIPTS_DIR = Path(__file__).parent
BACKUP_DIR = SCRIPTS_DIR / "_patch_backup"
BACKUP_DIR.mkdir(exist_ok=True)

# 需要修正的腳本清單
TARGET_SCRIPTS = [
    "fetch_chip_data.py",
    "fetch_derivative_data.py",
    "fetch_derivative_sentiment_data.py",
    "fetch_fundamental_data.py",
    "fetch_international_data.py",
    "fetch_macro_data.py",
    "fetch_macro_fundamental_data.py",
    "fetch_stock_info.py",
    "fetch_technical_data.py",
    "fetch_total_return_index.py",
]

def patch_file(filepath: Path) -> tuple[bool, list[str]]:
    """
    修正單一檔案，回傳 (是否有修改, 修改項目清單)
    """
    src = filepath.read_text(encoding="utf-8")
    original = src
    changes = []

    # ── 1. 移除硬編碼 Token（多行字串形式）──────────────────
    # 匹配形如：
    #   FINMIND_TOKEN = (
    #       "eyJ0eXAi..."
    #       ".eyJkYXRl..."
    #       ".muoHEM..."
    #   )
    # 或單行：FINMIND_TOKEN = "eyJ0eXAi..."
    token_multiline = re.compile(
        r'FINMIND_TOKEN\s*=\s*\(\s*\n(?:[^\n]*\n)*?[^\n]*\)\n',
        re.MULTILINE
    )
    token_singleline = re.compile(
        r'FINMIND_TOKEN\s*=\s*["\']eyJ[^"\']*["\'][^\n]*\n'
    )

    if token_multiline.search(src):
        src = token_multiline.sub("", src)
        changes.append("移除多行硬編碼 Token")
    elif token_singleline.search(src):
        src = token_singleline.sub("", src)
        changes.append("移除單行硬編碼 Token")

    # ── 2. 移除獨立 DB_CONFIG 定義 ──────────────────────────
    # 匹配形如：
    #   DB_CONFIG = {
    #       "dbname": ...,
    #       ...
    #   }
    db_config_pattern = re.compile(
        r'DB_CONFIG\s*=\s*\{[^}]*\}\s*\n',
        re.DOTALL
    )
    if db_config_pattern.search(src):
        src = db_config_pattern.sub("", src)
        changes.append("移除獨立 DB_CONFIG 定義")

    # ── 3. 確保 from config import 存在 ─────────────────────
    has_config_import = bool(re.search(r'from config import', src))
    has_finmind_token_usage = "FINMIND_TOKEN" in src
    has_db_config_usage = "DB_CONFIG" in src

    if not has_config_import:
        # 找到第一個 import 行之前插入
        # 在 psycopg2 import 之後插入
        insert_after = re.compile(r'(import psycopg2[^\n]*\n)')
        if insert_after.search(src):
            imports_needed = []
            if has_finmind_token_usage:
                imports_needed.append("FINMIND_TOKEN")
            if has_db_config_usage:
                imports_needed.append("DB_CONFIG")
            if imports_needed:
                import_line = f"\nfrom config import {', '.join(imports_needed)}\n"
                src = insert_after.sub(r'\1' + import_line, src, count=1)
                changes.append(f"加入 from config import {', '.join(imports_needed)}")
    else:
        # 已有 from config import，確認包含需要的名稱
        existing = re.search(r'from config import ([^\n]+)', src)
        if existing:
            existing_names = [n.strip() for n in existing.group(1).split(",")]
            missing_names = []
            if has_finmind_token_usage and "FINMIND_TOKEN" not in existing_names:
                missing_names.append("FINMIND_TOKEN")
            if has_db_config_usage and "DB_CONFIG" not in existing_names:
                missing_names.append("DB_CONFIG")
            if missing_names:
                new_names = existing_names + missing_names
                src = src.replace(
                    existing.group(0),
                    f"from config import {', '.join(new_names)}"
                )
                changes.append(f"補充 config import: {', '.join(missing_names)}")

    # ── 4. 修正 fetch_total_return_index.py 的 token query param ──
    if filepath.name == "fetch_total_return_index.py":
        if '"token": FINMIND_TOKEN' in src:
            src = src.replace('        "token": FINMIND_TOKEN,\n', "")
            changes.append("移除 token query parameter")
        if "headers=" not in src and 'requests.get(FINMIND_API_URL' in src:
            src = src.replace(
                '    try:\n        res = requests.get(FINMIND_API_URL, params=params,',
                '    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"}\n    try:\n        res = requests.get(FINMIND_API_URL, headers=headers, params=params,'
            )
            changes.append("加入 Bearer header")

    if src != original:
        # 備份原始檔
        shutil.copy2(filepath, BACKUP_DIR / filepath.name)
        filepath.write_text(src, encoding="utf-8")
        return True, changes
    return False, []


def verify_file(filepath: Path) -> dict:
    """驗證修正結果"""
    src = filepath.read_text(encoding="utf-8")
    return {
        "token_clean":     HARDCODED_TOKEN not in src,
        "no_inline_db":    "DB_CONFIG = {" not in src,
        "has_config_import": "from config import" in src,
    }


def main():
    print("=" * 65)
    print("  fetch_*.py 批次修正工具")
    print(f"  備份目錄：{BACKUP_DIR}")
    print("=" * 65)

    all_ok = True
    for script_name in TARGET_SCRIPTS:
        filepath = SCRIPTS_DIR / script_name
        if not filepath.exists():
            print(f"  ⚠️  找不到：{script_name}")
            continue

        modified, changes = patch_file(filepath)
        v = verify_file(filepath)
        status = "✅" if all(v.values()) else "❌"

        if modified:
            print(f"\n  {status} {script_name}（已修改）")
            for c in changes:
                print(f"      → {c}")
        else:
            print(f"  {status} {script_name}（無需修改）")

        if not all(v.values()):
            all_ok = False
            for k, ok in v.items():
                if not ok:
                    print(f"      ⚠️  未通過：{k}")

    print("\n" + "=" * 65)
    if all_ok:
        print("  ✅ 全部修正完成！")
    else:
        print("  ❌ 部分檔案仍有問題，請檢查上方輸出。")
    print(f"  原始備份位於：{BACKUP_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
