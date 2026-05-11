
try:
    import config
    print("✅ 找到 config.py")
except ImportError:
    print("⚠️ 提醒：找不到 config.py。系統已完全切換至資料庫主權模式。")
    print("💡 如需更新資產，請直接操作資料庫 stocks 表或使用 enrichment 工具。")
