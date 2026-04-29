def fetch_eight_banks(conn, stock_ids, start, end, delay, force):
    """
    八大行庫買賣超 (TaiwanStockGovernmentBankBuySell)
    注意：此 Dataset 不支援 data_id，必須一次抓取全市場。
    """
    logger.info(f"\n=== [eight_banks] 開始抓取全市場資料 ({start} ~ {end}) ===")
    ensure_ddl(conn, DDL_EIGHT_BANKS)
    
    # 這裡我們採取按月抓取的策略，避免單次請求過大
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    current_dt = start_dt
    total = 0
    
    # 為了加速，我們先找出 DB 裡最大的日期
    if not force:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM eight_banks")
            last_date = cur.fetchone()[0]
            if last_date:
                current_dt = max(current_dt, datetime.combine(last_date + timedelta(days=1), datetime.min.time()))

    while current_dt <= end_dt:
        month_end = (current_dt.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        chunk_end = min(month_end, end_dt)
        
        s_str = current_dt.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")
        
        logger.info(f"  抓取 {s_str} ~ {e_str}...")
        rows = finmind_get("TaiwanStockGovernmentBankBuySell",
                           {"start_date": s_str, "end_date": e_str}, delay)
        
        if rows:
            # 雖然是抓全市場，我們還是可以過濾出我們感興趣的 87 支，或者乾脆全存
            # 這裡選擇全存，因為八大行庫資料對其他股票也有用
            records = [
                (r.get("date"), r.get("stock_id"), safe_int(r.get("buy")), safe_int(r.get("sell")))
                for r in rows
            ]
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, UPSERT_EIGHT_BANKS, records)
            conn.commit()
            total += len(records)
            logger.info(f"    → 寫入 {len(records):,} 筆（累計 {total:,} 筆）")
            
        current_dt = chunk_end + timedelta(days=1)
        if s_str == e_str and not rows: # 防止死循環
             current_dt += timedelta(days=1)

    logger.info(f"=== [eight_banks] 完成，累計 {total:,} 筆 ===")
