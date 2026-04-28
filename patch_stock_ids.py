import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace the select query with STOCK_CONFIGS
    old_code = """    with conn.cursor() as cur:
        cur.execute(
            "SELECT stock_id FROM stock_info WHERE type IN ('twse', 'otc') ORDER BY stock_id"
        )
        return [row[0] for row in cur.fetchall()]"""
        
    new_code = """    from config import STOCK_CONFIGS
    return list(STOCK_CONFIGS.keys())"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Patched {filepath}")

for script in ['scripts/fetch_chip_data.py', 'scripts/fetch_sponsor_chip_data.py', 'scripts/fetch_fundamental_data.py']:
    patch_file(script)
