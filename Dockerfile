# 使用輕量級 Python 鏡像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (PostgreSQL 開發庫)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案代碼
COPY scripts/ ./scripts/

# 設定 PYTHONPATH，確保 scripts 內的模組可互相引用
ENV PYTHONPATH=/app/scripts

# 預設執行指令：啟動投資組合優化器
# 未來可透過 docker run 指令切換至訓練或預測模式
CMD ["python", "scripts/portfolio_optimizer.py", "--budget", "100000"]
