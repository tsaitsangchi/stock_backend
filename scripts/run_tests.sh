export PYTHONPATH=$PYTHONPATH:$(pwd)/scripts
echo '🚀 啟動量子藍圖單元測試套件...'
./venv/bin/pytest tests/ -v
