# 技術實作規劃：遞迴自癒啟動 (Recursive Bootstrap)

## 1. 架構願景
「遞迴自癒啟動」旨在消除對 `PYTHONPATH` 或絕對路徑的依賴。系統必須具備在任何執行層級（Script Level, Module Level, Container Level）自動導航回核心根目錄的能力。

## 2. 核心運作邏輯

### 2.1 溯源路徑 (Path Tracing)
系統啟動時，獲取當前進入點檔案的絕對路徑：
```python
current_path = Path(__file__).resolve()
```

### 2.2 特徵標記識別 (Feature Marking)
向上逐層掃描父目錄，直到滿足以下「標記條件」之一：
*   **目錄名標記**：目錄名稱為 `scripts`。
*   **結構標記**：目錄下同時存在 `core/` 與 `fetchers/` (核心功能組件)。
*   **根目錄標記**：目錄下存在 `.env` 或 `requirements.txt`。

### 2.3 命名空間自癒 (Namespace Healing)
識別出 `scripts` 根目錄後，執行兩次注入：
1.  `sys.path.insert(0, str(scripts_dir))`：支援 `from core...` 匯入。
2.  `sys.path.insert(0, str(scripts_dir.parent))`：支援 `from scripts.core...` 匯入。

## 3. 現有實作評估 (v2.3)
在 `path_setup.py` v2.3 中，我們已實作了 `_resolve_scripts_dir` 函數，具備了基本的溯源能力。

## 4. 未來強化方向
- **支援多重專案根目錄**：在分散式架構下，可能存在多個不同的 scripts 目錄。
- **防止循環依賴**：確保 `path_setup` 是系統中第一個被載入且不依賴其他業務邏輯的模組。
- **快取機制**：在多並行訓練中，路徑解析結果應進行進程級快取，減少重複 I/O。

---
**文件狀態**：已整合至系統基準線 v5.1
