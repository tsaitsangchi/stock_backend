import os
import requests
import google.generativeai as genai
import argparse
from dotenv import load_dotenv

# ==========================================
# 1. 設定與環境變數
# ==========================================
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GITHUB_TOKEN or not GEMINI_API_KEY:
    print("❌ 錯誤：請確保 .env 檔案中設定了 GITHUB_TOKEN 與 GEMINI_API_KEY")
    exit(1)

# ==========================================
# 2. 初始化 Gemini API
# ==========================================
genai.configure(api_key=GEMINI_API_KEY)
# 使用最新的模型
DEFAULT_MODEL = 'gemini-flash-latest'

# ==========================================
# 3. 定義抓取 GitHub 程式碼的函式
# ==========================================
def fetch_github_file(owner, repo, file_path, branch, token):
    """透過 GitHub REST API 取得特定檔案的內容"""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    
    headers = {
        "Accept": "application/vnd.github.v3.raw",
        "Authorization": f"token {token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    print(f"🔄 正在從 GitHub 抓取檔案: {file_path} ...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print(f"✅ {file_path} 抓取成功！")
        return response.text
    else:
        print(f"❌ {file_path} 抓取失敗。狀態碼: {response.status_code}")
        return None

# ==========================================
# 4. 定義與 Gemini 對話的函式
# ==========================================
def analyze_code_with_gemini(files_content, task_description, model_name=DEFAULT_MODEL):
    """將程式碼與任務描述送給 Gemini 分析"""
    model = genai.GenerativeModel(model_name)
    
    # 組合多檔案內容
    code_block = ""
    for path, content in files_content.items():
        code_block += f"\n--- File: {path} ---\n```python\n{content}\n```\n"
    
    print(f"🤖 正在使用 {model_name} 進行分析...")
    
    prompt = f"""
    你是一位資深的軟體工程師。請根據以下任務描述，分析這段程式碼。
    
    【任務描述】: {task_description}
    
    【程式碼內容】:
    {code_block}
    """
    
    try:
        response = model.generate_content(prompt)
        print("\n" + "="*50)
        print("✨ Gemini 分析結果：")
        print("="*50 + "\n")
        print(response.text)
    except Exception as e:
         print(f"❌ Gemini API 發生錯誤: {e}")

# ==========================================
# 5. 主程式執行邏輯
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="GitHub 程式碼同步與 Gemini 分析工具")
    parser.add_argument("--owner", default="tsaitsangchi", help="GitHub 擁有者")
    parser.add_argument("--repo", default="stock_backend", help="GitHub 專案名稱")
    parser.add_argument("--files", nargs="+", default=["main.py"], help="要分析的檔案路徑（多個請用空格分開）")
    parser.add_argument("--branch", default="master", help="專案分支")
    parser.add_argument("--task", default="請總結這些檔案的主要功能，並指出有沒有需要改善的地方？", help="要給 Gemini 的任務描述")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="使用的 Gemini 模型名稱")
    
    args = parser.parse_args()
    
    files_content = {}
    for file_path in args.files:
        code = fetch_github_file(args.owner, args.repo, file_path, args.branch, GITHUB_TOKEN)
        if code:
            files_content[file_path] = code
            
    if files_content:
        analyze_code_with_gemini(files_content, args.task, args.model)
    else:
        print("⚠️ 未能抓取到任何有效的程式碼內容。")

if __name__ == "__main__":
    main()