from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import psycopg2
import bcrypt
import os
from scripts.config import DB_CONFIG

app = FastAPI(title="StockFront Auth Service")

# CORS 設定，允許 Vue 前端存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 正式環境建議縮小範圍
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 密碼加密與驗證工具
def hash_password(password: str) -> str:
    # 生成鹽並加密
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

class UserRegister(BaseModel):
    gmail: EmailStr
    password: str
    full_name: str = None

def get_db_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

@app.post("/api/auth/register")
def register_user(user: UserRegister, conn=Depends(get_db_conn)):
    try:
        print(f"Attempting to register user: {user.gmail}")
        cur = conn.cursor()
        
        # 1. 檢查 Gmail 是否已存在
        cur.execute("SELECT id FROM public.users WHERE gmail = %s", (user.gmail,))
        if cur.fetchone():
            print(f"Registration failed: {user.gmail} already exists")
            raise HTTPException(status_code=400, detail="Gmail already registered")
        
        # 2. 加密密碼
        hashed_password = hash_password(user.password)
        
        # 3. 寫入資料庫
        print(f"Inserting into database: {user.gmail}")
        cur.execute(
            "INSERT INTO public.users (gmail, password_hash, full_name) VALUES (%s, %s, %s)",
            (user.gmail, hashed_password, user.full_name)
        )
        conn.commit()
        cur.close()
        print(f"Successfully registered: {user.gmail}")
        return {"message": "User registered successfully", "gmail": user.gmail}
    except Exception as e:
        print(f"CRITICAL ERROR during registration: {str(e)}")
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import sys
    try:
        print("Starting StockFront Auth Service...")
        uvicorn.run(app, host="0.0.0.0", port=8001)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
