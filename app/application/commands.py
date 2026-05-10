import uuid
from dataclasses import dataclass

@dataclass(frozen=True, slots=True, kw_only=True)
class RegisterUserCommand:
    """業務意圖：註冊新使用者"""
    email: str
    password_hash: str  # 敏感資訊通常已在 Controller 或 Security 層處理
    full_name: str

@dataclass(frozen=True, slots=True, kw_only=True)
class UpdateProfileCommand:
    """業務意圖：更新使用者個人資料"""
    user_id: uuid.UUID
    new_name: str

@dataclass(frozen=True, slots=True, kw_only=True)
class SuspendUserCommand:
    """業務意圖：對帳戶執行停權"""
    user_id: uuid.UUID
    reason: str