import uuid
from abc import ABC, abstractmethod
from typing import Optional
from app.domain.entities import User, Email

class UserRepository(ABC):
    """
    User 聚合根的專屬倉儲介面。
    領域層只認識這個契約，徹底與基礎設施解耦。
    """
    
    @abstractmethod
    def add(self, user: User) -> None:
        """將新建的 User 加入倉儲（此時尚未 Commit）"""
        pass

    @abstractmethod
    def get_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """透過唯一標識符獲取 User"""
        pass

    @abstractmethod
    def get_by_email(self, email: Email) -> Optional[User]:
        """透過 Email 獲取 User。注意：這裡傳入的是強型別的 Email 值物件"""
        pass
