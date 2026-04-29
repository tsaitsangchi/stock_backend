import uuid
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence

# 引入我們先前完成的頂級實體與值物件
from app.domain.entities import AggregateRoot, User, Email

# 💡 優化 1：宣告泛型變數 T，並且嚴格限制 T 必須是 AggregateRoot (聚合根) 的子類
# 這在架構上宣告了：我們「只允許」對聚合根進行持久化，嚴禁對其內部的子實體直接開 Repository
T = TypeVar("T", bound=AggregateRoot)

class Repository(Generic[T], ABC):
    """
    💡 泛型倉儲介面 (Generic Repository Interface)
    將所有聚合根共用的基本 CRUD 契約抽離，實踐極致的 DRY 原則。
    """
    
    @abstractmethod
    async def add(self, entity: T) -> None:
        """將新建的聚合根加入倉儲（等待 UoW 統一 Commit）"""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: uuid.UUID) -> T | None:
        """透過唯一標識符獲獲取聚合根 (💡 優化 3：使用 T | None 取代 Optional[T])"""
        pass

    @abstractmethod
    async def remove(self, entity: T) -> None:
        """從倉儲中移除該聚合根"""
        pass


class UserRepository(Repository[User], ABC):
    """
    User 聚合根的專屬倉儲介面。
    已自動繼承 add, get_by_id, remove 等方法，此處僅需定義 User 專屬的業務查詢。
    """

    # 💡 優化 2：全面非同步化 (AsyncIO)，為底層的高併發資料庫驅動 (如 asyncpg) 鋪路
    @abstractmethod
    async def get_by_email(self, email: Email) -> User | None:
        """
        透過 Email 值物件獲取 User。
        確保查詢在編譯期就能獲得 Email 格式正確的保障。
        """
        pass
