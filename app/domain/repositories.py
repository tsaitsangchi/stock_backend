import uuid
from typing import Protocol, runtime_checkable

# 引入我們先前完成的頂級實體與值物件
from app.domain.entities import AggregateRoot, User, Email

# 💡 優化 1 & 2：採用 Python 3.12+ 專屬的型別參數語法 (PEP 695)
# 同時宣告實體泛型 [T] (必須是 AggregateRoot 的子類) 與主鍵泛型 [ID]
# 這宣告了：支援任意聚合根與其對應之主鍵型別的完美抽象
@runtime_checkable
class Repository[T: AggregateRoot, ID](Protocol):
    """
    💡 終極泛型倉儲協定 (Protocol)
    基礎設施層在實作時無需顯式繼承，達成 100% 結構化解耦。
    """
    
    async def add(self, entity: T) -> None:
        """將新建的聚合根加入倉儲（等待 UoW 統一 Commit）"""
        ...

    async def get_by_id(self, entity_id: ID) -> T | None:
        """透過動態的主鍵型別 (ID) 獲取聚合根"""
        ...

    async def remove(self, entity: T) -> None:
        """從倉儲中移除該聚合根"""
        ...


# 💡 在定義具體倉儲時，精確綁定 T 為 User，ID 為 uuid.UUID
@runtime_checkable
class UserRepository(Repository[User, uuid.UUID], Protocol):
    """
    User 聚合根的專屬倉儲介面 (Protocol)。
    完美繼承並型別化：add(User), get_by_id(uuid.UUID), remove(User)。
    """

    async def get_by_email(self, email: Email) -> User | None:
        """
        透過 Email 值物件獲取 User。
        """
        ...
