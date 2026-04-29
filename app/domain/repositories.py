import uuid
from typing import TypeVar, Protocol, runtime_checkable

# 引入我們先前完成的頂級實體與值物件
from app.domain.entities import AggregateRoot, User, Email

# 宣告泛型變數 T，嚴格限制必須是 AggregateRoot (聚合根) 的子類
T = TypeVar("T", bound=AggregateRoot)

@runtime_checkable
class Repository(Protocol[T]):
    """
    💡 終極進化：使用 Protocol 結構型別取代 ABC 名目型別。
    基礎設施層在實作時，甚至不需要 import 此類別來繼承，達到 100% 的物理與邏輯解耦。
    """
    
    async def add(self, entity: T) -> None:
        """將新建的聚合根加入倉儲（等待 UoW 統一 Commit）"""
        ...

    async def get_by_id(self, entity_id: uuid.UUID) -> T | None:
        """透過唯一標識符獲獲取聚合根"""
        ...

    async def remove(self, entity: T) -> None:
        """從倉儲中移除該聚合根"""
        ...


@runtime_checkable
class UserRepository(Repository[User], Protocol):
    """
    User 聚合根的專屬倉儲介面 (Protocol)。
    """

    async def get_by_email(self, email: Email) -> User | None:
        """
        透過 Email 值物件獲取 User。
        """
        ...
