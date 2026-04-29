# ==========================================
# 標準函式庫與型別提示 (Standard Library & Typing)
# ==========================================
import uuid
from typing import Protocol, runtime_checkable, Sequence

# 引入領域層核心實體與值物件
from app.domain.entities import AggregateRoot, User, Email

# ==========================================
# 基礎設施契約 (Infrastructure Contracts)
# ==========================================

# 💡 優化：採用 Python 3.12+ 專屬的型別參數語法 (PEP 695)
# 同時宣告實體泛型 [T] (受限於 AggregateRoot) 與主鍵泛型 [ID]。
# 這在架構層級定義了：持久化操作「僅能」針對聚合根進行，嚴禁越權操作子實體。
@runtime_checkable
class Repository[T: AggregateRoot, ID](Protocol):
    """
    💡 終極泛型倉儲協定 (Generic Repository Protocol)。
    實踐「依賴倒置原則 (DIP)」與「結構型別 (Structural Subtyping)」。
    基礎設施層 (如 SQLAlchemy, MongoDB) 在實作時無需顯式繼承，達成 100% 邏輯解耦。
    """
    
    async def add(self, entity: T) -> None:
        """
        將新建的聚合根加入倉儲記憶體中。
        💡 注意：此操作通常不立即觸發 DB 寫入，需配合 Unit of Work 統一 Commit。
        """
        ...

    async def get_by_id(self, entity_id: ID) -> T | None:
        """
        透過動態的主鍵型別 (ID) 獲取唯一的聚合根實體。
        使用 T | None 代替 Optional[T]，擁抱現代 Python 3.10+ 語法。
        """
        ...

    async def remove(self, entity: T) -> None:
        """
        從倉儲中移除該聚合根。
        """
        ...


@runtime_checkable
class UserRepository(Repository[User, uuid.UUID], Protocol):
    """
    User 聚合根的專屬倉儲介面 (Protocol)。
    完美繼承基底 CRUD 契約，並擴充 User 領域特有的查詢行為。
    """

    async def get_by_email(self, email: Email) -> User | None:
        """
        透過強型別的 Email 值物件獲取 User。
        這保證了查詢發起時，Email 格式已經過領域層的嚴格校驗。
        """
        ...

    # 💡 業務擴充建議：獲取特定狀態的使用者
    # async def get_suspended_users(self) -> Sequence[User]:
    #     """
    #     獲取系統中所有處於停權狀態的使用者。
    #     """
    #     ...
