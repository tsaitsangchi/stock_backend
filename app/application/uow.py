from typing import Protocol, Self, runtime_checkable
from types import TracebackType
from app.domain.repositories import UserRepository

@runtime_checkable
class UnitOfWork(Protocol):
    """
    💡 應用層工作單元協定 (Unit of Work Protocol)。
    負責劃定原子性操作的交易邊界。實踐「所有成功或所有失敗」的 ACID 語意。
    應用層僅依賴此協定，完全隔離底層持久化技術（如 SQLAlchemy 或 MongoDB）。
    """
    
    # 宣告工作單元擁有的倉儲存取權
    users: UserRepository

    async def __aenter__(self) -> Self:
        """開啟交易邊界 (Transaction Start)"""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        離開 with 區塊時的自動化管理。
        若過程中發生 Exception，底層實作必須執行 rollback()。
        """
        ...

    async def commit(self) -> None:
        """
        提交交易。
        💡 注意：若發生並發衝突，此處應由底層拋出樂觀鎖例外 (ConcurrencyError)。
        """
        ...

    async def rollback(self) -> None:
        """回滾交易，還原至交易開始前的狀態"""
        ...