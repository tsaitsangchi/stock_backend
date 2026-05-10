import uuid
from app.application.uow import UnitOfWork
from app.application.commands import RegisterUserCommand, UpdateProfileCommand, SuspendUserCommand
from app.domain.entities import User, Email, DomainError

# ==========================================
# 應用層例外 (Application Exceptions)
# ==========================================
class ApplicationError(Exception): """所有應用層錯誤的基底"""
class EmailAlreadyExistsError(ApplicationError): pass
class UserNotFoundError(ApplicationError): pass

# ==========================================
# 使用者相關應用服務 (User Application Services)
# ==========================================

class RegisterUserUseCase:
    """
    💡 業務情境：使用者註冊。
    協調領域模型與持久化層，確保註冊過程的原子性。
    """
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def execute(self, command: RegisterUserCommand) -> uuid.UUID:
        async with self.uow:
            # 1. 業務前置驗證
            email_vo = Email(command.email)
            if await self.uow.users.get_by_email(email_vo):
                raise EmailAlreadyExistsError(f"Email {command.email} 已被佔用")

            # 2. 調用領域工廠建立實體
            user = User.create_for_registration(
                email=command.email,
                hashed_password=command.password_hash,
                full_name=command.full_name
            )

            # 3. 持久化
            await self.uow.users.add(user)
            await self.uow.commit()

            # 4. 交易成功後的副作用處理
            user.mark_as_persisted()
            events = user.pop_events()
            # 💡 TODO: MessageBus.publish(events) -> 異步發送歡迎信、同步至 ElasticSearch 等
            
            return user.id


class UpdateProfileUseCase:
    """
    💡 業務情境：更新個人個資。
    展示「充血模型」與「樂觀鎖」的協作。
    """
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def execute(self, command: UpdateProfileCommand) -> None:
        async with self.uow:
            # 1. 重建聚合根
            user = await self.uow.users.get_by_id(command.user_id)
            if not user:
                raise UserNotFoundError(f"找不到使用者: {command.user_id}")

            # 2. 委派給領域實體執行業務邏輯 (封裝了狀態防禦與更新時間)
            user.update_full_name(command.new_name)

            # 3. 提交變更 (若版本衝突將拋出樂觀鎖例外)
            await self.uow.commit()

            # 4. 派發領域事件
            events = user.pop_events()
            # 💡 TODO: MessageBus.publish(events)


class SuspendUserUseCase:
    """
    💡 業務情境：帳號停權。
    封裝管理員對使用者權限的控制邏輯。
    """
    def __init__(self, uow: UnitOfWork):
        self.uow = uow

    async def execute(self, command: SuspendUserCommand) -> None:
        async with self.uow:
            user = await self.uow.users.get_by_id(command.user_id)
            if not user:
                raise UserNotFoundError(f"找不到使用者: {command.user_id}")

            # 執行停權業務行為
            user.suspend(reason=command.reason)

            await self.uow.commit()
            
            events = user.pop_events()
            # 💡 TODO: MessageBus.publish(events)