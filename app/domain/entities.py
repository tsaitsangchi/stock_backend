import re
import uuid
from abc import ABC
from dataclasses import dataclass, field, InitVar
from datetime import datetime, timezone
# 💡 優化 2：引入 StrEnum 與 ClassVar
from enum import StrEnum
from typing import Self, Callable, ClassVar

# ==========================================
# 領域基礎與例外 (Domain Exceptions)
# ==========================================
# 💡 基礎架構優化：為所有例外類別加上 __slots__ = ()。
class DomainError(Exception): __slots__ = () 
class InvalidEmailError(DomainError): __slots__ = () 
class InvalidProfileError(DomainError): __slots__ = () 
class InvalidPasswordHashError(DomainError): __slots__ = () 
class InvalidStateError(DomainError): __slots__ = () 

# 封裝系統時間的獲取。
def utc_now() -> datetime: 
    return datetime.now(timezone.utc) 

# ==========================================
# 領域枚舉 (Domain Enums)
# ==========================================
# 💡 優化 2：使用現代化的 StrEnum (Python 3.11+)
class UserStatus(StrEnum):
    PENDING = "PENDING"       # 待完善資料/待驗證
    ACTIVE = "ACTIVE"         # 正常啟用
    SUSPENDED = "SUSPENDED"   # 停權帳戶

# ==========================================
# 領域事件 (Domain Events)
# ==========================================
@dataclass(frozen=True, slots=True, kw_only=True) 
class DomainEvent(ABC): 
    occurred_on: datetime = field(default_factory=utc_now) 

@dataclass(frozen=True, slots=True, kw_only=True) 
class UserRegisteredEvent(DomainEvent): 
    user_id: uuid.UUID 
    email: str 

@dataclass(frozen=True, slots=True, kw_only=True) 
class ProfileUpdatedEvent(DomainEvent): 
    user_id: uuid.UUID 
    new_name: str 

@dataclass(frozen=True, slots=True, kw_only=True)
class PasswordChangedEvent(DomainEvent):
    user_id: uuid.UUID

@dataclass(frozen=True, slots=True, kw_only=True)
class UserSuspendedEvent(DomainEvent):
    user_id: uuid.UUID
    reason: str

@dataclass(frozen=True, slots=True, kw_only=True)
class UserReactivatedEvent(DomainEvent):
    user_id: uuid.UUID

# ==========================================
# 基礎設施：聚合根 (Aggregate Root)
# ==========================================
@dataclass(kw_only=True, slots=True) 
class AggregateRoot(ABC): 
    _events: list[DomainEvent] = field(default_factory=list, init=False, repr=False) 

    def _record_event(self, event: DomainEvent) -> None: 
        self._events.append(event) 
        
    def pop_events(self) -> list[DomainEvent]: 
        events = self._events[:] 
        self._events.clear() 
        return events 

# ==========================================
# 值物件 (Value Objects)
# ==========================================
@dataclass(frozen=True, slots=True) 
class Password: 
    hashed_value: str = field(repr=False) 
    
    def __post_init__(self): 
        if len(self.hashed_value) < 30 or " " in self.hashed_value: 
            raise InvalidPasswordHashError("安全嚴重警告：密碼 Hash 長度異常或包含空白") 
        valid_prefixes = ("$2", "$argon2", "pbkdf2:")
        if not self.hashed_value.startswith(valid_prefixes):
            raise InvalidPasswordHashError("安全嚴重警告：不支援的 Hash 演算法特徵")

    @classmethod 
    def from_trusted_db(cls, hashed_value: str) -> Self: 
        obj = cls.__new__(cls) 
        object.__setattr__(obj, 'hashed_value', hashed_value) 
        return obj 
            
    def matches(self, plain_password: str, verifier: Callable[[str, str], bool]) -> bool: 
        return verifier(plain_password, self.hashed_value) 
        
    def __str__(self): return "********" 

@dataclass(frozen=True, slots=True) 
class Email: 
    address: str 
    # 💡 優化 3：明確標示為類別變數 ClassVar，滿足極致嚴格的靜態型別檢查
    _REGEX: ClassVar[re.Pattern] = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$") 

    def __post_init__(self): 
        if len(self.address) > 254:
            raise InvalidEmailError("Email 長度超過標準限制 (254字元)")

        normalized = self.address.lower().strip() 
        if self.address != normalized:
            object.__setattr__(self, 'address', normalized) 
            
        if not self._REGEX.match(normalized): 
            raise InvalidEmailError(f"無效的 Email 格式: {normalized}") 

    @classmethod 
    def from_trusted_db(cls, address: str) -> Self: 
        obj = cls.__new__(cls) 
        object.__setattr__(obj, 'address', address) 
        return obj 

    def __str__(self): return self.address 

# ==========================================
# 領域實體 (Domain Entity)
# ==========================================
@dataclass(kw_only=True, slots=True, weakref_slot=True) 
class User(AggregateRoot): 
    # 💡 優化 1：終極封印！將所有影響一致性的欄位改為 InitVar，徹底防堵外部直接賦值 [cite: 318, 391]
    id_init: InitVar[uuid.UUID | None] = None
    email_init: InitVar[Email | str]
    password_init: InitVar[Password | str]
    full_name_init: InitVar[str | None] = None 
    
    # 真正的內部記憶體狀態 (init=False 代表不透過建構子初始化)
    _id: uuid.UUID = field(init=False, repr=True)
    _email: Email = field(init=False, repr=True)
    _password: Password = field(init=False, repr=False)
    _status: UserStatus = field(default=UserStatus.PENDING, init=False)
    _version: int = field(default=0, init=False)
    _is_persisted: bool = field(default=False, init=False, repr=False) 
    _full_name: str | None = field(default=None, init=False) 
    created_at: datetime | None = field(default=None, init=False) 
    updated_at: datetime | None = field(default=None, init=False) 

    def __post_init__(self, id_init: uuid.UUID | None, email_init: Email | str, password_init: Password | str, full_name_init: str | None): 
        # 靈魂鎖定
        self._id = id_init if id_init is not None else uuid.uuid4()
        # 轉型與校驗
        self._email = Email(email_init) if isinstance(email_init, str) else email_init
        self._password = Password(password_init) if isinstance(password_init, str) else password_init
        
        if full_name_init is not None: 
            normalized_name = full_name_init.strip() 
            if len(normalized_name) < 2: 
                raise InvalidProfileError("姓名長度不足") 
            self._full_name = normalized_name 
            
        now = utc_now() 
        self.created_at = now 
        self.updated_at = now 
        
        # 自動狀態推移
        if self._status == UserStatus.PENDING and self._full_name:
            self._status = UserStatus.ACTIVE

    # ==========================================
    # 語意化工廠方法 (Factory Methods)
    # ==========================================
    @classmethod 
    def create_for_registration(cls, email: str, hashed_password: str, full_name: str) -> Self: 
        user = cls(email_init=email, password_init=hashed_password, full_name_init=full_name) 
        user._record_event(UserRegisteredEvent(user_id=user.id, email=str(user.email))) 
        return user 

    @classmethod 
    def from_database(cls, id: uuid.UUID, email: str, password_hash: str, 
                      created_at: datetime, updated_at: datetime, version: int,
                      status: UserStatus | str, full_name: str | None = None) -> Self: 
        user = cls( 
            id_init=id,
            email_init=Email.from_trusted_db(email), 
            password_init=Password.from_trusted_db(password_hash),
            full_name_init=None # 繞過 __post_init__ 校驗，下方強制覆寫可信資料
        ) 
        user._full_name = full_name 
        user.created_at = created_at
        user.updated_at = updated_at
        user._version = version
        user._status = UserStatus(status) if isinstance(status, str) else status
        user._is_persisted = True 
        return user 

    def mark_as_persisted(self): 
        self._is_persisted = True 

    def _mark_updated(self, current_time: datetime | None = None) -> datetime:
        now = current_time or utc_now()
        self.updated_at = now
        self._version += 1
        return now

    # ==========================================
    # 核心業務邏輯 (Business Behaviors)
    # ==========================================
    def update_full_name(self, new_name: str, current_time: datetime | None = None): 
        if self._status == UserStatus.SUSPENDED:
            raise InvalidStateError("停權的使用者無法修改姓名")

        normalized_name = new_name.strip() 
        if not normalized_name or len(normalized_name) < 2: 
            raise InvalidProfileError("姓名長度不足或無效") 
        
        if self._full_name != normalized_name: 
            self._full_name = normalized_name 
            now = self._mark_updated(current_time)
            
            if self._status == UserStatus.PENDING:
                self._status = UserStatus.ACTIVE

            self._record_event(ProfileUpdatedEvent( 
                user_id=self.id, 
                new_name=normalized_name, 
                occurred_on=now 
            )) 

    def change_password(self, new_hashed_password: str, current_time: datetime | None = None):
        if self._status == UserStatus.SUSPENDED:
            raise InvalidStateError("停權的使用者無法變更密碼")
            
        self._password = Password(new_hashed_password)
        now = self._mark_updated(current_time)
        self._record_event(PasswordChangedEvent(user_id=self.id, occurred_on=now))

    def suspend(self, reason: str, current_time: datetime | None = None):
        if self._status == UserStatus.SUSPENDED: return 
        self._status = UserStatus.SUSPENDED
        now = self._mark_updated(current_time)
        self._record_event(UserSuspendedEvent(user_id=self.id, reason=reason, occurred_on=now))

    def reactivate(self, current_time: datetime | None = None):
        if self._status != UserStatus.SUSPENDED: return 
        self._status = UserStatus.ACTIVE
        now = self._mark_updated(current_time)
        self._record_event(UserReactivatedEvent(user_id=self.id, occurred_on=now))

    # ==========================================
    # 唯讀存取器 (Read-Only Properties)
    # ==========================================
    @property
    def id(self) -> uuid.UUID: return self._id

    @property
    def email(self) -> Email: return self._email

    @property
    def password(self) -> Password: return self._password

    @property 
    def full_name(self) -> str | None: return self._full_name 

    @property
    def status(self) -> UserStatus: return self._status

    @property
    def version(self) -> int: return self._version

    def is_persisted(self) -> bool: return self._is_persisted 

    @property 
    def is_profile_complete(self) -> bool: 
        return self._full_name is not None and len(self._full_name) >= 2 

    def __eq__(self, other: object) -> bool: 
        if not isinstance(other, User): return NotImplemented 
        return self.id == other.id 

    def __hash__(self) -> int: return hash(self.id) 