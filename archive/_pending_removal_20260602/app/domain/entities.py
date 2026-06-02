# ==========================================
# 標準函式庫引入區 (Standard Library Imports)
# ==========================================
import re  # 引入正則表達式，用於 Email 的格式校驗與長度攔截 (防止 ReDoS 攻擊)。
import uuid  # 用於生成全局唯一標識符 (UUID)。切斷實體對資料庫自動遞增 ID 的依賴，讓實體在記憶體誕生那一刻就具備身分主權。
from abc import ABC  # 引入抽象基底類別 (Abstract Base Class)，用於限制特定類別 (如事件、聚合根) 無法被直接實例化。
from dataclasses import dataclass, field, InitVar  # dataclass 核心工具：field 用於精細控制欄位行為，InitVar 用於接收初始化專用參數 (不保存為屬性)。
from datetime import datetime, timezone  # 時間與時區模組，強制全系統的時間戳記皆使用絕對一致的 UTC 時區。
from enum import StrEnum  # Python 3.11+ 的 StrEnum，定義強型別的業務狀態機，且在 JSON 或 DB 序列化時能直接作為字串使用。
from typing import Self, Callable, ClassVar  # 現代化型別提示：Self 代表類別自身，Callable 用於方法注入，ClassVar 用於嚴格標註類別變數。

# ==========================================
# 領域基礎與例外 (Domain Exceptions)
# ==========================================
# 💡 記憶體極限優化：為所有自訂例外加上 __slots__ = ()。
# 這會消除例外物件預設生成的 __dict__，在發生大量錯誤的極端壓測情境下，能省下可觀的記憶體並微幅提升效能。
class DomainError(Exception): __slots__ = () 
class InvalidEmailError(DomainError): __slots__ = () 
class InvalidProfileError(DomainError): __slots__ = () 
class InvalidPasswordHashError(DomainError): __slots__ = () 
class InvalidStateError(DomainError): __slots__ = ()  # 專門攔截違反狀態機規則的操作 (如：停權使用者試圖改名)

# ==========================================
# 系統層級與時間工具 (Domain Clock)
# ==========================================
def utc_now() -> datetime: 
    """
    獲取系統當下 UTC 時間。
    💡 架構考量：取代 lambda 函數。因為 Python 原生的 pickle 模組無法序列化 lambda，
    改用具名函數能確保「領域事件」可以被安全地推送到 Celery、RabbitMQ 或 Kafka 等非同步佇列中。
    """
    return datetime.now(timezone.utc) 

# ==========================================
# 領域枚舉 (Domain Enums)
# ==========================================
class UserStatus(StrEnum):
    """
    💡 業務狀態機：確保使用者的生命週期是受控的。
    """
    PENDING = "PENDING"       # 待完善資料/待驗證階段
    ACTIVE = "ACTIVE"         # 正常啟用階段
    SUSPENDED = "SUSPENDED"   # 停權階段 (此狀態下拒絕大部分業務操作)

# ==========================================
# 領域事件 (Domain Events)
# ==========================================
# frozen=True 確保事件一旦發生便不可篡改；slots=True 壓榨記憶體；kw_only=True 強制要求具名傳參以防錯位。
@dataclass(frozen=True, slots=True, kw_only=True) 
class DomainEvent(ABC): 
    # 允許外部注入時間，若無則預設呼叫 utc_now()。這解決了「實體更新時間」與「事件發生時間」撕裂的致命漏洞。
    occurred_on: datetime = field(default_factory=utc_now) 

@dataclass(frozen=True, slots=True, kw_only=True) 
class UserRegisteredEvent(DomainEvent): 
    user_id: uuid.UUID  # 紀錄發生事件的實體 ID
    email: str          # 供後續的 Email 寄送服務使用

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
    reason: str  # 紀錄停權的具體原因，供審計日誌 (Audit Log) 追蹤

@dataclass(frozen=True, slots=True, kw_only=True)
class UserReactivatedEvent(DomainEvent):
    user_id: uuid.UUID

# ==========================================
# 基礎設施：聚合根基底 (Aggregate Root Base)
# ==========================================
@dataclass(kw_only=True, slots=True) 
class AggregateRoot(ABC): 
    """
    💡 領域基礎設施：實踐 DRY (Don't Repeat Yourself) 原則。
    未來若有 Order, Wallet 等核心實體，只要繼承此類別就能自動獲得事件收集與發佈的能力。
    """
    # 事件佇列：init=False 表示不透過建構子賦值；repr=False 防止 print 時印出一大串事件造成日誌污染。
    _events: list[DomainEvent] = field(default_factory=list, init=False, repr=False) 

    def _record_event(self, event: DomainEvent) -> None: 
        """供實體內部的業務方法呼叫，將新產生的領域事件推入記憶體佇列。"""
        self._events.append(event) 
        
    def pop_events(self) -> list[DomainEvent]: 
        """供外部的 Unit of Work 呼叫，一次性取出並清空事件佇列，確保事件發佈的 Exactly-Once (精確一次) 語意。"""
        events = self._events[:] 
        self._events.clear() 
        return events 

# ==========================================
# 值物件 (Value Objects)
# ==========================================
@dataclass(frozen=True, slots=True) 
class Password: 
    # 隱藏 Hash 密碼：repr=False 防止在任何錯誤堆疊或 print 中意外洩漏。
    hashed_value: str = field(repr=False) 
    
    def __post_init__(self): 
        # 💡 安全邊界保護：初步過濾，防止開發者手滑傳入明文密碼 (如 "123456")。
        if len(self.hashed_value) < 30 or " " in self.hashed_value: 
            raise InvalidPasswordHashError("安全嚴重警告：密碼 Hash 長度異常或包含空白") 
            
        # 💡 演算法特徵校驗：確保流入領域層的 Hash 符合系統的安全演算法標準。
        valid_prefixes = ("$2", "$argon2", "pbkdf2:")
        if not self.hashed_value.startswith(valid_prefixes):
            raise InvalidPasswordHashError("安全嚴重警告：不支援的 Hash 演算法特徵")

    @classmethod 
    def from_trusted_db(cls, hashed_value: str) -> Self: 
        """
        💡 效能駭客通道：當資料是從資料庫 (可信來源) 讀取時，跳過耗時的 __post_init__ 校驗。
        這在批次撈取上萬筆資料時，能省下極大量的 CPU 時間。
        """
        obj = cls.__new__(cls) 
        object.__setattr__(obj, 'hashed_value', hashed_value)  # 突破 frozen=True 的限制強行賦值
        return obj 
            
    def matches(self, plain_password: str, verifier: Callable[[str, str], bool]) -> bool: 
        """💡 行為分離：驗證演算法不寫死在領域層，而是由外部 (如資安模組) 以函數指標 (Callable) 注入。"""
        return verifier(plain_password, self.hashed_value) 
        
    def __str__(self): 
        """無論如何轉字串，永遠只輸出星號，做到最嚴格的防護。"""
        return "********" 

@dataclass(frozen=True, slots=True) 
class Email: 
    address: str 
    # 💡 ClassVar：明確告知 MyPy 這是一個類別變數，只會在記憶體編譯一次，避免每次實例化都重新消耗 Regex 編譯資源。
    _REGEX: ClassVar[re.Pattern] = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$") 

    def __post_init__(self): 
        # 💡 防禦 Regex DoS (ReDoS) 攻擊：在執行耗時的正則比對前，先用極低成本的 len() 攔截惡意超長字串。
        if len(self.address) > 254:
            raise InvalidEmailError("Email 長度超過標準限制 (254字元)")

        normalized = self.address.lower().strip() 
        # 💡 效能微調：只有當真的有發生空白或大小寫轉換時，才呼叫底層的 __setattr__。
        if self.address != normalized:
            object.__setattr__(self, 'address', normalized) 
            
        if not self._REGEX.match(normalized): 
            raise InvalidEmailError(f"無效的 Email 格式: {normalized}") 

    @classmethod 
    def from_trusted_db(cls, address: str) -> Self: 
        """效能駭客通道：跳過所有驗證邏輯，專供 Repository 還原實體使用。"""
        obj = cls.__new__(cls) 
        object.__setattr__(obj, 'address', address) 
        return obj 

    def __str__(self): return self.address 

# ==========================================
# 領域實體 (Domain Entity)
# ==========================================
# 💡 weakref_slot=True 是維持 SQLAlchemy Identity Map (物件追蹤地圖) 不崩潰的深海級防護網。
@dataclass(kw_only=True, slots=True, weakref_slot=True) 
class User(AggregateRoot): 
    
    # ==========================================
    # 屬性封裝 - 終極封印防線
    # ==========================================
    # 使用 InitVar 接收初始化參數。它們不會成為物件的真實屬性，藉此徹底封殺了 user.email = "..." 這種非法賦值。
    id_init: InitVar[uuid.UUID | None] = None
    email_init: InitVar[Email | str]
    password_init: InitVar[Password | str]
    full_name_init: InitVar[str | None] = None 
    
    # 真正的內部記憶體狀態。init=False 代表不透過建構子初始化，全權交給我們手寫的 __post_init__ 掌控。
    _id: uuid.UUID = field(init=False, repr=True)
    _email: Email = field(init=False, repr=True)
    _password: Password = field(init=False, repr=False)
    
    # 業務狀態與樂觀鎖 (Optimistic Locking) 版號
    _status: UserStatus = field(default=UserStatus.PENDING, init=False)
    _version: int = field(default=0, init=False)
    
    # 持久化追蹤：讓 Unit of Work 知道該下 INSERT 還是 UPDATE 指令
    _is_persisted: bool = field(default=False, init=False, repr=False) 
    
    _full_name: str | None = field(default=None, init=False) 
    created_at: datetime | None = field(default=None, init=False) 
    updated_at: datetime | None = field(default=None, init=False) 

    # ==========================================
    # 初始化生命週期
    # ==========================================
    def __post_init__(self, id_init: uuid.UUID | None, email_init: Email | str, password_init: Password | str, full_name_init: str | None): 
        # 💡 靈魂鎖定：優先使用外部傳入的 ID (從 DB 撈取時)，若無則生成全新 UUID，保證身分的不可變性。
        self._id = id_init if id_init is not None else uuid.uuid4()
        
        # 防禦性轉型：確保存進實體的永遠是強型別的值物件 (Value Object)
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
        
        # 自動狀態推移：如果在初始化當下就提供了姓名，自動將帳號升級為啟用狀態。
        if self._status == UserStatus.PENDING and self._full_name:
            self._status = UserStatus.ACTIVE

    # ==========================================
    # 語意化工廠方法 (Factory Methods)
    # ==========================================
    @classmethod 
    def create_for_registration(cls, email: str, hashed_password: str, full_name: str) -> Self: 
        """情境 1：專用於系統接收註冊請求時，建立全新的使用者。"""
        user = cls(email_init=email, password_init=hashed_password, full_name_init=full_name) 
        # 紀錄註冊事件
        user._record_event(UserRegisteredEvent(user_id=user.id, email=str(user.email))) 
        return user 

    @classmethod 
    def from_database(cls, id: uuid.UUID, email: str, password_hash: str, 
                      created_at: datetime, updated_at: datetime, version: int,
                      status: UserStatus | str, full_name: str | None = None) -> Self: 
        """情境 2：專用於 Repository 從資料庫還原物件，完美擁抱型別提示與效能快速通道。"""
        user = cls( 
            id_init=id,
            email_init=Email.from_trusted_db(email), 
            password_init=Password.from_trusted_db(password_hash),
            full_name_init=None  # 刻意阻斷原本的校驗，於下方強制覆寫資料庫可信狀態
        ) 
        user._full_name = full_name 
        user.created_at = created_at
        user.updated_at = updated_at
        user._version = version
        user._status = UserStatus(status) if isinstance(status, str) else status
        user._is_persisted = True  # 標示為已持久化
        return user 

    # ==========================================
    # 內部輔助與 UoW 握手協定
    # ==========================================
    def mark_as_persisted(self): 
        """供 Unit of Work 成功 COMMIT 後呼叫，修正記憶體內物件狀態。"""
        self._is_persisted = True 

    def _mark_updated(self, current_time: datetime | None = None) -> datetime:
        """內部集中收斂：所有業務行為觸發時，必定推進樂觀鎖版號與更新時間。"""
        now = current_time or utc_now()
        self.updated_at = now
        self._version += 1
        return now

    # ==========================================
    # 核心業務邏輯 (Business Behaviors)
    # ==========================================
    def update_full_name(self, new_name: str, current_time: datetime | None = None): 
        # 💡 狀態防禦：業務規則明訂，停權使用者無法異動個資。
        if self._status == UserStatus.SUSPENDED:
            raise InvalidStateError("停權的使用者無法修改姓名")

        normalized_name = new_name.strip() 
        if not normalized_name or len(normalized_name) < 2: 
            raise InvalidProfileError("姓名長度不足或無效") 
        
        # 冪等性：若發生實質變更才處理
        if self._full_name != normalized_name: 
            self._full_name = normalized_name 
            now = self._mark_updated(current_time)
            
            # 狀態推移
            if self._status == UserStatus.PENDING:
                self._status = UserStatus.ACTIVE

            # 💡 時間一致性 (Time Consistency)：將相同的 `now` 塞入事件，杜絕審計追蹤發生時間撕裂。
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
        if self._status == UserStatus.SUSPENDED: return  # 已停權則直接返回
        self._status = UserStatus.SUSPENDED
        now = self._mark_updated(current_time)
        self._record_event(UserSuspendedEvent(user_id=self.id, reason=reason, occurred_on=now))

    def reactivate(self, current_time: datetime | None = None):
        if self._status != UserStatus.SUSPENDED: return 
        self._status = UserStatus.ACTIVE
        now = self._mark_updated(current_time)
        self._record_event(UserReactivatedEvent(user_id=self.id, occurred_on=now))

    # ==========================================
    # 唯讀存取器與比較協定 (Read-Only Properties & Protocols)
    # ==========================================
    # 透過 Property 對外完全封鎖賦值操作，徹底保護領域的不變性 (Invariants)。
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
        """
        💡 DDD 實體相等性：只要唯一標識符 (ID) 相同，即視為同一個實體。
        回傳 NotImplemented 是 Python 的優雅實踐，交由直譯器嘗試反向比對。
        """
        if not isinstance(other, User): return NotImplemented 
        return self.id == other.id 

    def __hash__(self) -> int: 
        """為支援將此物件作為 dict 的 key 或放入 set，透過唯一 ID 產出 Hash 值。"""
        return hash(self.id)