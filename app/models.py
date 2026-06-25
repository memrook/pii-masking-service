# ============================================================
# models.py — Pydantic-схемы запросов и ответов API
# ============================================================

from pydantic import BaseModel, Field


class MaskRequest(BaseModel):
    text: str = Field(..., description="Исходный текст для маскирования (НЕ уже замаскированный)")
    session_id: str = Field(..., description="ID сессии (для хранения маппинга)")
    language: str = Field(default="ru", description="Язык текста: 'ru' или 'en'. Фиксируется на всю сессию.")
    prepend_hint: bool = Field(
        default=False,
        description="Вшить подсказку про метки в начало masked_text. ВНИМАНИЕ: нарушает "
                    "прямой round-trip — unmask(masked_text) вернёт текст с остатком подсказки. "
                    "В боевом потоке mask→LLM→unmask это безопасно (LLM не повторяет подсказку). "
                    "Координаты masked_spans уже учитывают сдвиг. Если меток нет — ничего не вшивается."
    )


    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Иванов Иван Петрович, ИНН 772512345678, тел. +7-999-123-45-67 из ООО 'Рога и Копыта'",
                "session_id": "user_12345_req_001",
                "language": "ru",
                "prepend_hint": False
            }
        }
    }


class MaskedSpan(BaseModel):
    start: int = Field(..., description="Начало out в masked_text (Unicode code points)")
    end: int = Field(..., description="Конец out в masked_text (Unicode code points)")
    type: str = Field(..., description="Канонический тип сущности")


class MaskResponse(BaseModel):
    masked_text: str = Field(..., description="Текст с суррогатами/метками вместо PII")
    entities_found: list[str] = Field(
        default_factory=list,
        description="Канонические типы найденных PII-сущностей (без оригинальных значений)"
    )
    hint: str | None = Field(
        default=None,
        description="Подсказка для LLM про метки [Имя N]. None, если меток в результате нет. "
                    "Возвращается отдельно — клиент сам добавляет её перед отправкой в LLM."
    )
    masked_spans: list[MaskedSpan] = Field(
        default_factory=list,
        description="Диапазоны заменённых фрагментов в masked_text. Индексы — code points: "
                    "на стороне UI применять через Array.from(masked_text)."
    )
    session_id: str
    ttl: int = Field(..., description="Время жизни маппинга сессии в секундах")


class UnmaskRequest(BaseModel):
    text: str = Field(..., description="Текст с PII-токенами для восстановления")
    session_id: str = Field(..., description="ID сессии, маппинг которой использовать")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "[Имя 1], ИНН 6591557797, тел. +7 921 555-84-12 из [Организация 1]",
                "session_id": "user_12345_req_001"
            }
        }
    }


class UnmaskResponse(BaseModel):
    unmasked_text: str = Field(..., description="Текст с восстановленными оригинальными значениями")
    session_id: str
    tokens_replaced: int = Field(..., description="Количество заменённых токенов")


class UnmaskChunkRequest(BaseModel):
    text: str = Field(..., description="Фрагмент текста с PII-токенами (SSE chunk)")
    session_id: str = Field(..., description="ID сессии")
    is_final: bool = Field(default=False, description="True — последний чанк, сбросить буфер хвоста")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Договор с [Им",
                "session_id": "user_12345_req_001",
                "is_final": False,
            }
        }
    }


class UnmaskChunkResponse(BaseModel):
    unmasked_text: str = Field(..., description="Демаскированная безопасная часть чанка")
    session_id: str


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    redis: str
    models_loaded: bool
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    active_sessions: int
    supported_entity_types: list[str]
    language: str
