# ============================================================
# models.py — Pydantic-схемы запросов и ответов API
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional


class MaskRequest(BaseModel):
    text: str = Field(..., description="Текст для маскирования")
    session_id: str = Field(..., description="ID сессии (для хранения маппинга)")
    language: str = Field(default="ru", description="Язык текста: 'ru' или 'en'")


    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Иванов Иван Петрович, ИНН 772512345678, тел. +7-999-123-45-67 из ООО 'Рога и Копыта'",
                "session_id": "user_12345_req_001",
                "language": "ru"
            }
        }
    }


class MaskResponse(BaseModel):
    masked_text: str = Field(..., description="Текст с заменёнными PII-сущностями")
    entities_found: list[str] = Field(
        default_factory=list,
        description="Типы найденных PII-сущностей (без оригинальных значений)"
    )
    session_id: str
    ttl: int = Field(..., description="Время жизни маппинга сессии в секундах")


class UnmaskRequest(BaseModel):
    text: str = Field(..., description="Текст с PII-токенами для восстановления")
    session_id: str = Field(..., description="ID сессии, маппинг которой использовать")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "PERSON_1, ИНН INN_1, тел. PHONE_1 из ORG_1",
                "session_id": "user_12345_req_001"
            }
        }
    }


class UnmaskResponse(BaseModel):
    unmasked_text: str = Field(..., description="Текст с восстановленными оригинальными значениями")
    session_id: str
    tokens_replaced: int = Field(..., description="Количество заменённых токенов")


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
