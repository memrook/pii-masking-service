# ============================================================
# main.py — FastAPI приложение PII Masking Service
# ============================================================

import json
import logging
import pathlib
import time
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from .config import settings
from .masker import load_models, mask_text, unmask_text, is_models_loaded
from .models import (
    MaskRequest, MaskResponse,
    UnmaskRequest, UnmaskResponse,
    SessionDeleteResponse, HealthResponse, StatsResponse,
)

# ----------------------------------------------------------
# Логирование
# ----------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()
logging.basicConfig(level=settings.log_level)


# ----------------------------------------------------------
# Redis клиент
# ----------------------------------------------------------
_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
        )
    return _redis


# ----------------------------------------------------------
# Lifespan: загружаем модели при старте
# ----------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("pii_masking_service.startup", message="Загрузка NLP моделей...")
    start = time.time()
    load_models()
    logger.info("pii_masking_service.ready", elapsed=round(time.time() - start, 2))
    yield
    # Shutdown
    if _redis:
        await _redis.aclose()
    logger.info("pii_masking_service.shutdown")


# ----------------------------------------------------------
# FastAPI приложение
# ----------------------------------------------------------
app = FastAPI(
    title="PII Masking Service",
    description="Маскирование чувствительных данных (ФИО, телефоны, ИНН, организации) для русского и английского текста.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    root_path=settings.api_root_path,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/ui/")


# ----------------------------------------------------------
# API Key авторизация (опционально)
# ----------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def check_api_key(api_key: str | None = Security(api_key_header)):
    if not settings.api_key:
        return  # авторизация отключена
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")


# ----------------------------------------------------------
# Эндпоинты
# ----------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(redis: aioredis.Redis = Depends(get_redis)):
    """Проверка состояния сервиса."""
    try:
        await redis.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    return HealthResponse(
        status="ok" if redis_status == "ok" and is_models_loaded() else "degraded",
        redis=redis_status,
        models_loaded=is_models_loaded(),
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats(
    redis: aioredis.Redis = Depends(get_redis),
    _=Depends(check_api_key),
):
    """Статистика: активные сессии, поддерживаемые типы сущностей."""
    keys = await redis.keys("pii:mapping:*")
    return StatsResponse(
        active_sessions=len(keys),
        supported_entity_types=["PERSON", "PHONE", "INN", "OGRN", "ORG",
                             "EMAIL", "PASSPORT", "SNILS", "ADDRESS",
                             "CARD", "VIN"],
        language="ru, en",
    )


@app.post("/mask", response_model=MaskResponse, tags=["Masking"])
async def mask(
    request: MaskRequest,
    redis: aioredis.Redis = Depends(get_redis),
    _=Depends(check_api_key),
):
    """
    Маскирует PII в тексте и сохраняет маппинг в Redis с TTL.

    - Заменяет ФИО → PERSON_N
    - Заменяет телефоны → PHONE_N
    - Заменяет ИНН → INN_N
    - Заменяет ОГРН → OGRN_N
    - Заменяет организации → ORG_N
    - Заменяет email → EMAIL_N
    - Заменяет паспорта → PASSPORT_N
    - Заменяет СНИЛС → SNILS_N
    """
    if not is_models_loaded():
        raise HTTPException(status_code=503, detail="Модели ещё загружаются")

    try:
        masked_text, mapping, entity_types = mask_text(
            text=request.text,
            session_id=request.session_id,
            language=request.language,
        )
    except Exception as e:
        logger.error("mask.error", session_id=request.session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка маскирования: {e}")

    # Сохраняем маппинг в Redis с TTL
    redis_key = f"pii:mapping:{request.session_id}"
    if mapping:
        # Загружаем существующий маппинг (если сессия уже есть) и мерджим
        existing_raw = await redis.get(redis_key)
        existing = json.loads(existing_raw) if existing_raw else {}
        existing.update(mapping)
        await redis.setex(redis_key, settings.mapping_ttl, json.dumps(existing, ensure_ascii=False))

    logger.info(
        "mask.success",
        session_id=request.session_id,
        entities_found=entity_types,
        tokens_created=len(mapping),
    )

    return MaskResponse(
        masked_text=masked_text,
        entities_found=entity_types,
        session_id=request.session_id,
    )


@app.post("/unmask", response_model=UnmaskResponse, tags=["Masking"])
async def unmask(
    request: UnmaskRequest,
    redis: aioredis.Redis = Depends(get_redis),
    _=Depends(check_api_key),
):
    """
    Восстанавливает оригинальные значения из токенов в тексте.
    Использует маппинг, сохранённый ранее командой /mask.
    """
    redis_key = f"pii:mapping:{request.session_id}"
    raw = await redis.get(redis_key)

    if not raw:
        # Маппинг не найден — возвращаем текст как есть
        logger.warning("unmask.mapping_not_found", session_id=request.session_id)
        return UnmaskResponse(
            unmasked_text=request.text,
            session_id=request.session_id,
            tokens_replaced=0,
        )

    mapping: dict[str, str] = json.loads(raw)
    unmasked_text, tokens_replaced = unmask_text(request.text, mapping)

    logger.info("unmask.success", session_id=request.session_id, tokens_replaced=tokens_replaced)

    return UnmaskResponse(
        unmasked_text=unmasked_text,
        session_id=request.session_id,
        tokens_replaced=tokens_replaced,
    )


@app.delete("/session/{session_id}", response_model=SessionDeleteResponse, tags=["Session"])
async def delete_session(
    session_id: str,
    redis: aioredis.Redis = Depends(get_redis),
    _=Depends(check_api_key),
):
    """Явно удаляет маппинг сессии из Redis (до истечения TTL)."""
    redis_key = f"pii:mapping:{session_id}"
    deleted = await redis.delete(redis_key)
    return SessionDeleteResponse(session_id=session_id, deleted=bool(deleted))


# ----------------------------------------------------------
# Static UI — must be mounted LAST to avoid intercepting API routes
# ----------------------------------------------------------
_static_dir = pathlib.Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="ui")


# ----------------------------------------------------------
# Точка входа
# ----------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        root_path=settings.api_root_path,
    )
