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
from fastapi import FastAPI, HTTPException, Security, Depends, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security.api_key import APIKeyHeader

from .config import settings
from .masker import load_models, mask_text, unmask_text, find_safe_boundary, is_models_loaded, _get_token_prefixes
from .models import (
    MaskRequest, MaskResponse,
    UnmaskRequest, UnmaskResponse,
    UnmaskChunkRequest, UnmaskChunkResponse,
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
    docs_url=None,
    redoc_url=None,
    root_path=settings.api_root_path,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------
# API Key авторизация (опционально)
# ----------------------------------------------------------
_openapi_prefix = settings.api_root_path.rstrip("/") if settings.api_root_path else ""


@app.get("/api/docs", include_in_schema=False)
async def custom_swagger_ui() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=f"{_openapi_prefix}/openapi.json",
        title=app.title + " — Swagger UI",
    )


@app.get("/api/redoc", include_in_schema=False)
async def custom_redoc() -> HTMLResponse:
    return get_redoc_html(
        openapi_url=f"{_openapi_prefix}/openapi.json",
        title=app.title + " — ReDoc",
    )


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def check_api_key(api_key: str | None = Security(api_key_header)):
    if not settings.api_key:
        return  # авторизация отключена
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")


# ----------------------------------------------------------
# API роутер (все эндпоинты под /api)
# ----------------------------------------------------------
api = APIRouter(prefix="/api")


@api.get("/health", response_model=HealthResponse, tags=["System"])
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


@api.get("/stats", response_model=StatsResponse, tags=["System"])
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
                             "CARD", "VIN", "ACCOUNT", "BIK", "KPP"],
        language="ru, en",
    )


@api.post("/mask", response_model=MaskResponse, tags=["Masking"])
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

    # Загружаем существующий маппинг ДО маскирования для stateful allocation
    redis_key = f"pii:mapping:{request.session_id}"
    existing_raw = await redis.get(redis_key)
    existing_mapping: dict[str, str] = json.loads(existing_raw) if existing_raw else {}

    try:
        masked_text, mapping, entity_types = mask_text(
            text=request.text,
            session_id=request.session_id,
            language=request.language,
            existing_mapping=existing_mapping,
        )
    except Exception as e:
        logger.error("mask.error", session_id=request.session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Ошибка маскирования: {e}")

    # Сохраняем полный маппинг в Redis с TTL (всегда, даже при пустом mapping)
    await redis.setex(redis_key, settings.mapping_ttl, json.dumps(mapping, ensure_ascii=False))

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
        ttl=settings.mapping_ttl,
    )


@api.post("/unmask", response_model=UnmaskResponse, tags=["Masking"])
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


@api.post("/unmask-chunk", response_model=UnmaskChunkResponse, tags=["Masking"])
async def unmask_chunk(
    request: UnmaskChunkRequest,
    redis: aioredis.Redis = Depends(get_redis),
    _=Depends(check_api_key),
):
    """
    Потоковое демаскирование SSE-чанков с буфером хвоста.

    Алгоритм:
    1. Читает mapping сессии (404 если не было /mask).
    2. Читает буферный хвост из предыдущего чанка.
    3. Объединяет хвост + новый чанк.
    4. Находит безопасную границу (хвост может содержать неполный токен).
    5. Демаскирует безопасную часть, сохраняет новый хвост.
    6. На is_final=True сбрасывает всё, удаляет хвост.
    """
    mapping_key = f"pii:mapping:{request.session_id}"
    tail_key = f"pii:tail:{request.session_id}"

    mapping_raw = await redis.get(mapping_key)
    if mapping_raw is None:
        raise HTTPException(status_code=404, detail="Сессия не найдена. Сначала вызовите /mask.")

    mapping: dict[str, str] = json.loads(mapping_raw)
    tail: str = (await redis.get(tail_key)) or ""

    combined = tail + request.text

    if request.is_final:
        safe_end = len(combined)
        await redis.delete(tail_key)
    else:
        prefixes = _get_token_prefixes()
        safe_end = find_safe_boundary(combined, prefixes)
        new_tail = combined[safe_end:]
        if new_tail:
            await redis.setex(tail_key, settings.mapping_ttl, new_tail)
        else:
            await redis.delete(tail_key)

    safe_text = combined[:safe_end]
    unmasked, _ = unmask_text(safe_text, mapping)

    return UnmaskChunkResponse(unmasked_text=unmasked, session_id=request.session_id)


@api.delete("/session/{session_id}", response_model=SessionDeleteResponse, tags=["Session"])
async def delete_session(
    session_id: str,
    redis: aioredis.Redis = Depends(get_redis),
    _=Depends(check_api_key),
):
    """Явно удаляет маппинг и буферный хвост сессии из Redis (до истечения TTL)."""
    deleted_map = await redis.delete(f"pii:mapping:{session_id}")
    deleted_tail = await redis.delete(f"pii:tail:{session_id}")
    return SessionDeleteResponse(session_id=session_id, deleted=bool(deleted_map or deleted_tail))


app.include_router(api)

# ----------------------------------------------------------
# UI — single-file, no static mount needed
# ----------------------------------------------------------
_index_html = pathlib.Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(_index_html, media_type="text/html")


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
    )
