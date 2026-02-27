# PII Masking Service

Микросервис для маскирования чувствительных данных в тексте перед отправкой во внешние LLM API. Работает как самостоятельный HTTP-сервис — подключается к любому проекту двумя строками кода.

## Что маскирует

| Тип | Пример | Токен |
|-----|--------|-------|
| ФИО | Иванов Иван Петрович | `PERSON_1` |
| Телефон | +7-999-123-45-67 | `PHONE_1` |
| ИНН | 772512345678 | `INN_1` |
| ОГРН / ОГРНИП | 1027700132195 | `OGRN_1` |
| Организация | ООО «Рога и Копыта» | `ORG_1` |
| Email | ivan@company.ru | `EMAIL_1` |
| Паспорт РФ | 4510 123456 | `PASSPORT_1` |
| СНИЛС | 112-233-445 95 | `SNILS_1` |

Поддерживаемые языки: **русский** (natasha NER + Presidio) и **английский** (spaCy + Presidio).

## Быстрый старт

```bash
cp .env.example .env
# Задайте REDIS_PASSWORD в .env

docker compose up -d
# Swagger UI: http://localhost:8000/docs
```

## API

### `POST /mask`

Маскирует PII в тексте и сохраняет маппинг в Redis с TTL.

```bash
curl -X POST http://localhost:8000/mask \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Иванов Иван, ИНН 772512345678, тел. +7-999-123-45-67, ООО Ромашка",
    "session_id": "req_001",
    "language": "ru"
  }'
```

```json
{
  "masked_text": "PERSON_1, ИНН INN_1, тел. PHONE_1, ORG_1",
  "entities_found": ["PER", "INN_RU", "PHONE_RU", "ORG"],
  "session_id": "req_001"
}
```

### `POST /unmask`

Восстанавливает оригинальные значения из ответа LLM.

```bash
curl -X POST http://localhost:8000/unmask \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Договор с PERSON_1 (ИНН: INN_1) подписан.",
    "session_id": "req_001"
  }'
```

```json
{
  "unmasked_text": "Договор с Иванов Иван (ИНН: 772512345678) подписан.",
  "session_id": "req_001",
  "tokens_replaced": 2
}
```

### `DELETE /session/{session_id}`

Явно удаляет маппинг сессии до истечения TTL.

### `GET /health`

Статус сервиса и Redis.

### `GET /docs`

Swagger UI с интерактивным тестированием всех эндпоинтов.

## Интеграция в проект

### Python (asyncio)

```python
import httpx

PII_URL = "http://pii-masking:8000"
session_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"

# Перед отправкой в LLM
async with httpx.AsyncClient() as client:
    r = await client.post(f"{PII_URL}/mask",
        json={"text": prompt, "session_id": session_id})
    masked_prompt = r.json()["masked_text"]

# После получения ответа от LLM
    r = await client.post(f"{PII_URL}/unmask",
        json={"text": llm_response, "session_id": session_id})
    final_answer = r.json()["unmasked_text"]

    await client.delete(f"{PII_URL}/session/{session_id}")
```

### Подключение к Docker Compose проекту

**Вариант 1 — внешняя сеть** (сервис запущен отдельно):

```yaml
# docker-compose.yml вашего проекта
services:
  your-app:
    networks:
      - pii-net

networks:
  pii-net:
    external: true
    name: pii-masking-service_pii-net
```

**Вариант 2 — inline через override**:

```bash
docker compose -f docker-compose.yml -f docker-compose.pii-inline.yml up -d
```

```yaml
# docker-compose.pii-inline.yml
services:
  pii-masking:
    build: ../pii-masking-service
    networks: [pii-net]
  pii-redis:
    image: redis:7.2-alpine
    networks: [pii-net]
networks:
  pii-net:
    driver: bridge
```

## Конфигурация

Все параметры задаются через `.env`:

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `REDIS_PASSWORD` | — | Пароль Redis (обязательно) |
| `MAPPING_TTL` | `300` | TTL маппинга в Redis, секунды |
| `API_KEY` | пусто | API Key для авторизации (пусто = без авторизации) |
| `API_WORKERS` | `2` | Число воркеров uvicorn |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `PII_PORT` | `8000` | Порт при standalone запуске |

Префиксы токенов тоже конфигурируемы: `TOKEN_PERSON`, `TOKEN_PHONE`, `TOKEN_INN` и т.д.

### Авторизация по API Key

Если `API_KEY` задан — все запросы (кроме `/health`) требуют заголовок `X-API-Key`:

```bash
curl -X POST http://localhost:8000/mask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key" \
  -d '{"text": "...", "session_id": "...", "language": "ru"}'
```

В Swagger UI (`/docs`) нажмите кнопку **Authorize** (вверху справа) и введите ключ.

## Структура проекта

```
pii-masking-service/
├── app/
│   ├── main.py      — FastAPI приложение, эндпоинты, lifespan
│   ├── masker.py    — логика NER и маскирования (natasha + Presidio)
│   ├── models.py    — Pydantic схемы запросов/ответов
│   └── config.py    — настройки через pydantic-settings
├── Dockerfile
├── requirements.txt
├── docker-compose.yml   — standalone деплой (сервис + Redis)
└── .env.example
```

## Стек

- **FastAPI** + **uvicorn** — HTTP сервер
- **natasha** — Russian NER (ФИО, организации)
- **spaCy** (`ru_core_news_lg`, `en_core_web_lg`) — NLP backbone
- **Microsoft Presidio** — фреймворк анонимизации (телефоны, email, ИНН и др.)
- **Redis** — хранилище маппинга с TTL

## Важно

- Маппинг хранится **только в памяти Redis** с TTL (по умолчанию 5 минут), на диск не пишется
- Сервис **не логирует** оригинальные PII-значения, только типы найденных сущностей
- При `llm_provider=ollama` (self-hosted LLM) маскирование не нужно — данные не покидают инфраструктуру
