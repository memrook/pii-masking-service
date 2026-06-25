# PII Masking Service

Микросервис для маскирования чувствительных данных в тексте перед отправкой во внешние LLM API. Работает как самостоятельный HTTP-сервис — подключается к любому проекту двумя строками кода.

## Как маскирует — гибрид

Чтобы внешняя LLM осмысленно работала с обезличенными данными (а не игнорировала «коды» вроде `PERSON_1`), применяется гибридная схема:

| Группа | Типы | Представление | Пример |
|--------|------|---------------|--------|
| **Суррогаты** (цифры/коды) | PHONE, INN, OGRN, EMAIL, PASSPORT, SNILS, CARD, VIN, ACCOUNT, BIK, KPP | синтаксически правдоподобные тестовые значения — LLM воспроизводит дословно | `+7 921 555-84-12`, `6591557797` |
| **Метки** (склоняемые/текстовые) | PERSON, ORG, ADDRESS | инертная метка в скобках | `[Имя 1]`, `[Организация 1]`, `[Адрес 1]` |

> Суррогаты — **не валидные реквизиты**, а правдоподобные тестовые значения. Метки оставлены метками, т.к. русская морфология (падежи) сломала бы точный обратный разбор реалистичного суррогата.

**Подсказка для LLM.** В ответе `/mask` есть поле `hint` (про метки `[Имя N]`). По умолчанию оно возвращается **отдельно** и НЕ вшито в `masked_text`. Клиент сам добавляет `hint` перед отправкой текста в LLM, если хочет.

Опционально можно передать `prepend_hint: true` — тогда подсказка вшивается в начало `masked_text`, а координаты `masked_spans` автоматически сдвигаются. Учтите: при этом прямой `unmask(masked_text)` вернёт текст с остатком подсказки (нарушение тождества `unmask(mask(text)) == text`). В боевом потоке `mask → LLM → unmask` это безопасно — LLM не повторяет подсказку в ответе, поэтому до `unmask` она не доходит. Если меток в результате нет — `prepend_hint` ничего не вшивает.

Поддерживаемые языки: **русский** (natasha NER + Presidio) и **английский** (spaCy + Presidio). Для английского из меток детектируется только `PERSON` (`[Name N]`).

## Контракт

- `/mask` принимает **исходный** текст; повторная обработка уже замаскированного текста не поддерживается.
- Один язык на `session_id` (фиксируется первым `/mask`; иначе `409`).
- Запросы одной сессии — **строго последовательны** (single-writer): и `/mask`, и `/unmask-chunk`.
- Если новый текст содержит ранее выданный этой сессией токен — `409` (нужна новая сессия).

## Быстрый старт

```bash
cp .env.example .env
# Задайте REDIS_PASSWORD и ОБЯЗАТЕЛЬНО SURROGATE_SECRET:
#   openssl rand -hex 32
# Один секрет на все воркеры — иначе round-trip сломается. Без секрета сервис не стартует.

docker compose up -d
# Swagger UI: http://localhost:6111/docs
```

## API

### `POST /mask`

Маскирует PII в тексте и сохраняет маппинг в Redis с TTL.

```bash
curl -X POST http://localhost:6111/mask \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Иванов Иван, ИНН 772512345678, тел. +7-999-123-45-67, ООО Ромашка",
    "session_id": "req_001",
    "language": "ru"
  }'
```

```json
{
  "masked_text": "[Имя 1], ИНН 6591557797, тел. +7 921 555-84-12, [Организация 1]",
  "entities_found": ["PERSON", "INN", "PHONE", "ORG"],
  "hint": "Примечание: фрагменты в квадратных скобках вида [Имя N] — это обезличенные реальные данные. Сохраняйте их в ответе дословно, без изменений.",
  "masked_spans": [{"start": 0, "end": 7, "type": "PERSON"}, "..."],
  "session_id": "req_001",
  "ttl": 300
}
```

`masked_spans` — диапазоны заменённых фрагментов в `masked_text`; индексы в **Unicode code points** (на стороне JS применять через `Array.from(masked_text)`). `hint` = `null`, если меток в результате нет.

### `POST /unmask`

Восстанавливает оригинальные значения из ответа LLM. Обратная замена — однопроходная (без каскада), `tokens_replaced` считает фактические вхождения.

```bash
curl -X POST http://localhost:6111/unmask \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Договор с [Имя 1] (ИНН: 6591557797) подписан.",
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

### `POST /unmask-chunk` *(streaming)*

Потоковое демаскирование для SSE-стримов. Буфер хвоста в Redis удерживает фрагмент, который может оказаться началом незавершённого токена (метка/суррогат разрезаны между чанками), до следующего вызова. Учитывает префиксную неоднозначность: короткий токен, являющийся префиксом более длинного, тоже удерживается.

```bash
# Чанк 1 — метка разрезана
curl -X POST http://localhost:6111/api/unmask-chunk \
  -H "Content-Type: application/json" \
  -d '{"text": "Договор с [Им", "session_id": "req_001", "is_final": false}'
# → {"unmasked_text": "Договор с ", "session_id": "req_001"}

# Чанк 2 — is_final=true сбрасывает буфер
curl -X POST http://localhost:6111/api/unmask-chunk \
  -H "Content-Type: application/json" \
  -d '{"text": "я 1] готов", "session_id": "req_001", "is_final": true}'
# → {"unmasked_text": "Иванов Иван готов", "session_id": "req_001"}
```

**Протокол:** сначала вызовите `/mask` — это создаёт сессию в Redis. Без предварительного `/mask` вернёт 404. При `is_final=true` хвостовой буфер (`pii:tail:{sid}`) удаляется.

**Идемпотентность в сессии:** несколько вызовов `/mask` в одной сессии переиспользуют тот же токен для того же оригинала (`"Иванов"` → `[Имя 1]` во всех вызовах; суррогат для одного ИНН стабилен в пределах сессии). Между разными сессиями суррогаты одного и того же PII **различаются** (HMAC-сид с `session_id`).

### `DELETE /session/{session_id}`

Явно удаляет маппинг и хвостовой буфер сессии до истечения TTL.

### `GET /stats`

Статистика: количество активных сессий и список поддерживаемых типов сущностей. Требует `X-API-Key` если авторизация включена.

### `GET /health`

Статус сервиса и Redis.

### `GET /docs`

Swagger UI с интерактивным тестированием всех эндпоинтов.

## Интеграция в проект

### Python (asyncio)

```python
import httpx

PII_URL = "http://pii-masking:6111"
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
| `SURROGATE_SECRET` | — | **Обязателен.** HMAC-секрет для суррогатов, один на все воркеры. Без него сервис не стартует. Генерация: `openssl rand -hex 32` |
| `MAPPING_TTL` | `300` | TTL маппинга в Redis, секунды |
| `API_KEY` | пусто | API Key для авторизации (пусто = без авторизации) |
| `API_WORKERS` | `2` | Число воркеров uvicorn |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `PII_PORT` | `6111` | Порт при standalone запуске |

Префиксы токенов тоже конфигурируемы: `TOKEN_PERSON`, `TOKEN_PHONE`, `TOKEN_INN` и т.д.

### Авторизация по API Key

Если `API_KEY` задан — все запросы (кроме `/health`) требуют заголовок `X-API-Key`:

```bash
curl -X POST http://localhost:6111/mask \
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
