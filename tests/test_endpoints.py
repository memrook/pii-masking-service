# tests/test_endpoints.py — интеграционные тесты endpoints через ASGI + fake Redis
import json
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app, get_redis
from app import masker
from tests.helpers import FakeDoc, FakeNERTagger, FakePresidioAnalyzer, FakeSpan


# ----------------------------------------------------------
# Fake Redis
# ----------------------------------------------------------

class FakeRedis:
    """Минимальная in-memory реализация Redis для тестов."""

    def __init__(self):
        self._store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                deleted += 1
        return deleted

    async def ttl(self, key: str) -> int:
        return 300 if key in self._store else -2

    async def ping(self) -> bool:
        return True

    async def keys(self, pattern: str) -> list[str]:
        return list(self._store.keys())


# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------

@pytest.fixture
def fake_redis():
    return FakeRedis()


@pytest_asyncio.fixture
async def client(fake_redis, monkeypatch):
    """AsyncClient с подменённым Redis и заглушками NLP."""
    monkeypatch.setattr(masker, "_models_loaded", True)
    monkeypatch.setattr(masker, "_segmenter", object())
    monkeypatch.setattr(masker, "Doc", FakeDoc)
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger())
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer())

    app.dependency_overrides[get_redis] = lambda: fake_redis
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


# ----------------------------------------------------------
# /mask endpoint tests
# ----------------------------------------------------------

@pytest.mark.asyncio
async def test_mask_endpoint_creates_empty_mapping_when_no_pii(client, fake_redis):
    """/mask без PII → Redis ключ всё равно создаётся с {}."""
    resp = await client.post("/api/mask", json={
        "text": "привет как дела",
        "session_id": "sess1",
        "language": "ru",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess1"
    assert data["entities_found"] == []

    stored = await fake_redis.get("pii:mapping:sess1")
    assert stored is not None
    assert json.loads(stored) == {}


@pytest.mark.asyncio
async def test_mask_endpoint_persists_continuous_tokens_in_session(client, fake_redis, monkeypatch):
    """Два /mask в одной сессии → PERSON_1 и PERSON_2, оба в Redis."""
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
    ]))
    resp1 = await client.post("/api/mask", json={
        "text": "Иванов пришёл",
        "session_id": "sess2",
        "language": "ru",
    })
    assert resp1.status_code == 200

    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Петров"),
    ]))
    resp2 = await client.post("/api/mask", json={
        "text": "Петров уходит",
        "session_id": "sess2",
        "language": "ru",
    })
    assert resp2.status_code == 200

    stored = json.loads(await fake_redis.get("pii:mapping:sess2"))
    assert stored.get("PERSON_1") == "Иванов"
    assert stored.get("PERSON_2") == "Петров"


@pytest.mark.asyncio
async def test_mask_endpoint_reuses_token_for_same_person(client, fake_redis, monkeypatch):
    """Одинаковый original в двух /mask → один и тот же токен."""
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
    ]))
    await client.post("/api/mask", json={"text": "Иванов", "session_id": "sess3", "language": "ru"})

    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
    ]))
    resp2 = await client.post("/api/mask", json={"text": "Иванов снова", "session_id": "sess3", "language": "ru"})
    assert resp2.json()["masked_text"] == "PERSON_1 снова"

    stored = json.loads(await fake_redis.get("pii:mapping:sess3"))
    assert list(stored.keys()) == ["PERSON_1"]


# ----------------------------------------------------------
# /unmask-chunk endpoint tests
# ----------------------------------------------------------

@pytest.mark.asyncio
async def test_unmask_chunk_missing_session_returns_404(client):
    """/unmask-chunk без предварительного /mask → 404."""
    resp = await client.post("/api/unmask-chunk", json={
        "text": "любой текст",
        "session_id": "nonexistent",
        "is_final": True,
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_unmask_chunk_split_token_across_chunks(client, fake_redis, monkeypatch):
    """Токен разрезан между чанками: 'PERSON' + '_1 готов' → 'Иванов готов'."""
    await fake_redis.setex("pii:mapping:sess4", 300, json.dumps({"PERSON_1": "Иванов"}))

    resp1 = await client.post("/api/unmask-chunk", json={
        "text": "Договор с PERSON",
        "session_id": "sess4",
        "is_final": False,
    })
    assert resp1.status_code == 200
    assert resp1.json()["unmasked_text"] == "Договор с "

    resp2 = await client.post("/api/unmask-chunk", json={
        "text": "_1 готов",
        "session_id": "sess4",
        "is_final": True,
    })
    assert resp2.status_code == 200
    assert resp2.json()["unmasked_text"] == "Иванов готов"


@pytest.mark.asyncio
async def test_unmask_chunk_empty_mapping_passthrough(client, fake_redis):
    """Сессия с пустым маппингом → текст возвращается как есть, 200 OK."""
    await fake_redis.setex("pii:mapping:sess5", 300, json.dumps({}))

    resp = await client.post("/api/unmask-chunk", json={
        "text": "обычный текст без токенов",
        "session_id": "sess5",
        "is_final": True,
    })
    assert resp.status_code == 200
    assert resp.json()["unmasked_text"] == "обычный текст без токенов"


@pytest.mark.asyncio
async def test_unmask_chunk_is_final_removes_tail_key(client, fake_redis):
    """После is_final=True ключ pii:tail:{sid} должен отсутствовать."""
    await fake_redis.setex("pii:mapping:sess6", 300, json.dumps({"PERSON_1": "Иванов"}))
    await fake_redis.setex("pii:tail:sess6", 300, "PERSON")

    await client.post("/api/unmask-chunk", json={
        "text": "_1",
        "session_id": "sess6",
        "is_final": True,
    })

    tail = await fake_redis.get("pii:tail:sess6")
    assert tail is None


@pytest.mark.asyncio
async def test_unmask_chunk_buffers_partial_token(client, fake_redis):
    """Частичный токен в конце чанка уходит в буфер, не в ответ."""
    await fake_redis.setex("pii:mapping:sess7", 300, json.dumps({"PERSON_1": "Иванов"}))

    resp = await client.post("/api/unmask-chunk", json={
        "text": "текст PERS",
        "session_id": "sess7",
        "is_final": False,
    })
    assert resp.status_code == 200
    # "PERS" осталось в буфере, не в ответе
    assert "PERS" not in resp.json()["unmasked_text"]
    assert resp.json()["unmasked_text"] == "текст "

    tail = await fake_redis.get("pii:tail:sess7")
    assert tail == "PERS"


# ----------------------------------------------------------
# DELETE /session endpoint tests
# ----------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_session_removes_mapping_and_tail(client, fake_redis):
    """DELETE /session удаляет оба ключа: mapping и tail."""
    await fake_redis.setex("pii:mapping:sess8", 300, json.dumps({"PERSON_1": "X"}))
    await fake_redis.setex("pii:tail:sess8", 300, "PERSON")

    resp = await client.delete("/api/session/sess8")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    assert await fake_redis.get("pii:mapping:sess8") is None
    assert await fake_redis.get("pii:tail:sess8") is None


@pytest.mark.asyncio
async def test_delete_session_nonexistent_returns_deleted_false(client):
    """DELETE несуществующей сессии → deleted=False."""
    resp = await client.delete("/api/session/nonexistent_xyz")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is False
