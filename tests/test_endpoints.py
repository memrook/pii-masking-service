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


def mapping_obj(language: str, entries: list[dict]) -> str:
    return json.dumps({"language": language, "entries": entries}, ensure_ascii=False)


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
async def test_mask_endpoint_creates_mapping_when_no_pii(client, fake_redis):
    """/mask без PII → Redis ключ создаётся с языком и пустыми entries; hint=None."""
    resp = await client.post("/api/mask", json={
        "text": "привет как дела", "session_id": "sess1", "language": "ru",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "sess1"
    assert data["entities_found"] == []
    assert data["hint"] is None
    assert data["masked_spans"] == []

    stored = json.loads(await fake_redis.get("pii:mapping:sess1"))
    assert stored == {"language": "ru", "entries": []}


@pytest.mark.asyncio
async def test_mask_endpoint_marker_returns_hint(client, fake_redis, monkeypatch):
    """Метка в результате → возвращается непустой hint и masked_spans."""
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
    ]))
    resp = await client.post("/api/mask", json={
        "text": "Иванов пришёл", "session_id": "sessH", "language": "ru",
    })
    data = resp.json()
    assert data["masked_text"] == "[Имя 1] пришёл"
    assert data["hint"] is not None and "[Имя N]" in data["hint"]
    assert data["masked_spans"][0] == {"start": 0, "end": 7, "type": "PERSON"}


@pytest.mark.asyncio
async def test_mask_endpoint_prepend_hint_embeds_and_offsets_spans(client, fake_redis, monkeypatch):
    """prepend_hint=true вшивает подсказку в начало masked_text и сдвигает span'ы."""
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
    ]))
    resp = await client.post("/api/mask", json={
        "text": "Иванов пришёл", "session_id": "sessP", "language": "ru", "prepend_hint": True,
    })
    data = resp.json()
    # подсказка вшита в начало, метка следует за ней
    assert data["masked_text"].startswith(data["hint"])
    assert "[Имя 1] пришёл" in data["masked_text"]
    # span указывает на фактический "[Имя 1]" в итоговом тексте (со сдвигом)
    sp = data["masked_spans"][0]
    assert data["masked_text"][sp["start"]:sp["end"]] == "[Имя 1]"


@pytest.mark.asyncio
async def test_mask_endpoint_prepend_hint_no_markers_noop(client, fake_redis, monkeypatch):
    """prepend_hint=true без меток (только суррогаты) ничего не вшивает."""
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        __import__("presidio_analyzer").RecognizerResult(
            entity_type="INN_RU", start=4, end=16, score=0.9),
    ]))
    resp = await client.post("/api/mask", json={
        "text": "ИНН 772512345678", "session_id": "sessPN", "language": "ru", "prepend_hint": True,
    })
    data = resp.json()
    assert data["hint"] is None
    assert not data["masked_text"].startswith("Примечание")


@pytest.mark.asyncio
async def test_mask_endpoint_surrogate_no_hint(client, fake_redis, monkeypatch):
    """Только суррогаты (без меток) → hint=None."""
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        __import__("presidio_analyzer").RecognizerResult(
            entity_type="INN_RU", start=4, end=16, score=0.9),
    ]))
    resp = await client.post("/api/mask", json={
        "text": "ИНН 772512345678", "session_id": "sessS", "language": "ru",
    })
    data = resp.json()
    assert data["hint"] is None
    assert "772512345678" not in data["masked_text"]


@pytest.mark.asyncio
async def test_mask_endpoint_idempotent_marker_in_session(client, fake_redis, monkeypatch):
    """Одинаковый original в двух /mask → один и тот же out."""
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([FakeSpan(0, 6, "PER", "Иванов")]))
    await client.post("/api/mask", json={"text": "Иванов", "session_id": "sess3", "language": "ru"})

    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([FakeSpan(0, 6, "PER", "Иванов")]))
    resp2 = await client.post("/api/mask", json={"text": "Иванов снова", "session_id": "sess3", "language": "ru"})
    assert resp2.json()["masked_text"] == "[Имя 1] снова"

    stored = json.loads(await fake_redis.get("pii:mapping:sess3"))
    persons = [e for e in stored["entries"] if e["type"] == "PERSON"]
    assert len(persons) == 1


@pytest.mark.asyncio
async def test_mask_endpoint_language_mismatch_returns_409(client, fake_redis):
    """Второй /mask той же сессии с другим языком → 409."""
    await fake_redis.setex("pii:mapping:sessL", 300, mapping_obj("ru", []))
    resp = await client.post("/api/mask", json={
        "text": "hello", "session_id": "sessL", "language": "en",
    })
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_mask_endpoint_text_contains_prior_out_returns_409(client, fake_redis):
    """Новый текст содержит ранее выданный токен сессии → 409."""
    entries = [{"out": "+7 921 555-12-34", "original": "+79991234567",
                "type": "PHONE", "kind": "surrogate"}]
    await fake_redis.setex("pii:mapping:sessC", 300, mapping_obj("ru", entries))
    resp = await client.post("/api/mask", json={
        "text": "перезвоните на +7 921 555-12-34 пожалуйста",
        "session_id": "sessC", "language": "ru",
    })
    assert resp.status_code == 409


# ----------------------------------------------------------
# /unmask endpoint tests
# ----------------------------------------------------------

@pytest.mark.asyncio
async def test_unmask_endpoint_restores_values(client, fake_redis):
    entries = [
        {"out": "[Имя 1]", "original": "Иванов Иван", "type": "PERSON", "kind": "marker", "index": 1},
        {"out": "6591557797", "original": "772512345678", "type": "INN", "kind": "surrogate"},
    ]
    await fake_redis.setex("pii:mapping:sessU", 300, mapping_obj("ru", entries))
    resp = await client.post("/api/unmask", json={
        "text": "Договор с [Имя 1] (ИНН 6591557797)", "session_id": "sessU",
    })
    data = resp.json()
    assert data["unmasked_text"] == "Договор с Иванов Иван (ИНН 772512345678)"
    assert data["tokens_replaced"] == 2


@pytest.mark.asyncio
async def test_unmask_endpoint_no_mapping_passthrough(client):
    resp = await client.post("/api/unmask", json={"text": "любой текст", "session_id": "nope"})
    assert resp.status_code == 200
    assert resp.json()["tokens_replaced"] == 0


# ----------------------------------------------------------
# /unmask-chunk endpoint tests
# ----------------------------------------------------------

@pytest.mark.asyncio
async def test_unmask_chunk_missing_session_returns_404(client):
    resp = await client.post("/api/unmask-chunk", json={
        "text": "любой текст", "session_id": "nonexistent", "is_final": True,
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_unmask_chunk_split_marker_across_chunks(client, fake_redis):
    """Метка разрезана между чанками: '[Им' + 'я 1] готов' → 'Иванов готов'."""
    entries = [{"out": "[Имя 1]", "original": "Иванов", "type": "PERSON", "kind": "marker", "index": 1}]
    await fake_redis.setex("pii:mapping:sess4", 300, mapping_obj("ru", entries))

    resp1 = await client.post("/api/unmask-chunk", json={
        "text": "Договор с [Им", "session_id": "sess4", "is_final": False,
    })
    assert resp1.json()["unmasked_text"] == "Договор с "

    resp2 = await client.post("/api/unmask-chunk", json={
        "text": "я 1] готов", "session_id": "sess4", "is_final": True,
    })
    assert resp2.json()["unmasked_text"] == "Иванов готов"


@pytest.mark.asyncio
async def test_unmask_chunk_full_out_at_end_demasked_when_not_prefix(client, fake_redis):
    """Полный out в конце чанка, не являющийся префиксом другого, демаскируется сразу."""
    entries = [{"out": "[Имя 1]", "original": "Иванов", "type": "PERSON", "kind": "marker", "index": 1}]
    await fake_redis.setex("pii:mapping:sessF", 300, mapping_obj("ru", entries))
    resp = await client.post("/api/unmask-chunk", json={
        "text": "привет [Имя 1]", "session_id": "sessF", "is_final": False,
    })
    assert resp.json()["unmasked_text"] == "привет Иванов"


@pytest.mark.asyncio
async def test_unmask_chunk_empty_mapping_passthrough(client, fake_redis):
    await fake_redis.setex("pii:mapping:sess5", 300, mapping_obj("ru", []))
    resp = await client.post("/api/unmask-chunk", json={
        "text": "обычный текст без токенов", "session_id": "sess5", "is_final": True,
    })
    assert resp.status_code == 200
    assert resp.json()["unmasked_text"] == "обычный текст без токенов"


@pytest.mark.asyncio
async def test_unmask_chunk_is_final_removes_tail_key(client, fake_redis):
    entries = [{"out": "[Имя 1]", "original": "Иванов", "type": "PERSON", "kind": "marker", "index": 1}]
    await fake_redis.setex("pii:mapping:sess6", 300, mapping_obj("ru", entries))
    await fake_redis.setex("pii:tail:sess6", 300, "[Имя")

    await client.post("/api/unmask-chunk", json={
        "text": " 1]", "session_id": "sess6", "is_final": True,
    })
    assert await fake_redis.get("pii:tail:sess6") is None


@pytest.mark.asyncio
async def test_unmask_chunk_buffers_partial_token(client, fake_redis):
    entries = [{"out": "[Имя 1]", "original": "Иванов", "type": "PERSON", "kind": "marker", "index": 1}]
    await fake_redis.setex("pii:mapping:sess7", 300, mapping_obj("ru", entries))

    resp = await client.post("/api/unmask-chunk", json={
        "text": "текст [Им", "session_id": "sess7", "is_final": False,
    })
    assert resp.json()["unmasked_text"] == "текст "
    assert await fake_redis.get("pii:tail:sess7") == "[Им"


# ----------------------------------------------------------
# DELETE /session endpoint tests
# ----------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_session_removes_mapping_and_tail(client, fake_redis):
    await fake_redis.setex("pii:mapping:sess8", 300, mapping_obj("ru", []))
    await fake_redis.setex("pii:tail:sess8", 300, "[Имя")

    resp = await client.delete("/api/session/sess8")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert await fake_redis.get("pii:mapping:sess8") is None
    assert await fake_redis.get("pii:tail:sess8") is None


@pytest.mark.asyncio
async def test_delete_session_nonexistent_returns_deleted_false(client):
    resp = await client.delete("/api/session/nonexistent_xyz")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is False


# ----------------------------------------------------------
# /stats endpoint tests
# ----------------------------------------------------------

@pytest.mark.asyncio
async def test_stats_lists_entity_types(client):
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    types = set(resp.json()["supported_entity_types"])
    assert {"ACCOUNT", "BIK", "KPP", "PERSON", "PHONE"} <= types
