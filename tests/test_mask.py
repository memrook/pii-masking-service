# tests/test_mask.py
import pytest
from presidio_analyzer import RecognizerResult
from app import masker
from app.masker import mask_text, find_safe_boundary
from tests.helpers import FakeNERTagger, FakeSpan, FakePresidioAnalyzer


# ── helpers ──────────────────────────────────────────────────────────────────

def presidio_result(entity_type: str, start: int, end: int) -> RecognizerResult:
    return RecognizerResult(entity_type=entity_type, start=start, end=end, score=0.9)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_mask_raises_if_models_not_loaded(monkeypatch):
    """mask_text must raise RuntimeError when called before load_models()."""
    monkeypatch.setattr(masker, "_models_loaded", False)
    with pytest.raises(RuntimeError, match="Модели не загружены"):
        mask_text("текст", "sess")


def test_mask_person_from_natasha(setup_stubs, monkeypatch):
    """Natasha PER span → token PERSON_1 inserted, mapping correct."""
    text = "Иванов Иван пришёл"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 11, "PER", "Иванов Иван"),
    ]))

    masked, mapping, types = mask_text(text, "sess")

    assert "PERSON_1" in masked
    assert mapping["PERSON_1"] == "Иванов Иван"
    assert "PER" in types
    assert "Иванов Иван" not in masked


def test_mask_org_from_natasha(setup_stubs, monkeypatch):
    """Natasha ORG span → token ORG_1."""
    text = "Сотрудник ООО Ромашка уволен"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(10, 21, "ORG", "ООО Ромашка"),
    ]))

    masked, mapping, types = mask_text(text, "sess")

    assert mapping["ORG_1"] == "ООО Ромашка"
    assert "ООО Ромашка" not in masked
    assert "ORG" in types


def test_mask_phone_from_presidio(setup_stubs, monkeypatch):
    """Presidio PHONE_RU result → token PHONE_1."""
    text = "Звоните на +79991234567"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PHONE_RU", 11, 23),
    ]))

    masked, mapping, types = mask_text(text, "sess")

    assert mapping["PHONE_1"] == "+79991234567"
    assert "+79991234567" not in masked
    assert "PHONE_RU" in types


def test_mask_inn_from_presidio(setup_stubs, monkeypatch):
    """Presidio INN_RU result → token INN_1."""
    text = "ИНН 772512345678"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("INN_RU", 4, 16),
    ]))

    masked, mapping, types = mask_text(text, "sess")

    assert mapping["INN_1"] == "772512345678"
    assert "772512345678" not in masked


def test_mask_passport_with_nom_sign(setup_stubs, monkeypatch):
    """Паспорт в формате '4511 № 654321' должен маскироваться."""
    text = "паспорт серии 4511 № 654321"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PASSPORT_RU", 14, 27),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert mapping["PASSPORT_1"] == "4511 № 654321"
    assert "4511" not in masked


def test_mask_passport_with_labels(setup_stubs, monkeypatch):
    """Паспорт в формате 'серия 4511 номер 654321' должен маскироваться."""
    text = "паспорт серия 4511 номер 654321"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PASSPORT_RU", 8, 31),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert mapping["PASSPORT_1"] == "серия 4511 номер 654321"
    assert "4511" not in masked


def test_mask_presidio_receives_correct_entities(setup_stubs, monkeypatch):
    """mask_text must request the right entity types from Presidio."""
    analyzer = FakePresidioAnalyzer()
    monkeypatch.setattr(masker, "_presidio_analyzer", analyzer)

    mask_text("текст", "sess")

    expected = {"INN_RU", "OGRN_RU", "PHONE_NUMBER", "PHONE_RU",
                "EMAIL_ADDRESS", "PASSPORT_RU", "SNILS_RU", "ADDRESS_RU",
                "PERSON", "CARD", "CREDIT_CARD", "VIN"}
    assert set(analyzer.last_entities_requested) == expected


def test_mask_presidio_overlap_with_natasha_is_skipped(setup_stubs, monkeypatch):
    """Presidio result covering same span as natasha result must be skipped."""
    text = "ООО Ромашка"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 11, "ORG", "ООО Ромашка"),
    ]))
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("ORG", 0, 11),   # same span as natasha
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    # Only one token created, not two
    assert list(mapping.keys()) == ["ORG_1"]
    assert masked == "ORG_1"


def test_mask_right_to_left_replacement(setup_stubs, monkeypatch):
    """Two entities: replaced right-to-left so character indices stay valid."""
    text = "Петров звонил на +79991234567"
    #       0..6               17..29
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Петров"),
    ]))
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PHONE_RU", 17, 29),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert mapping["PERSON_1"] == "Петров"
    assert mapping["PHONE_1"] == "+79991234567"
    assert masked == "PERSON_1 звонил на PHONE_1"


def test_mask_counter_increments_for_same_type(setup_stubs, monkeypatch):
    """Two persons → PERSON_1 and PERSON_2."""
    text = "Иванов и Петров"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
        FakeSpan(9, 15, "PER", "Петров"),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert "PERSON_1" in mapping
    assert "PERSON_2" in mapping
    assert mapping["PERSON_1"] == "Иванов"
    assert mapping["PERSON_2"] == "Петров"
    assert masked == "PERSON_1 и PERSON_2"


def test_mask_english_skips_natasha(setup_stubs, monkeypatch):
    """language='en' must not call natasha Doc, but must still call Presidio."""
    call_log = []

    class TrackingFakeDoc:
        def __init__(self, text):
            call_log.append("Doc created")

        def segment(self, s): pass
        def tag_ner(self, t): pass

    analyzer = FakePresidioAnalyzer()
    monkeypatch.setattr(masker, "Doc", TrackingFakeDoc)
    monkeypatch.setattr(masker, "_presidio_analyzer", analyzer)

    mask_text("Hello world", "sess", language="en")

    assert call_log == [], "Doc should not be instantiated for language='en'"
    assert analyzer.last_entities_requested != [], "Presidio must still be called for language='en'"


def test_mask_empty_text(setup_stubs):
    masked, mapping, types = mask_text("", "sess")
    assert masked == ""
    assert mapping == {}
    assert types == []


def test_mask_text_with_no_pii(setup_stubs):
    """Clean text with no entities returns text unchanged."""
    text = "Сегодня хорошая погода"
    masked, mapping, types = mask_text(text, "sess")
    assert masked == text
    assert mapping == {}
    assert types == []


def test_mask_address_from_presidio(setup_stubs, monkeypatch):
    """Presidio ADDRESS_RU result → token ADDRESS_1."""
    address = "г. Москва, ул. Мира, д. 5, кв. 25"
    text = "адрес: " + address
    start = text.index(address)
    end = start + len(address)
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("ADDRESS_RU", start, end),
    ]))

    masked, mapping, types = mask_text(text, "sess")

    assert mapping["ADDRESS_1"] == address
    assert "ул. Мира" not in masked
    assert "ADDRESS_RU" in types


# ── stateful allocator tests ─────────────────────────────────────────────────

def test_mask_continues_counters_from_existing_mapping(setup_stubs, monkeypatch):
    """mask_text с existing_mapping продолжает счётчики, не сбрасывает их."""
    text = "Сидоров пришёл"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 7, "PER", "Сидоров"),
    ]))
    existing = {"PERSON_5": "Петров"}

    masked, mapping, _ = mask_text(text, "sess", existing_mapping=existing)

    assert "PERSON_6" in mapping
    assert mapping["PERSON_6"] == "Сидоров"
    assert mapping["PERSON_5"] == "Петров"
    assert masked == "PERSON_6 пришёл"


def test_mask_reuses_token_for_same_original(setup_stubs, monkeypatch):
    """Тот же original в existing_mapping → возвращается тот же токен, новый не создаётся."""
    text = "Иванов здесь"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
    ]))
    existing = {"PERSON_1": "Иванов"}

    masked, mapping, _ = mask_text(text, "sess", existing_mapping=existing)

    assert masked == "PERSON_1 здесь"
    assert list(mapping.keys()) == ["PERSON_1"]  # не добавлено PERSON_2
    assert mapping["PERSON_1"] == "Иванов"


def test_mask_empty_text_with_existing_mapping(setup_stubs):
    """Пустой текст с existing_mapping возвращает existing_mapping без изменений."""
    existing = {"PERSON_1": "Иванов"}
    masked, mapping, types = mask_text("", "sess", existing_mapping=existing)
    assert masked == ""
    assert mapping == {"PERSON_1": "Иванов"}
    assert types == []


# ── find_safe_boundary tests ─────────────────────────────────────────────────

PREFIXES = ["PERSON", "PHONE", "INN"]


def test_safe_boundary_full_token_in_middle():
    """Полный токен в середине строки — граница конец строки (всё безопасно)."""
    text = "Привет PERSON_1 как дела"
    assert find_safe_boundary(text, PREFIXES) == len(text)


def test_safe_boundary_full_token_at_end():
    """Полный токен в конце — держим его в буфере (может продолжиться: PERSON_15)."""
    text = "Привет PERSON_1"
    boundary = find_safe_boundary(text, PREFIXES)
    assert boundary == text.index("PERSON_1")


def test_safe_boundary_partial_prefix_at_end():
    """Частичный префикс в конце строки — держим в буфере."""
    text = "Привет PERS"
    boundary = find_safe_boundary(text, PREFIXES)
    assert boundary == text.index("PERS")


def test_safe_boundary_full_prefix_no_underscore():
    """Полный префикс без _ в конце — держим в буфере (split: 'PERSON' + '_1')."""
    text = "Привет PERSON"
    boundary = find_safe_boundary(text, PREFIXES)
    assert boundary == text.index("PERSON")


def test_safe_boundary_full_prefix_with_underscore():
    """Префикс + _ в конце — держим в буфере."""
    text = "Привет PERSON_"
    boundary = find_safe_boundary(text, PREFIXES)
    assert boundary == text.index("PERSON_")


def test_safe_boundary_clean_text():
    """Чистый текст без токенов — граница конец строки."""
    text = "Привет как дела"
    assert find_safe_boundary(text, PREFIXES) == len(text)


def test_safe_boundary_empty_text():
    """Пустой текст → 0."""
    assert find_safe_boundary("", PREFIXES) == 0


def test_address_recognizer_patterns():
    """Regex patterns inside the recognizer match all target address formats."""
    import re
    from app.masker import _make_address_recognizer
    recognizer = _make_address_recognizer()
    patterns = {p.name: re.compile(p.regex, re.IGNORECASE) for p in recognizer.patterns}

    # ADDRESS_FULL: city prefix required
    assert patterns["ADDRESS_FULL"].search("г. Москва, ул. Мира, д. 5, кв. 25")
    assert patterns["ADDRESS_FULL"].search("Московская обл., г. Химки, ш. Ленинградское, д. 1")

    # ADDRESS_STREET_ONLY: street abbreviation + house, no city
    assert patterns["ADDRESS_STREET_ONLY"].search("ул. Пушкина, д. 10")

    # ADDRESS_NO_ABBR: full word street type, bare number
    assert patterns["ADDRESS_NO_ABBR"].search("проспект Ленина, 12")
