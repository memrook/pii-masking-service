# tests/test_mask.py
import pytest
from presidio_analyzer import RecognizerResult
from app import masker
from app.masker import mask_text, find_safe_boundary
from tests.helpers import FakeNERTagger, FakeSpan, FakePresidioAnalyzer


# ── helpers ──────────────────────────────────────────────────────────────────

def presidio_result(entity_type: str, start: int, end: int) -> RecognizerResult:
    return RecognizerResult(entity_type=entity_type, start=start, end=end, score=0.9)


def presidio_result_scored(entity_type: str, start: int, end: int, score: float) -> RecognizerResult:
    return RecognizerResult(entity_type=entity_type, start=start, end=end, score=score)


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


def test_mask_presidio_receives_correct_entities_for_russian(setup_stubs, monkeypatch):
    """For language='ru': PHONE_RU used (not PHONE_NUMBER), PERSON excluded (natasha handles it)."""
    analyzer = FakePresidioAnalyzer()
    monkeypatch.setattr(masker, "_presidio_analyzer", analyzer)

    mask_text("текст", "sess", language="ru")

    entities = set(analyzer.last_entities_requested)
    assert "PHONE_RU" in entities
    assert "PHONE_NUMBER" not in entities
    assert "PERSON" not in entities
    assert entities >= {"INN_RU", "OGRN_RU", "EMAIL_ADDRESS", "PASSPORT_RU",
                        "SNILS_RU", "ADDRESS_RU", "CARD", "CREDIT_CARD", "VIN"}


def test_mask_presidio_receives_correct_entities_for_english(setup_stubs, monkeypatch):
    """For language='en': PHONE_NUMBER and PERSON included (natasha not used)."""
    analyzer = FakePresidioAnalyzer()
    monkeypatch.setattr(masker, "_presidio_analyzer", analyzer)

    mask_text("some text", "sess", language="en")

    entities = set(analyzer.last_entities_requested)
    assert "PHONE_NUMBER" in entities
    assert "PERSON" in entities
    assert "PHONE_RU" not in entities


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


# ── overlap resolution tests ─────────────────────────────────────────────────

def test_presidio_custom_entity_wins_over_builtin_on_overlap(setup_stubs, monkeypatch):
    """INN_RU (custom) beats PHONE_NUMBER (built-in) when both cover same span."""
    text = "ИНН 7707083893"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result_scored("INN_RU",      4, 14, 0.85),
        presidio_result_scored("PHONE_NUMBER", 4, 14, 0.85),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert masked == "ИНН INN_1"
    assert mapping == {"INN_1": "7707083893"}


def test_presidio_overlapping_spans_no_corrupted_output(setup_stubs, monkeypatch):
    """Overlapping Presidio spans must not produce broken interleaved tokens."""
    text = "ИНН 7707083893 и паспорт 4511 654321"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result_scored("INN_RU",      4, 14, 0.85),
        presidio_result_scored("PHONE_NUMBER", 4, 14, 0.80),   # overlaps INN_RU
        presidio_result_scored("PASSPORT_RU", 24, 37, 0.90),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert "7707083893" not in masked
    assert "4511 654321" not in masked
    assert "INN_1" in masked
    assert "PASSPORT_1" in masked
    # No interleaved fragments like "PHONE_1NN_1" or "INN_1ONE_1"
    assert "PHONE" not in masked


def test_higher_score_presidio_wins_over_lower_on_overlap(setup_stubs, monkeypatch):
    """When two Presidio results overlap, the one with higher score wins."""
    text = "номер 123456789"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result_scored("SNILS_RU",    6, 15, 0.50),
        presidio_result_scored("PHONE_NUMBER", 6, 15, 0.90),  # higher score
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    # Exactly one substitution, no corruption
    token_keys = list(mapping.keys())
    assert len(token_keys) == 1
    assert "123456789" not in masked


def test_natasha_allcaps_single_word_not_masked(setup_stubs, monkeypatch):
    """Single all-caps NER span (document keyword) is ignored as false positive."""
    text = "ДОГОВОР между сторонами"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 7, "ORG", "ДОГОВОР"),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert "ДОГОВОР" in masked
    assert mapping == {}


def test_natasha_multiword_allcaps_org_is_masked(setup_stubs, monkeypatch):
    """Multi-word all-caps span is a real org name, must be masked."""
    text = "ООО РОМАШКА"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 11, "ORG", "ООО РОМАШКА"),
    ]))

    masked, mapping, _ = mask_text(text, "sess")

    assert "ООО РОМАШКА" not in masked
    assert mapping.get("ORG_1") == "ООО РОМАШКА"


# ── PHONE_RU не должен ловить подстроку внутри длинных цифр ─────────────────

def test_phone_ru_does_not_match_inside_long_digit_sequence():
    """PHONE_7PLUS должен иметь (?<!\\d)/(?!\\d) — иначе ловит 8XXXXXXXXXX внутри 20-значного счёта."""
    import re
    from app.masker import _make_ru_phone_recognizer
    rec = _make_ru_phone_recognizer()
    p = next(p for p in rec.patterns if p.name == "PHONE_7PLUS")
    assert not re.search(p.regex, "40702810000000000111")
    # Реальные телефоны по-прежнему ловятся
    assert re.search(p.regex, "+7 (495) 123-45-67")
    assert re.search(p.regex, "89001234567")


# ── ACCOUNT/BIK/KPP recognizer-уровень (реальные recognizer'ы, не Fake) ───────

def test_account_recognizer_matches_20_digit():
    from app.masker import _make_account_recognizer
    rec = _make_account_recognizer()
    res = rec.analyze("счёт 40702810000000000111", entities=["ACCOUNT_RU"])
    assert len(res) == 1
    assert (res[0].end - res[0].start) == 20


def test_bik_recognizer_requires_context_to_the_left():
    """BIK_RU без слова 'БИК' слева → пусто."""
    from app.masker import _make_bik_recognizer
    rec = _make_bik_recognizer()
    assert rec.analyze("номер 123456789", entities=["BIK_RU"]) == []
    res = rec.analyze("БИК 044525225", entities=["BIK_RU"])
    assert len(res) == 1
    assert res[0].start == 4 and res[0].end == 13


def test_kpp_recognizer_requires_context_to_the_left():
    """KPP_RU без слова 'КПП' слева → пусто."""
    from app.masker import _make_kpp_recognizer
    rec = _make_kpp_recognizer()
    assert rec.analyze("идентификатор 230801001", entities=["KPP_RU"]) == []
    res = rec.analyze("КПП: 230801001", entities=["KPP_RU"])
    assert len(res) == 1


def test_bik_and_kpp_adjacent_do_not_cross_fire():
    """`КПП 230801001, БИК 044525225` — каждое число помечено только своим типом.

    Цифры между context-словом и матчем разделяют их: KPP_RU не считает
    БИК-число своим, BIK_RU не считает КПП-число своим.
    """
    from app.masker import _make_bik_recognizer, _make_kpp_recognizer
    text = "КПП 230801001, БИК 044525225"

    bik_res = _make_bik_recognizer().analyze(text, entities=["BIK_RU"])
    kpp_res = _make_kpp_recognizer().analyze(text, entities=["KPP_RU"])

    assert len(bik_res) == 1
    assert text[bik_res[0].start:bik_res[0].end] == "044525225"

    assert len(kpp_res) == 1
    assert text[kpp_res[0].start:kpp_res[0].end] == "230801001"


# ── overlap-резолвер для bank-кодов ──────────────────────────────────────────

def test_resolve_overlap_higher_score_wins_for_bank_codes():
    """Когда BIK_RU и KPP_RU перекрываются на одном span — побеждает большая score."""
    from app.masker import _resolve_presidio_overlaps
    high = presidio_result_scored("BIK_RU", 0, 9, 0.95)
    low  = presidio_result_scored("KPP_RU", 0, 9, 0.85)
    kept = _resolve_presidio_overlaps([low, high])
    assert len(kept) == 1
    assert kept[0].entity_type == "BIK_RU"


# ── Integration через mask_text + FakeAnalyzer ────────────────────────────────

def test_account_masked_via_mask_text(setup_stubs, monkeypatch):
    """ACCOUNT_RU span → ACCOUNT_1."""
    text = "расчётный счёт 40702810000000000111"
    start = text.index("40702810000000000111")
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("ACCOUNT_RU", start, start + 20),
    ]))
    masked, mapping, _ = mask_text(text, "sess")
    assert masked == "расчётный счёт ACCOUNT_1"
    assert mapping == {"ACCOUNT_1": "40702810000000000111"}


def test_bik_masked_via_mask_text(setup_stubs, monkeypatch):
    """BIK_RU span → BIK_1."""
    text = "БИК 044525225"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("BIK_RU", 4, 13),
    ]))
    masked, mapping, _ = mask_text(text, "sess")
    assert masked == "БИК BIK_1"
    assert mapping == {"BIK_1": "044525225"}


def test_kpp_masked_via_mask_text(setup_stubs, monkeypatch):
    """KPP_RU span → KPP_1."""
    text = "КПП: 230801001"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("KPP_RU", 5, 14),
    ]))
    masked, mapping, _ = mask_text(text, "sess")
    assert masked == "КПП: KPP_1"
    assert mapping == {"KPP_1": "230801001"}


def test_account_bik_kpp_in_presidio_entity_request(setup_stubs, monkeypatch):
    """mask_text запрашивает у Presidio типы ACCOUNT_RU, BIK_RU, KPP_RU для ru."""
    analyzer = FakePresidioAnalyzer()
    monkeypatch.setattr(masker, "_presidio_analyzer", analyzer)
    mask_text("текст", "sess", language="ru")
    entities = set(analyzer.last_entities_requested)
    assert {"ACCOUNT_RU", "BIK_RU", "KPP_RU"} <= entities


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
