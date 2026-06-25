# tests/test_mask.py
import re

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


def out_for(entries, original):
    """out, выданный для конкретного оригинала."""
    return next(e["out"] for e in entries if e["original"] == original)


def entry_for(entries, original):
    return next(e for e in entries if e["original"] == original)


# ── базовое маскирование: метки ───────────────────────────────────────────────

def test_mask_raises_if_models_not_loaded(monkeypatch):
    monkeypatch.setattr(masker, "_models_loaded", False)
    with pytest.raises(RuntimeError, match="Модели не загружены"):
        mask_text("текст", "sess")


def test_mask_person_becomes_marker(setup_stubs, monkeypatch):
    """Natasha PER → метка [Имя 1]."""
    text = "Иванов Иван пришёл"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 11, "PER", "Иванов Иван"),
    ]))
    masked, entries, types, spans = mask_text(text, "sess")

    assert masked == "[Имя 1] пришёл"
    e = entry_for(entries, "Иванов Иван")
    assert e["out"] == "[Имя 1]" and e["type"] == "PERSON" and e["kind"] == "marker" and e["index"] == 1
    assert "PERSON" in types
    assert "Иванов Иван" not in masked


def test_mask_org_becomes_marker(setup_stubs, monkeypatch):
    text = "Сотрудник ООО Ромашка уволен"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(10, 21, "ORG", "ООО Ромашка"),
    ]))
    masked, entries, types, spans = mask_text(text, "sess")

    assert out_for(entries, "ООО Ромашка") == "[Организация 1]"
    assert "ООО Ромашка" not in masked
    assert "ORG" in types


def test_mask_address_becomes_marker(setup_stubs, monkeypatch):
    address = "г. Москва, ул. Мира, д. 5, кв. 25"
    text = "адрес: " + address
    start = text.index(address)
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("ADDRESS_RU", start, start + len(address)),
    ]))
    masked, entries, types, spans = mask_text(text, "sess")

    assert out_for(entries, address) == "[Адрес 1]"
    assert "ул. Мира" not in masked
    assert "ADDRESS" in types


def test_mask_counter_increments_for_same_type(setup_stubs, monkeypatch):
    text = "Иванов и Петров"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
        FakeSpan(9, 15, "PER", "Петров"),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")

    assert masked == "[Имя 1] и [Имя 2]"
    assert out_for(entries, "Иванов") == "[Имя 1]"
    assert out_for(entries, "Петров") == "[Имя 2]"


def test_natasha_span_clipped_at_newline(setup_stubs, monkeypatch):
    """Natasha жадно растянула ORG через \\n на слово со след. строки —
    спан клиппится по переносу, перенос строки сохраняется в выводе."""
    text = "ООО Ромашка \n БИК: 044525000"
    #       0123456789..  (over-extended span до "БИК")
    over = "ООО Ромашка \n БИК"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, len(over), "ORG", over),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")

    assert "\n" in masked, "перенос строки не должен поглощаться меткой"
    assert out_for(entries, "ООО Ромашка") == "[Организация 1]"
    # метка не захватила вторую строку
    assert "БИК: 044525000" in masked
    assert masked == "[Организация 1] \n БИК: 044525000"


def test_marker_collision_with_source_text_increments_index(setup_stubs, monkeypatch):
    """Если в тексте уже есть '[Имя 1]', новая метка получает следующий индекс."""
    text = "[Имя 1] и Петров"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(10, 16, "PER", "Петров"),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")
    # "[Имя 1]" в тексте занят → Петров стал "[Имя 2]"
    assert out_for(entries, "Петров") == "[Имя 2]"
    assert masked == "[Имя 1] и [Имя 2]"


# ── базовое маскирование: суррогаты ───────────────────────────────────────────

def test_mask_phone_becomes_surrogate(setup_stubs, monkeypatch):
    text = "Звоните на +79991234567"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PHONE_RU", 11, 23),
    ]))
    masked, entries, types, _ = mask_text(text, "sess")

    e = entry_for(entries, "+79991234567")
    assert e["type"] == "PHONE" and e["kind"] == "surrogate"
    assert re.fullmatch(r"\+7 9\d{2} 555-\d{2}-\d{2}", e["out"])
    assert "+79991234567" not in masked
    assert "PHONE" in types


def test_mask_inn_becomes_surrogate_same_length(setup_stubs, monkeypatch):
    text = "ИНН 772512345678"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("INN_RU", 4, 16),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")

    out = out_for(entries, "772512345678")
    assert len(out) == 12 and out.isdigit()
    assert "772512345678" not in masked


def test_mask_passport_surrogate_format(setup_stubs, monkeypatch):
    text = "паспорт серии 4511 № 654321"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PASSPORT_RU", 14, 27),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")
    out = out_for(entries, "4511 № 654321")
    assert re.fullmatch(r"\d{4} \d{6}", out)
    assert "4511 № 654321" not in masked


def test_canonicalization_credit_card_to_card(setup_stubs, monkeypatch):
    """CREDIT_CARD (en built-in) → канонический тип CARD."""
    text = "card 4111111111111111"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("CREDIT_CARD", 5, 21),
    ]))
    _, entries, types, _ = mask_text(text, "sess", language="en")
    assert entry_for(entries, "4111111111111111")["type"] == "CARD"
    assert "CARD" in types


def test_canonicalization_phone_number_to_phone(setup_stubs, monkeypatch):
    text = "call +12025550143"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PHONE_NUMBER", 5, 17),
    ]))
    _, entries, types, _ = mask_text(text, "sess", language="en")
    assert entry_for(entries, "+12025550143")["type"] == "PHONE"
    assert "PHONE" in types


# ── идемпотентность / детерминизм суррогатов ──────────────────────────────────

def test_surrogate_same_original_same_out_in_one_call(setup_stubs, monkeypatch):
    text = "счёт 40702810000000000111 и снова 40702810000000000111"
    s = text.index("40702810000000000111")
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("ACCOUNT_RU", s, s + 20),
        presidio_result("ACCOUNT_RU", text.rindex("40702810000000000111"),
                        text.rindex("40702810000000000111") + 20),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")
    # одна запись на оригинал, обе позиции заменены одним и тем же out
    accounts = [e for e in entries if e["type"] == "ACCOUNT"]
    assert len(accounts) == 1
    assert masked.count(accounts[0]["out"]) == 2


# ── canonical merge PER/PERSON через existing ─────────────────────────────────

def test_per_and_person_share_canonical_counter(setup_stubs, monkeypatch):
    """existing запись PERSON (index 5) → новая PER-сущность получает index 6."""
    text = "Сидоров пришёл"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 7, "PER", "Сидоров"),
    ]))
    existing = [{"out": "[Имя 5]", "original": "Петров", "type": "PERSON",
                 "kind": "marker", "index": 5}]
    masked, entries, _, _ = mask_text(text, "sess", existing_entries=existing)
    assert out_for(entries, "Сидоров") == "[Имя 6]"
    assert masked == "[Имя 6] пришёл"


def test_mask_reuses_out_for_same_original(setup_stubs, monkeypatch):
    text = "Иванов здесь"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Иванов"),
    ]))
    existing = [{"out": "[Имя 1]", "original": "Иванов", "type": "PERSON",
                 "kind": "marker", "index": 1}]
    masked, entries, _, _ = mask_text(text, "sess", existing_entries=existing)
    assert masked == "[Имя 1] здесь"
    assert len([e for e in entries if e["type"] == "PERSON"]) == 1


def test_surrogate_deterministic_across_calls_same_session(setup_stubs, monkeypatch):
    text = "ИНН 772512345678"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("INN_RU", 4, 16),
    ]))
    _, e1, _, _ = mask_text(text, "sess")
    out1 = out_for(e1, "772512345678")
    # повторный вызов с уже накопленными entries возвращает тот же out
    _, e2, _, _ = mask_text(text, "sess", existing_entries=e1)
    assert out_for(e2, "772512345678") == out1
    assert len([e for e in e2 if e["type"] == "INN"]) == 1


# ── masked_spans ──────────────────────────────────────────────────────────────

def test_masked_spans_point_to_correct_fragments(setup_stubs, monkeypatch):
    """Координaты span указывают на фактические out в masked_text (замены разной длины)."""
    text = "Петров звонил на +79991234567"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 6, "PER", "Петров"),
    ]))
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PHONE_RU", 17, 29),
    ]))
    masked, entries, _, spans = mask_text(text, "sess")

    # каждый span вырезает из masked_text именно тот out, что в записи
    out_by_type = {e["type"]: e["out"] for e in entries}
    for sp in spans:
        assert masked[sp["start"]:sp["end"]] == out_by_type[sp["type"]]
    # метка имени в начале
    assert masked.startswith("[Имя 1] звонил на ")


def test_mask_empty_text(setup_stubs):
    masked, entries, types, spans = mask_text("", "sess")
    assert masked == "" and entries == [] and types == [] and spans == []


def test_mask_text_with_no_pii(setup_stubs):
    text = "Сегодня хорошая погода"
    masked, entries, types, spans = mask_text(text, "sess")
    assert masked == text and entries == [] and types == [] and spans == []


def test_mask_empty_text_with_existing_entries(setup_stubs):
    existing = [{"out": "[Имя 1]", "original": "Иванов", "type": "PERSON",
                 "kind": "marker", "index": 1}]
    masked, entries, types, _ = mask_text("", "sess", existing_entries=existing)
    assert masked == "" and entries == existing and types == []


def test_mask_english_skips_natasha(setup_stubs, monkeypatch):
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
    assert call_log == []
    assert analyzer.last_entities_requested != []


# ── overlap с natasha ─────────────────────────────────────────────────────────

def test_mask_presidio_overlap_with_natasha_is_skipped(setup_stubs, monkeypatch):
    text = "ООО Ромашка"
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger([
        FakeSpan(0, 11, "ORG", "ООО Ромашка"),
    ]))
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("ORG", 0, 11),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")
    assert len(entries) == 1
    assert masked == "[Организация 1]"


# ── _presidio_entities (язык) ─────────────────────────────────────────────────

def test_mask_presidio_receives_correct_entities_for_russian(setup_stubs, monkeypatch):
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
    analyzer = FakePresidioAnalyzer()
    monkeypatch.setattr(masker, "_presidio_analyzer", analyzer)
    mask_text("some text", "sess", language="en")
    entities = set(analyzer.last_entities_requested)
    assert "PHONE_NUMBER" in entities
    assert "PERSON" in entities
    assert "PHONE_RU" not in entities


def test_account_bik_kpp_in_presidio_entity_request(setup_stubs, monkeypatch):
    analyzer = FakePresidioAnalyzer()
    monkeypatch.setattr(masker, "_presidio_analyzer", analyzer)
    mask_text("текст", "sess", language="ru")
    entities = set(analyzer.last_entities_requested)
    assert {"ACCOUNT_RU", "BIK_RU", "KPP_RU"} <= entities


# ── английские метки ──────────────────────────────────────────────────────────

def test_english_person_marker_label(setup_stubs, monkeypatch):
    text = "John Smith called"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result("PERSON", 0, 10),
    ]))
    masked, entries, _, _ = mask_text(text, "sess", language="en")
    assert out_for(entries, "John Smith") == "[Name 1]"
    assert masked == "[Name 1] called"


# ── find_safe_boundary (по фактическим out) ──────────────────────────────────

OUTS = ["[Имя 1]", "+7 921 555-12-34"]


def test_safe_boundary_full_marker_in_middle():
    text = "Привет [Имя 1] как дела"
    assert find_safe_boundary(text, OUTS) == len(text)


def test_safe_boundary_full_marker_at_end_not_prefix_demasked():
    """Полный out в конце, не являющийся префиксом другого, демаскируется сразу."""
    text = "Привет [Имя 1]"
    assert find_safe_boundary(text, OUTS) == len(text)


def test_safe_boundary_partial_marker_at_end_held():
    text = "Привет [Им"
    assert find_safe_boundary(text, OUTS) == text.index("[Им")


def test_safe_boundary_prefix_ambiguity_holds_shorter_out():
    """out1 ⊂ out2: полный короткий out в конце удерживается, пока возможно продолжение."""
    outs = ["12345", "123456"]
    text = "abc12345"
    assert find_safe_boundary(text, outs) == text.index("12345")


def test_safe_boundary_clean_text():
    assert find_safe_boundary("Привет как дела", OUTS) == len("Привет как дела")


def test_safe_boundary_empty_text():
    assert find_safe_boundary("", OUTS) == 0


def test_safe_boundary_no_outs():
    assert find_safe_boundary("любой текст", []) == len("любой текст")


# ══════════════════════════════════════════════════════════════════════════════
# Recognizer-уровневые тесты (реальные recognizer'ы, формат вывода не зависит
# от схемы маскирования) — без изменений по сути.
# ══════════════════════════════════════════════════════════════════════════════

def test_phone_ru_does_not_match_inside_long_digit_sequence():
    from app.masker import _make_ru_phone_recognizer
    rec = _make_ru_phone_recognizer()
    p = next(p for p in rec.patterns if p.name == "PHONE_7PLUS")
    assert not re.search(p.regex, "40702810000000000111")
    assert re.search(p.regex, "+7 (495) 123-45-67")
    assert re.search(p.regex, "89001234567")


def test_account_recognizer_matches_20_digit():
    from app.masker import _make_account_recognizer
    rec = _make_account_recognizer()
    res = rec.analyze("счёт 40702810000000000111", entities=["ACCOUNT_RU"])
    assert len(res) == 1
    assert (res[0].end - res[0].start) == 20


def test_bik_recognizer_requires_context_to_the_left():
    from app.masker import _make_bik_recognizer
    rec = _make_bik_recognizer()
    assert rec.analyze("номер 123456789", entities=["BIK_RU"]) == []
    res = rec.analyze("БИК 044525225", entities=["BIK_RU"])
    assert len(res) == 1
    assert res[0].start == 4 and res[0].end == 13


def test_kpp_recognizer_requires_context_to_the_left():
    from app.masker import _make_kpp_recognizer
    rec = _make_kpp_recognizer()
    assert rec.analyze("идентификатор 230801001", entities=["KPP_RU"]) == []
    res = rec.analyze("КПП: 230801001", entities=["KPP_RU"])
    assert len(res) == 1


def test_bik_and_kpp_adjacent_do_not_cross_fire():
    from app.masker import _make_bik_recognizer, _make_kpp_recognizer
    text = "КПП 230801001, БИК 044525225"
    bik_res = _make_bik_recognizer().analyze(text, entities=["BIK_RU"])
    kpp_res = _make_kpp_recognizer().analyze(text, entities=["KPP_RU"])
    assert len(bik_res) == 1
    assert text[bik_res[0].start:bik_res[0].end] == "044525225"
    assert len(kpp_res) == 1
    assert text[kpp_res[0].start:kpp_res[0].end] == "230801001"


def test_resolve_overlap_higher_score_wins_for_bank_codes():
    from app.masker import _resolve_presidio_overlaps
    high = presidio_result_scored("BIK_RU", 0, 9, 0.95)
    low = presidio_result_scored("KPP_RU", 0, 9, 0.85)
    kept = _resolve_presidio_overlaps([low, high])
    assert len(kept) == 1
    assert kept[0].entity_type == "BIK_RU"


def test_presidio_custom_entity_wins_over_builtin_on_overlap(setup_stubs, monkeypatch):
    text = "ИНН 7707083893"
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer([
        presidio_result_scored("INN_RU", 4, 14, 0.85),
        presidio_result_scored("PHONE_NUMBER", 4, 14, 0.85),
    ]))
    masked, entries, _, _ = mask_text(text, "sess")
    # один токен, тип INN
    assert len(entries) == 1
    assert entries[0]["type"] == "INN"
    assert "7707083893" not in masked


def test_address_recognizer_patterns():
    from app.masker import _make_address_recognizer
    recognizer = _make_address_recognizer()
    patterns = {p.name: re.compile(p.regex, re.IGNORECASE) for p in recognizer.patterns}
    assert patterns["ADDRESS_FULL"].search("г. Москва, ул. Мира, д. 5, кв. 25")
    assert patterns["ADDRESS_FULL"].search("Московская обл., г. Химки, ш. Ленинградское, д. 1")
    assert patterns["ADDRESS_STREET_ONLY"].search("ул. Пушкина, д. 10")
    assert patterns["ADDRESS_NO_ABBR"].search("проспект Ленина, 12")
