# tests/test_unmask.py
from app.masker import unmask_text


def test_unmask_single_token():
    text = "Позвони PERSON_1 завтра"
    mapping = {"PERSON_1": "Иванову"}
    result, count = unmask_text(text, mapping)
    assert result == "Позвони Иванову завтра"
    assert count == 1


def test_unmask_multiple_tokens():
    text = "PERSON_1 звонил с PHONE_1"
    mapping = {"PERSON_1": "Петров", "PHONE_1": "+79991234567"}
    result, count = unmask_text(text, mapping)
    assert result == "Петров звонил с +79991234567"
    assert count == 2


def test_unmask_long_token_before_short():
    """PERSON_10 must not be partially matched as PERSON_1 + '0'."""
    text = "PERSON_1 и PERSON_10"
    mapping = {"PERSON_1": "Иванов", "PERSON_10": "Сидоров"}
    result, count = unmask_text(text, mapping)
    assert result == "Иванов и Сидоров"
    assert count == 2


def test_unmask_token_not_in_text():
    text = "Привет, мир"
    mapping = {"PERSON_1": "Иванов"}
    result, count = unmask_text(text, mapping)
    assert result == "Привет, мир"
    assert count == 0


def test_unmask_empty_mapping():
    text = "Текст без токенов"
    result, count = unmask_text(text, {})
    assert result == "Текст без токенов"
    assert count == 0


def test_unmask_returns_correct_count():
    text = "INN_1 и EMAIL_1 и EMAIL_2"
    mapping = {"INN_1": "123456789012", "EMAIL_1": "a@b.ru", "EMAIL_2": "c@d.ru"}
    _, count = unmask_text(text, mapping)
    assert count == 3
