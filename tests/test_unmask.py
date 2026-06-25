# tests/test_unmask.py — однопроходное демаскирование
from app.masker import unmask_text


def test_unmask_single_marker():
    text = "Позвони [Имя 1] завтра"
    result, count = unmask_text(text, {"[Имя 1]": "Иванову"})
    assert result == "Позвони Иванову завтра"
    assert count == 1


def test_unmask_marker_and_surrogate():
    text = "[Имя 1] звонил с +7 921 555-12-34"
    out_map = {"[Имя 1]": "Петров", "+7 921 555-12-34": "+79991234567"}
    result, count = unmask_text(text, out_map)
    assert result == "Петров звонил с +79991234567"
    assert count == 2


def test_unmask_longest_match_first():
    """'[Имя 10]' не должен частично матчиться как '[Имя 1]' + '0]'."""
    text = "[Имя 1] и [Имя 10]"
    out_map = {"[Имя 1]": "Иванов", "[Имя 10]": "Сидоров"}
    result, count = unmask_text(text, out_map)
    assert result == "Иванов и Сидоров"
    assert count == 2


def test_unmask_token_not_in_text():
    result, count = unmask_text("Привет, мир", {"[Имя 1]": "Иванов"})
    assert result == "Привет, мир"
    assert count == 0


def test_unmask_empty_mapping():
    result, count = unmask_text("Текст без токенов", {})
    assert result == "Текст без токенов"
    assert count == 0


def test_unmask_counts_occurrences_not_keys():
    """tokens_replaced считает фактические вхождения, а не число ключей."""
    text = "6591557797 и 6591557797 и user@example.com"
    out_map = {"6591557797": "772512345678", "user@example.com": "real@mail.ru"}
    _, count = unmask_text(text, out_map)
    assert count == 3  # ИНН дважды + email один раз


def test_unmask_no_cascade():
    """Восстановленный original содержит чужой out — он НЕ заменяется повторно."""
    # out "AAA" → original "BBB-CCC"; out "CCC" → original "ZZZ".
    # Однопроходный re.sub не должен превратить вставленный "CCC" в "ZZZ".
    text = "AAA"
    out_map = {"AAA": "BBB-CCC", "CCC": "ZZZ"}
    result, count = unmask_text(text, out_map)
    assert result == "BBB-CCC"
    assert count == 1
