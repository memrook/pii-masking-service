# tests/test_surrogates.py — генераторы суррогатов
import re

from app.surrogates import generate_surrogate, _luhn_check_digit

SECRET = "unit-test-secret"


# ── независимые валидаторы контрольных сумм ───────────────────────────────────

def _valid_inn(s: str) -> bool:
    d = [int(c) for c in s]
    if len(s) == 10:
        w = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        return (sum(x * y for x, y in zip(d, w)) % 11) % 10 == d[9]
    if len(s) == 12:
        w11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        w12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
        n11 = (sum(x * y for x, y in zip(d, w11)) % 11) % 10
        n12 = (sum(x * y for x, y in zip(d, w12)) % 11) % 10
        return n11 == d[10] and n12 == d[11]
    return False


def _valid_luhn(s: str) -> bool:
    digits = [int(c) for c in s]
    total = 0
    for i, dd in enumerate(reversed(digits)):
        if i % 2 == 1:
            dd *= 2
            if dd > 9:
                dd -= 9
        total += dd
    return total % 10 == 0


def _valid_snils(digits: str) -> bool:
    d = [int(c) for c in digits[:9]]
    total = sum(x * (9 - i) for i, x in enumerate(d))
    rem = total % 101
    check = 0 if rem in (100, 101) else rem
    return f"{check:02d}" == digits[9:11]


def gen(ctype, original, session="s1", source="", occupied=None):
    return generate_surrogate(ctype, original, session, source, occupied or set(), secret=SECRET)


# ── контрольные суммы / формат ────────────────────────────────────────────────

def test_inn_10_valid_and_length():
    out = gen("INN", "7707083893")           # 10-значный оригинал
    assert len(out) == 10 and _valid_inn(out)


def test_inn_12_valid_and_length():
    out = gen("INN", "772512345678")          # 12-значный оригинал
    assert len(out) == 12 and _valid_inn(out)


def test_card_luhn_and_length_follows_original():
    out = gen("CARD", "4111 1111 1111 1111")
    digits = re.sub(r"\D", "", out)
    assert len(digits) == 16 and _valid_luhn(digits)
    # разбивка по группам как у оригинала (4-4-4-4 через пробел)
    assert re.fullmatch(r"\d{4} \d{4} \d{4} \d{4}", out)


def test_card_amex_15_length_preserved():
    out = gen("CARD", "3714 496353 98431")
    digits = re.sub(r"\D", "", out)
    assert len(digits) == 15 and _valid_luhn(digits)


def test_snils_with_dashes_format_and_checksum():
    out = gen("SNILS", "112-233-445 95")
    assert re.fullmatch(r"\d{3}-\d{3}-\d{3} \d{2}", out)
    assert _valid_snils(re.sub(r"\D", "", out))


def test_snils_plain_11_digits():
    out = gen("SNILS", "11223344595")
    assert re.fullmatch(r"\d{11}", out)
    assert _valid_snils(out)


def test_ogrn_13_and_ogrnip_15_lengths():
    assert len(gen("OGRN", "1027700132195")) == 13
    assert len(gen("OGRN", "304500116000157")) == 15


def test_bik_9_digits_prefix_04():
    out = gen("BIK", "044525225")
    assert re.fullmatch(r"04\d{7}", out)


def test_kpp_9_digits():
    assert re.fullmatch(r"\d{9}", gen("KPP", "773601001"))


def test_account_20_digits():
    assert re.fullmatch(r"\d{20}", gen("ACCOUNT", "40702810000000000111"))


def test_vin_17_iso_alphabet_no_ioq():
    out = gen("VIN", "1HGBH41JXMN109186")
    assert len(out) == 17
    assert re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", out)
    assert not set("IOQ") & set(out)


def test_passport_format():
    assert re.fullmatch(r"\d{4} \d{6}", gen("PASSPORT", "4511 654321"))


def test_phone_format_with_555_block():
    out = gen("PHONE", "+79991234567")
    assert re.fullmatch(r"\+7 9\d{2} 555-\d{2}-\d{2}", out)


def test_email_reserved_domain():
    out = gen("EMAIL", "ivan@company.ru")
    assert out.endswith("@example.com")
    assert re.fullmatch(r"[a-z]+\d*@example\.com", out)


# ── детерминизм / межсессионность / коллизии ──────────────────────────────────

def test_deterministic_within_session():
    a = gen("INN", "772512345678", session="sX")
    b = gen("INN", "772512345678", session="sX")
    assert a == b


def test_different_across_sessions():
    a = gen("INN", "772512345678", session="sA")
    b = gen("INN", "772512345678", session="sB")
    assert a != b


def test_collision_with_source_text_forces_reseed():
    # принудим коллизию: первый кандидат добавим в source как подстроку
    first = gen("INN", "772512345678", session="sC")
    out = gen("INN", "772512345678", session="sC", source=f"в тексте есть {first} число")
    assert out != first


def test_collision_with_occupied_outputs():
    first = gen("PHONE", "+79990000001", session="sD")
    out = gen("PHONE", "+79990000002", session="sD", occupied={first})
    assert out != first


def test_luhn_check_digit_helper():
    # 7992739871 → контрольная 3 (классический пример Луна)
    assert _luhn_check_digit([7, 9, 9, 2, 7, 3, 9, 8, 7, 1]) == 3
