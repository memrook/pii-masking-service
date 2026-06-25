# ============================================================
# surrogates.py — генераторы синтаксически правдоподобных
#                 тестовых значений (суррогатов) для PII
#
# ВАЖНО: это НЕ «валидные реквизиты», а правдоподобные ТЕСТОВЫЕ значения.
# Контрольные суммы вычисляются лишь для того, чтобы значение выглядело
# правдоподобно и LLM воспроизводила его дословно. Согласованность пар
# (ACCOUNT/BIK), контрольная цифра VIN и т.п. как настоящие НЕ заявляются.
#
# Детерминизм: суррогат стабилен ВНУТРИ сессии (один PII → один суррогат),
# но не коррелирует между сессиями (один PII у разных пользователей →
# разные суррогаты). Достигается HMAC-сидом с session_id.
# ============================================================

import hashlib
import hmac
import random

from .config import settings


# ----------------------------------------------------------
# Сид
# ----------------------------------------------------------

def _seed(secret: str, session_id: str, canonical_type: str,
          normalized_original: str, salt: int) -> int:
    """HMAC-SHA256 сид. Разделители \\0 исключают неоднозначность склейки полей."""
    msg = f"{session_id}\0{canonical_type}\0{normalized_original}\0{salt}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    return int.from_bytes(digest, "big")


# ----------------------------------------------------------
# Генераторы по типам (каждый принимает rng и original)
# original нужен только чтобы повторить разрядность/формат.
# ----------------------------------------------------------

def _digits(rng: random.Random, n: int) -> list[int]:
    return [rng.randint(0, 9) for _ in range(n)]


def _gen_inn(rng: random.Random, original: str) -> str:
    """ИНН: 10 цифр (юрлицо) или 12 (физлицо) по длине оригинала."""
    n = sum(c.isdigit() for c in original)
    length = 12 if n >= 12 else 10
    if length == 10:
        w = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        d = _digits(rng, 9)
        c = (sum(x * y for x, y in zip(d, w)) % 11) % 10
        d.append(c)
        return "".join(map(str, d))
    # 12 цифр: две контрольные
    w11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    w12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]
    d = _digits(rng, 10)
    n11 = (sum(x * y for x, y in zip(d, w11)) % 11) % 10
    d.append(n11)
    n12 = (sum(x * y for x, y in zip(d, w12)) % 11) % 10
    d.append(n12)
    return "".join(map(str, d))


def _gen_ogrn(rng: random.Random, original: str) -> str:
    """ОГРН (13) / ОГРНИП (15) по длине оригинала."""
    n = sum(c.isdigit() for c in original)
    length = 15 if n >= 15 else 13
    body_len = length - 1
    # Первая цифра 1..9 чтобы не было ведущего нуля
    body = [rng.randint(1, 9)] + _digits(rng, body_len - 1)
    body_num = int("".join(map(str, body)))
    mod = 13 if length == 15 else 11
    check = (body_num % mod) % 10
    return "".join(map(str, body)) + str(check)


def _gen_snils(rng: random.Random, original: str) -> str:
    """СНИЛС: контрольная сумма по mod 101. Формат повторяет оригинал."""
    d = _digits(rng, 9)
    total = sum(x * (9 - i) for i, x in enumerate(d))
    rem = total % 101
    check = 0 if rem in (100, 101) else rem
    digits = "".join(map(str, d)) + f"{check:02d}"
    if "-" in original:
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:9]} {digits[9:11]}"
    return digits


def _luhn_check_digit(body: list[int]) -> int:
    """Контрольная цифра Луна для последовательности без неё."""
    total = 0
    # позиция справа: первая (самая правая из body) удваивается
    for i, d in enumerate(reversed(body)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def _gen_card(rng: random.Random, original: str) -> str:
    """Карта: длина 13–19 по оригиналу, контрольная цифра Луна. Формат — как оригинал."""
    n = sum(c.isdigit() for c in original)
    length = n if 13 <= n <= 19 else 16
    body = [rng.randint(1, 9)] + _digits(rng, length - 2)
    body.append(_luhn_check_digit(body))
    digits = "".join(map(str, body))
    # Повторяем разбивку оригинала по группам, если она была через пробел/дефис
    sep = " " if " " in original else ("-" if "-" in original else "")
    if sep:
        groups, i = [], 0
        for part in original.replace("-", " ").split():
            ln = sum(c.isdigit() for c in part)
            groups.append(digits[i:i + ln])
            i += ln
        if i == len(digits):
            return sep.join(groups)
    return digits


def _gen_bik(rng: random.Random, original: str) -> str:
    """БИК: 9 цифр, префикс 04 (территория РФ). Валидность не заявляется."""
    return "04" + "".join(map(str, _digits(rng, 7)))


def _gen_kpp(rng: random.Random, original: str) -> str:
    """КПП: 9 цифр (NNNN PP XXX), без пробелов. Правдоподобный формат."""
    tax = [rng.randint(1, 9)] + _digits(rng, 3)        # код налогового органа
    reason = [0, rng.randint(1, 5)]                     # причина постановки
    serial = _digits(rng, 3)
    return "".join(map(str, tax + reason + serial))


def _gen_account(rng: random.Random, original: str) -> str:
    """Расчётный счёт: 20 цифр, префикс 40 (балансовый счёт). Без заявления валидности."""
    return "40" + "".join(map(str, _digits(rng, 18)))


_VIN_ALPHABET = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # без I, O, Q (ISO 3779)


def _gen_vin(rng: random.Random, original: str) -> str:
    """VIN: 17 символов из алфавита ISO 3779. Контрольная цифра не заявляется."""
    return "".join(rng.choice(_VIN_ALPHABET) for _ in range(17))


def _gen_passport(rng: random.Random, original: str) -> str:
    """Паспорт РФ: серия (4) + номер (6). Формат 'DDDD DDDDDD'."""
    series = "".join(map(str, [rng.randint(1, 9)] + _digits(rng, 3)))
    number = "".join(map(str, _digits(rng, 6)))
    return f"{series} {number}"


_EMAIL_CONS = "bcdfghklmnprstv"
_EMAIL_VOWELS = "aeiou"


def _gen_email(rng: random.Random, original: str) -> str:
    """Email: произносимый латинский local-part + зарезервированный домен example.com."""
    def syllable() -> str:
        return rng.choice(_EMAIL_CONS) + rng.choice(_EMAIL_VOWELS)
    local = "".join(syllable() for _ in range(rng.randint(2, 4)))
    if rng.random() < 0.5:
        local += str(rng.randint(1, 99))
    return f"{local}@example.com"


def _gen_phone(rng: random.Random, original: str) -> str:
    """Телефон РФ: +7 9XX 555-XX-XX.

    Блок 555 в номерной части — практика «непубликуемых» тестовых номеров,
    минимизирует риск совпадения с реальным абонентом. Формат фиксированный.
    """
    code = "".join(map(str, _digits(rng, 2)))      # 9XX → 9 + 2 цифры
    a = "".join(map(str, _digits(rng, 2)))
    b = "".join(map(str, _digits(rng, 2)))
    return f"+7 9{code} 555-{a}-{b}"


_GENERATORS = {
    "INN": _gen_inn,
    "OGRN": _gen_ogrn,
    "SNILS": _gen_snils,
    "CARD": _gen_card,
    "BIK": _gen_bik,
    "KPP": _gen_kpp,
    "ACCOUNT": _gen_account,
    "VIN": _gen_vin,
    "PASSPORT": _gen_passport,
    "EMAIL": _gen_email,
    "PHONE": _gen_phone,
}


def supported_types() -> frozenset[str]:
    return frozenset(_GENERATORS.keys())


# ----------------------------------------------------------
# Публичный API
# ----------------------------------------------------------

def generate_surrogate(
    canonical_type: str,
    original: str,
    session_id: str,
    source_text: str,
    occupied_outputs: set[str],
    secret: str | None = None,
) -> str:
    """Возвращает детерминированный суррогат для original.

    Args:
        canonical_type: канонический тип (см. config.ALL_CANONICAL_TYPES).
        original: исходное значение (для определения разрядности/формата).
        session_id: ID сессии — входит в сид (стабильность внутри сессии).
        source_text: исходный текст — кандидат отвергается, если встречается в нём
                     как подстрока (иначе unmask затронет настоящий фрагмент текста).
        occupied_outputs: уже занятые out-значения сессии (исключаем коллизии,
                          в т.ч. подстрочные в обе стороны).
        secret: HMAC-секрет; по умолчанию settings.surrogate_secret.

    Raises:
        KeyError: если для canonical_type нет генератора.
    """
    gen = _GENERATORS[canonical_type]
    secret = secret if secret is not None else settings.surrogate_secret
    normalized = original.strip()

    salt = 0
    while True:
        rng = random.Random(_seed(secret, session_id, canonical_type, normalized, salt))
        candidate = gen(rng, original)
        if candidate not in source_text and not _collides(candidate, occupied_outputs):
            return candidate
        salt += 1


def _collides(candidate: str, occupied: set[str]) -> bool:
    """Коллизия, если candidate и любое занятое значение — подстроки друг друга.

    Двусторонняя проверка защищает однопроходный unmask от ситуации, когда один
    суррогат является подстрокой другого и приводит к неверной замене.
    """
    for o in occupied:
        if candidate == o or candidate in o or o in candidate:
            return True
    return False
