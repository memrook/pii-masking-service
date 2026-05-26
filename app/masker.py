# ============================================================
# masker.py — логика маскирования и демаскирования PII
#
# Поддерживаемые типы сущностей:
#   PERSON    — ФИО (natasha NER)
#   PHONE     — номера телефонов (Presidio + regex)
#   INN       — ИНН физлиц (12 цифр) и юрлиц (10 цифр)
#   OGRN      — ОГРН / ОГРНИП
#   ORG       — названия организаций (natasha NER)
#   EMAIL     — email-адреса (Presidio)
#   PASSPORT  — серия и номер паспорта РФ
#   SNILS     — СНИЛС
#   ADDRESS   — адреса РФ (Presidio + regex)
#   CARD      — номера банковских карт (regex + алгоритм Луна)
#   VIN       — идентификационный номер ТС (ISO 3779)
# ============================================================

import re
import logging

from natasha import Segmenter, NewsEmbedding, NewsNERTagger, Doc
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from .config import settings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------

def _luhn_check(number: str) -> bool:
    """Проверка номера карты по алгоритму Луна."""
    digits = [int(d) for d in number]
    digits.reverse()
    total = sum(
        d - 9 if d > 9 else d
        for i, d in enumerate(digits)
        for d in [d * 2 if i % 2 == 1 else d]
    )
    return total % 10 == 0


# ----------------------------------------------------------
# Кастомные Presidio-recognizer'ы для российских сущностей
# ----------------------------------------------------------

def _make_inn_recognizer() -> PatternRecognizer:
    """ИНН юрлица (10 цифр) и физлица (12 цифр)."""
    return PatternRecognizer(
        supported_entity="INN_RU",
        supported_language="ru",
        patterns=[
            Pattern("INN_12", r"\b\d{12}\b", 0.85),
            Pattern("INN_10", r"\b\d{10}\b", 0.75),
        ],
        context=["инн", "инн:", "inn", "налогоплательщик"],
    )


def _make_ogrn_recognizer() -> PatternRecognizer:
    """ОГРН (13 цифр) и ОГРНИП (15 цифр)."""
    return PatternRecognizer(
        supported_entity="OGRN_RU",
        supported_language="ru",
        patterns=[
            Pattern("OGRN_15", r"\b\d{15}\b", 0.85),
            Pattern("OGRN_13", r"\b\d{13}\b", 0.80),
        ],
        context=["огрн", "огрнип", "ogrn"],
    )


def _make_passport_recognizer() -> PatternRecognizer:
    """Паспорт РФ: серия (4 цифры) номер (6 цифр).

    Покрываемые форматы:
      4511 654321              — пробел
      4511654321               — слитно
      4511 № 654321            — с символом №
      серия 4511 номер 654321  — с подписями (именит.)
      серии 4511 номер 654321  — с подписями (родит.)
      4511 серии 654321        — серия после цифр
      серии 03 10 № 123456     — серия двумя двузначными группами + №
    """
    return PatternRecognizer(
        supported_entity="PASSPORT_RU",
        supported_language="ru",
        patterns=[
            Pattern("PASSPORT_SPLIT_SERIES", r"(?:серия|серии)\s+\d{2}\s+\d{2}\s*[№#]\s*\d{6}", 0.97),
            Pattern("PASSPORT_NOM",          r"\b\d{4}\s*№\s*\d{6}\b", 0.90),
            Pattern("PASSPORT_LABELED",      r"(?:серия|серии)\s+\d{4}\s+номер[а]?\s+\d{6}", 0.95),
            Pattern("PASSPORT_REVERSED",     r"\b\d{4}\s+(?:серия|серии)\s+\d{6}\b", 0.85),
            Pattern("PASSPORT_SPACE",        r"\b\d{4}\s\d{6}\b", 0.85),
            Pattern("PASSPORT_NOSPACE",      r"\b\d{4}\d{6}\b", 0.70),
        ],
        context=["паспорт", "серия", "серии", "номер паспорта", "passport"],
    )


def _make_snils_recognizer() -> PatternRecognizer:
    """СНИЛС: NNN-NNN-NNN NN."""
    return PatternRecognizer(
        supported_entity="SNILS_RU",
        supported_language="ru",
        patterns=[
            Pattern("SNILS", r"\b\d{3}-\d{3}-\d{3}\s\d{2}\b", 0.90),
            Pattern("SNILS_NOSPACE", r"\b\d{11}\b", 0.65),
        ],
        context=["снилс", "страховой", "snils"],
    )


def _make_ru_phone_recognizer() -> PatternRecognizer:
    """Российские телефоны: +7, 8, форматы с дефисами и скобками."""
    return PatternRecognizer(
        supported_entity="PHONE_RU",
        supported_language="ru",
        patterns=[
            Pattern("PHONE_7PLUS", r"(\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}", 0.90),
            Pattern("PHONE_SHORT",  r"\b\d{3}[\s\-]\d{2}[\s\-]\d{2}\b", 0.60),
        ],
    )


def _make_address_recognizer() -> PatternRecognizer:
    """Адрес РФ. Три паттерна покрывают все основные форматы записи.

    FORMAT 1 — с городом (регион опц.): г. Москва, ул. Мира, д. 5, кв. 25
                                         Московская обл., г. Химки, ш. Ленинградское, д. 1
    FORMAT 2 — только улица:            ул. Пушкина, д. 10
    FORMAT 3 — без аббревиатур:         проспект Ленина, 12  (нужен контекст)
    """
    _GEO = (
        r"(?:[\w\-]+(?:\s+[\w\-]+)*\s+(?:обл\.|кр\.|р-на?),\s*)?"   # регион (опц.)
        r"(?:г\.|город|пос\.|с\.)\s+[\w\-]+,\s*"                      # город (обяз.)
    )
    _ST_ABBR = r"(?:ул\.|пр\.|пр-т|просп\.|пер\.|ш\.|наб\.|бул\.|пл\.|мкр\.)"
    _ST_FULL = r"(?:проспект|улица|переулок|шоссе|набережная|бульвар|площадь)"
    _HOUSE   = r"д\.\s*\d+[\w\/\-]*"
    _SUITE   = r"(?:,\s*(?:кв\.|оф\.|корп\.|стр\.)\s*\d+[\w\/]*)?"

    return PatternRecognizer(
        supported_entity="ADDRESS_RU",
        supported_language="ru",
        patterns=[
            Pattern(
                "ADDRESS_FULL",
                rf"{_GEO}(?:{_ST_ABBR}|{_ST_FULL})\s+[\w\s\-\.\"«»]{{2,40}},\s*{_HOUSE}{_SUITE}",
                0.85,
            ),
            Pattern(
                "ADDRESS_STREET_ONLY",
                rf"{_ST_ABBR}\s+[\w\s\-\.\"«»]{{2,40}},\s*{_HOUSE}{_SUITE}",
                0.75,
            ),
            Pattern(
                "ADDRESS_NO_ABBR",
                rf"{_ST_FULL}\s+[\w\s\-]{{2,40}},\s*\d+[а-яёА-ЯЁ]?(?:\/\d+)?",
                0.70,
            ),
        ],
        context=["адрес", "адресу", "прописан", "зарегистрирован", "проживает",
                 "место жительства", "регистрация", "прописка"],
    )


def _make_card_recognizer(language: str = "ru") -> PatternRecognizer:
    """Банковские карты: 13–19 цифр с проверкой алгоритма Луна.

    Покрываемые форматы:
      4111 1111 1111 1111   — пробелы (Visa/Mastercard/Мир)
      4111-1111-1111-1111   — дефисы
      4111111111111111      — слитно
      3714 496353 98431     — Amex (15 цифр, 4-6-5)
    """
    class _CardRecognizer(PatternRecognizer):
        def validate_result(self, pattern_text: str):
            digits = re.sub(r"\D", "", pattern_text)
            if not (13 <= len(digits) <= 19):
                return False
            return _luhn_check(digits)

    return _CardRecognizer(
        supported_entity="CARD",
        supported_language=language,
        patterns=[
            Pattern("CARD_SPACED",  r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b", 0.85),
            Pattern("CARD_AMEX",    r"\b\d{4}[\s\-]\d{6}[\s\-]\d{5}\b",             0.85),
            Pattern("CARD_PLAIN",   r"\b\d{13,19}\b",                                0.50),
        ],
        context=["карта", "карточка", "номер карты", "card", "visa",
                 "mastercard", "мир", "maestro", "оплата", "платёж"],
    )


def _make_vin_recognizer(language: str = "ru") -> PatternRecognizer:
    """VIN транспортного средства по ISO 3779: 17 символов [A-HJ-NPR-Z0-9].

    Буквы I, O, Q исключены стандартом во избежание путаницы с цифрами.
    """
    return PatternRecognizer(
        supported_entity="VIN",
        supported_language=language,
        patterns=[
            Pattern("VIN", r"\b[A-HJ-NPR-Z0-9]{17}\b", 0.70),
        ],
        context=["vin", "вин", "кузов", "птс", "стс", "свидетельство",
                 "транспортное средство", "идентификационный номер"],
    )


# ----------------------------------------------------------
# Инициализация Presidio
# ----------------------------------------------------------

def _build_presidio_analyzer() -> AnalyzerEngine:
    """Создаём AnalyzerEngine с поддержкой русского языка через spaCy."""
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "ru", "model_name": "ru_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_md"},
        ],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["ru", "en"],
    )

    # Регистрируем кастомные recognizer'ы
    for recognizer in [
        _make_inn_recognizer(),
        _make_ogrn_recognizer(),
        _make_passport_recognizer(),
        _make_snils_recognizer(),
        _make_ru_phone_recognizer(),
        _make_address_recognizer(),
        _make_card_recognizer("ru"),
        _make_card_recognizer("en"),
        _make_vin_recognizer("ru"),
        _make_vin_recognizer("en"),
    ]:
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


# Глобальные экземпляры (инициализируются один раз через load_models)
_segmenter: Segmenter | None = None
_ner_tagger: NewsNERTagger | None = None
_presidio_analyzer: AnalyzerEngine | None = None
_presidio_anonymizer = AnonymizerEngine()
_models_loaded = False


def load_models() -> None:
    """Вызывается при старте приложения — загружает все модели."""
    global _segmenter, _ner_tagger, _presidio_analyzer, _models_loaded
    logger.info("Загрузка NLP моделей...")
    _segmenter = Segmenter()
    _ner_tagger = NewsNERTagger(NewsEmbedding())
    _presidio_analyzer = _build_presidio_analyzer()
    _models_loaded = True
    logger.info("Модели загружены успешно")


def is_models_loaded() -> bool:
    return _models_loaded


# ----------------------------------------------------------
# Карта: тип сущности → токен-префикс
# ----------------------------------------------------------
_ENTITY_TOKEN_MAP = {
    "PERSON":        settings.token_person,
    "PER":           settings.token_person,    # natasha использует PER
    "INN_RU":        settings.token_inn,
    "OGRN_RU":       settings.token_ogrn,
    "ORG":           settings.token_org,
    "EMAIL_ADDRESS": settings.token_email,
    "PHONE_NUMBER":  settings.token_phone,
    "PHONE_RU":      settings.token_phone,
    "PASSPORT_RU":   settings.token_passport,
    "SNILS_RU":      settings.token_snils,
    "ADDRESS_RU":    settings.token_address,
    "CARD":          settings.token_card,
    "CREDIT_CARD":   settings.token_card,      # Presidio built-in (EN)
    "VIN":           settings.token_vin,
}


# ----------------------------------------------------------
# Кеш regex-паттернов для find_safe_boundary
# ----------------------------------------------------------

_TAIL_PATTERN_CACHE: dict[tuple[str, ...], re.Pattern] = {}


def _get_token_prefixes() -> list[str]:
    """Возвращает все уникальные токен-префиксы из текущих настроек."""
    return list(dict.fromkeys([
        settings.token_person, settings.token_phone, settings.token_inn,
        settings.token_ogrn, settings.token_org, settings.token_email,
        settings.token_passport, settings.token_snils, settings.token_address,
        settings.token_card, settings.token_vin,
    ]))


def _build_tail_pattern(prefixes: list[str]) -> re.Pattern:
    parts: list[str] = []
    for p in prefixes:
        for i in range(1, len(p)):  # частичные: "P", "PE", ..., "PERSO"
            parts.append(re.escape(p[:i]))
        parts.append(re.escape(p))            # полный без "_": "PERSON"
        parts.append(re.escape(p) + r"_\d*")  # "PERSON_", "PERSON_1", "PERSON_15"
    parts.sort(key=len, reverse=True)
    return re.compile(r"(?:" + "|".join(parts) + r")$")


def _get_or_build_tail_pattern(prefixes: list[str]) -> re.Pattern:
    key = tuple(prefixes)
    cached = _TAIL_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached
    pattern = _build_tail_pattern(prefixes)
    _TAIL_PATTERN_CACHE[key] = pattern
    return pattern


def find_safe_boundary(text: str, prefixes: list[str]) -> int:
    """Возвращает позицию, до которой текст безопасно демаскировать.

    Держит в буфере хвост, который может оказаться началом незавершённого токена.
    """
    if not text:
        return 0
    pattern = _get_or_build_tail_pattern(prefixes)
    m = pattern.search(text)
    return m.start() if m else len(text)


# ----------------------------------------------------------
# Основная функция маскирования
# ----------------------------------------------------------

def mask_text(
    text: str,
    session_id: str,
    language: str = "ru",
    existing_mapping: dict[str, str] | None = None,
) -> tuple[str, dict[str, str], list[str]]:
    """
    Маскирует ПД в тексте.

    При передаче existing_mapping восстанавливает счётчики из него и переиспользует
    токены для тех же оригинальных значений (гарантирует, что «Иванов» всегда → PERSON_1).

    Returns:
        masked_text  — текст с токенами вместо ПД
        mapping      — полный {TOKEN: original_value} включая существующие
        entity_types — список типов сущностей, найденных в этом вызове
    """
    if not _models_loaded:
        raise RuntimeError("Модели не загружены. Вызовите load_models() при старте.")

    mapping: dict[str, str] = dict(existing_mapping or {})
    entity_types_found: list[str] = []

    # Восстанавливаем счётчики и обратный индекс из существующего mapping
    counters: dict[str, int] = {}
    reverse: dict[tuple[str, str], str] = {}
    for token, original in mapping.items():
        m = re.match(r"^([A-Z]+)_(\d+)$", token)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            counters[prefix] = max(counters.get(prefix, 0), num)
            reverse[(prefix, original)] = token

    def _allocate(prefix: str, original: str) -> str:
        """Возвращает существующий токен для original или создаёт новый."""
        existing = reverse.get((prefix, original))
        if existing:
            return existing
        counters[prefix] = counters.get(prefix, 0) + 1
        new_token = f"{prefix}_{counters[prefix]}"
        mapping[new_token] = original
        reverse[(prefix, original)] = new_token
        return new_token

    # --- Шаг 1: Natasha NER (Person + Org для русского) ---
    natasha_spans: list[tuple[int, int, str, str]] = []  # (start, end, type, value)

    if language == "ru":
        doc = Doc(text)
        doc.segment(_segmenter)
        doc.tag_ner(_ner_tagger)

        for span in doc.spans:
            if span.type in ("PER", "ORG"):
                entity_key = _ENTITY_TOKEN_MAP.get(span.type, span.type)
                token = _allocate(entity_key, span.text)
                natasha_spans.append((span.start, span.stop, span.type, token))
                if span.type not in entity_types_found:
                    entity_types_found.append(span.type)

    # --- Шаг 2: Presidio (телефоны, ИНН, ОГРН, email, паспорт, СНИЛС) ---
    results = _presidio_analyzer.analyze(
        text=text,
        language=language,
        entities=[
            "INN_RU", "OGRN_RU", "PHONE_NUMBER", "PHONE_RU",
            "EMAIL_ADDRESS", "PASSPORT_RU", "SNILS_RU", "ADDRESS_RU",
            "PERSON",        # Presidio тоже умеет находить имена (en)
            "CARD", "CREDIT_CARD",
            "VIN",
        ],
    )

    presidio_spans: list[tuple[int, int, str, str]] = []
    for result in results:
        # Проверяем, не перекрывается ли с уже найденным natasha
        overlaps = any(
            not (result.end <= ns[0] or result.start >= ns[1])
            for ns in natasha_spans
        )
        if overlaps:
            continue

        original = text[result.start:result.end]
        entity_key = _ENTITY_TOKEN_MAP.get(result.entity_type, result.entity_type)
        token = _allocate(entity_key, original)
        presidio_spans.append((result.start, result.end, result.entity_type, token))

        if result.entity_type not in entity_types_found:
            entity_types_found.append(result.entity_type)

    # --- Шаг 3: Применяем замены справа налево (сохраняем позиции) ---
    all_spans = sorted(natasha_spans + presidio_spans, key=lambda x: x[0], reverse=True)

    masked = text
    for start, end, _, token in all_spans:
        masked = masked[:start] + token + masked[end:]

    return masked, mapping, entity_types_found


# ----------------------------------------------------------
# Функция демаскирования
# ----------------------------------------------------------

def unmask_text(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """
    Восстанавливает оригинальные значения из токенов.

    Returns:
        unmasked_text  — восстановленный текст
        tokens_replaced — количество замен
    """
    result = text
    count = 0

    # Сортируем по убыванию длины токена — избегаем частичных замен
    # (например, PERSON_10 не должен матчиться раньше PERSON_1)
    for token, original in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
        if token in result:
            result = result.replace(token, original)
            count += 1

    return result, count
