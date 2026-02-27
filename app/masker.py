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
    """
    return PatternRecognizer(
        supported_entity="PASSPORT_RU",
        supported_language="ru",
        patterns=[
            Pattern("PASSPORT_NOM",      r"\b\d{4}\s*№\s*\d{6}\b", 0.90),
            Pattern("PASSPORT_LABELED",  r"(?:серия|серии)\s+\d{4}\s+номер[а]?\s+\d{6}", 0.95),
            Pattern("PASSPORT_REVERSED", r"\b\d{4}\s+(?:серия|серии)\s+\d{6}\b", 0.85),
            Pattern("PASSPORT_SPACE",    r"\b\d{4}\s\d{6}\b", 0.85),
            Pattern("PASSPORT_NOSPACE",  r"\b\d{4}\d{6}\b", 0.70),
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


# ----------------------------------------------------------
# Инициализация Presidio
# ----------------------------------------------------------

def _build_presidio_analyzer() -> AnalyzerEngine:
    """Создаём AnalyzerEngine с поддержкой русского языка через spaCy."""
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "ru", "model_name": "ru_core_news_lg"},
            {"lang_code": "en", "model_name": "en_core_web_lg"},
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
    "PERSON":       settings.token_person,
    "PER":          settings.token_person,    # natasha использует PER
    "INN_RU":       settings.token_inn,
    "OGRN_RU":      settings.token_ogrn,
    "ORG":          settings.token_org,
    "EMAIL_ADDRESS": settings.token_email,
    "PHONE_NUMBER": settings.token_phone,
    "PHONE_RU":     settings.token_phone,
    "PASSPORT_RU":  settings.token_passport,
    "SNILS_RU":     settings.token_snils,
    "ADDRESS_RU":   settings.token_address,
}


# ----------------------------------------------------------
# Основная функция маскирования
# ----------------------------------------------------------

def mask_text(text: str, session_id: str, language: str = "ru") -> tuple[str, dict[str, str], list[str]]:
    """
    Маскирует ПД в тексте.

    Returns:
        masked_text  — текст с токенами вместо ПД
        mapping      — {TOKEN: original_value} для последующего демаскирования
        entity_types — список найденных типов сущностей
    """
    if not _models_loaded:
        raise RuntimeError("Модели не загружены. Вызовите load_models() при старте.")

    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    entity_types_found: list[str] = []

    # --- Шаг 1: Natasha NER (Person + Org для русского) ---
    natasha_spans: list[tuple[int, int, str, str]] = []  # (start, end, type, value)

    if language == "ru":
        doc = Doc(text)
        doc.segment(_segmenter)
        doc.tag_ner(_ner_tagger)

        for span in doc.spans:
            if span.type in ("PER", "ORG"):
                entity_key = _ENTITY_TOKEN_MAP.get(span.type, span.type)
                counters[entity_key] = counters.get(entity_key, 0) + 1
                token = f"{entity_key}_{counters[entity_key]}"
                mapping[token] = span.text
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
            "PERSON",      # Presidio тоже умеет находить имена (en)
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
        counters[entity_key] = counters.get(entity_key, 0) + 1
        token = f"{entity_key}_{counters[entity_key]}"
        mapping[token] = original
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
