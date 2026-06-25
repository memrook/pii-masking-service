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
from .surrogates import generate_surrogate

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
    """Российские телефоны: +7, 8, форматы с дефисами и скобками.

    Lookbehind (?<!\\d) и lookahead (?!\\d) защищают от срабатывания на
    подстроку внутри длинной цифровой последовательности (расчётный счёт и т.п.).
    """
    return PatternRecognizer(
        supported_entity="PHONE_RU",
        supported_language="ru",
        patterns=[
            Pattern(
                "PHONE_7PLUS",
                r"(?<!\d)(\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)",
                0.90,
            ),
            Pattern("PHONE_SHORT", r"\b\d{3}[\s\-]\d{2}[\s\-]\d{2}\b", 0.60),
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


class _ContextRequiredRecognizer(PatternRecognizer):
    """PatternRecognizer, валидирующий каждый матч по строгому контексту слева.

    context-слово должно встречаться в LEFT_WINDOW символах перед матчем, и
    между концом context-слова и началом матча не должно быть другого цифрового
    блока (≥ NOISE_DIGITS цифр). Защищает от cross-firing когда два recognizer'а
    делят один regex (BIK_RU/KPP_RU оба матчат \\d{9}) и context-слова обоих
    находятся в соседних блоках реквизитов.
    """

    LEFT_WINDOW = 30
    NOISE_DIGITS = 4

    def analyze(self, text, entities, nlp_artifacts=None, regex_flags=None):
        results = super().analyze(text, entities, nlp_artifacts, regex_flags)
        if not self.context:
            return results

        ctx_terms = [c.lower() for c in self.context]
        text_lower = text.lower()
        noise_re = re.compile(rf"\d{{{self.NOISE_DIGITS},}}")
        kept = []

        for r in results:
            ws = max(0, r.start - self.LEFT_WINDOW)
            left_window = text_lower[ws:r.start]

            best_end_in_text = -1
            for term in ctx_terms:
                pos = left_window.rfind(term)
                if pos < 0:
                    continue
                end_in_text = ws + pos + len(term)
                if end_in_text > best_end_in_text:
                    best_end_in_text = end_in_text

            if best_end_in_text < 0:
                continue
            if noise_re.search(text[best_end_in_text:r.start]):
                continue
            kept.append(r)
        return kept


def _make_account_recognizer() -> PatternRecognizer:
    """Расчётный/корреспондентский счёт РФ: 20 цифр.

    Паттерн `\\b\\d{20}\\b` сам по себе очень специфичен (20 цифр подряд за
    пределами банковских реквизитов — редкость), поэтому context используется
    только для score-boost через стандартный механизм Presidio, но не обязателен.
    """
    return PatternRecognizer(
        supported_entity="ACCOUNT_RU",
        supported_language="ru",
        patterns=[Pattern("ACCOUNT_20", r"\b\d{20}\b", 0.85)],
        context=["счёт", "счет", "р/с", "к/с", "расчёт", "расчет", "корреспондент"],
    )


def _make_bik_recognizer() -> PatternRecognizer:
    """БИК: 9 цифр. Контекстное слово 'БИК' слева обязательно."""
    return _ContextRequiredRecognizer(
        supported_entity="BIK_RU",
        supported_language="ru",
        patterns=[Pattern("BIK_9", r"\b\d{9}\b", 0.85)],
        context=["бик", "bik"],
    )


def _make_kpp_recognizer() -> PatternRecognizer:
    """КПП: 9 цифр. Контекстное слово 'КПП' слева обязательно."""
    return _ContextRequiredRecognizer(
        supported_entity="KPP_RU",
        supported_language="ru",
        patterns=[Pattern("KPP_9", r"\b\d{9}\b", 0.85)],
        context=["кпп", "kpp"],
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
        _make_account_recognizer(),
        _make_bik_recognizer(),
        _make_kpp_recognizer(),
    ]:
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


# ----------------------------------------------------------
# Вспомогательные функции для mask_text
# ----------------------------------------------------------

def _presidio_entities(language: str) -> list[str]:
    """Entity types to request from Presidio depending on language.

    For Russian: PHONE_RU (precise +7/8 patterns) replaces aggressive built-in
    PHONE_NUMBER; PERSON is omitted — natasha NER handles PER/ORG instead.
    For English: built-in PHONE_NUMBER and PERSON are appropriate.
    """
    common = [
        "INN_RU", "OGRN_RU", "EMAIL_ADDRESS", "PASSPORT_RU", "SNILS_RU",
        "ADDRESS_RU", "CARD", "CREDIT_CARD", "VIN",
        "ACCOUNT_RU", "BIK_RU", "KPP_RU",
    ]
    if language == "ru":
        return common + ["PHONE_RU"]
    return common + ["PHONE_NUMBER", "PERSON"]


def _resolve_presidio_overlaps(results: list) -> list:
    """Resolve overlapping Presidio spans, keeping the highest-priority one.

    Priority (descending): custom _RU entities > built-in; higher score; longer span.
    Greedy left-to-right selection after sorting by priority.
    """
    _CUSTOM = frozenset({
        "INN_RU", "OGRN_RU", "PASSPORT_RU", "SNILS_RU", "ADDRESS_RU",
        "PHONE_RU", "CARD", "VIN",
        "ACCOUNT_RU", "BIK_RU", "KPP_RU",
    })

    def _key(r):
        return (r.entity_type in _CUSTOM, r.score, r.end - r.start)

    kept: list = []
    for result in sorted(results, key=_key, reverse=True):
        if not any(
            not (result.end <= k.start or result.start >= k.end)
            for k in kept
        ):
            kept.append(result)
    return kept


def _clip_natasha_span(text: str, start: int, stop: int) -> tuple[int, int]:
    """Клиппит natasha-спан по первому переносу строки и обрезает кромочные пробелы.

    Natasha NER иногда жадно растягивает PER/ORG через перенос строки, захватывая
    слова со следующей строки (напр. ORG = 'АО «Банк» \\n БИК'). Замена такого спана
    меткой съедала бы перенос строки и чужой текст. Сущность реально не пересекает
    строку — клиппим до первого '\\n'.
    """
    seg = text[start:stop]
    nl = seg.find("\n")
    if nl != -1:
        stop = start + nl
    while stop > start and text[stop - 1].isspace():
        stop -= 1
    while start < stop and text[start].isspace():
        start += 1
    return start, stop


def _is_natasha_fp(span_text: str) -> bool:
    """True if the natasha NER span is a likely false positive.

    Single all-caps words (document headers like ДОГОВОР, СТОРОНЫ) are not
    real entities — they're document structure keywords that confuse NER.
    """
    words = span_text.split()
    return len(words) == 1 and len(span_text) > 1 and span_text.isupper()


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
# Карта: тип сущности детектора → канонический тип
# ----------------------------------------------------------
# Разные детекторы дают разные entity_type для одной сущности (PER/PERSON,
# PHONE_RU/PHONE_NUMBER, CARD/CREDIT_CARD). Каноника сводит их к единому типу.
_CANONICAL_TYPE = {
    "PERSON":        "PERSON",
    "PER":           "PERSON",    # natasha использует PER
    "INN_RU":        "INN",
    "OGRN_RU":       "OGRN",
    "ORG":           "ORG",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER":  "PHONE",
    "PHONE_RU":      "PHONE",
    "PASSPORT_RU":   "PASSPORT",
    "SNILS_RU":      "SNILS",
    "ADDRESS_RU":    "ADDRESS",
    "CARD":          "CARD",
    "CREDIT_CARD":   "CARD",      # Presidio built-in (EN)
    "VIN":           "VIN",
    "ACCOUNT_RU":    "ACCOUNT",
    "BIK_RU":        "BIK",
    "KPP_RU":        "KPP",
}


def find_safe_boundary(text: str, outs: list[str]) -> int:
    """Позиция, до которой текст безопасно демаскировать (потоковый режим).

    Держит в хвосте суффикс, который является СОБСТВЕННЫМ ПРЕФИКСОМ хотя бы
    одного out — то есть может оказаться началом незавершённого out, который
    «дорастёт» в следующем чанке.

    Префиксная неоднозначность: если один out является префиксом другого
    (например "12345" ⊂ "123456"), то полностью совпавший короткий out в конце
    чанка тоже удерживается (он — собственный префикс длинного), пока
    продолжение не исключено.

    Args:
        text: объединённый хвост + текущий чанк.
        outs: фактические out-значения текущей сессии.
    """
    if not text:
        return 0
    outs = [o for o in outs if o]
    if not outs:
        return len(text)

    n = len(text)
    max_out_len = max(len(o) for o in outs)
    # Собственный префикс короче самого out, значит держим максимум max_out_len-1.
    for L in range(min(n, max_out_len - 1), 0, -1):
        suffix = text[n - L:]
        if any(len(suffix) < len(o) and o.startswith(suffix) for o in outs):
            return n - L
    return n


# ----------------------------------------------------------
# Основная функция маскирования
# ----------------------------------------------------------

def mask_text(
    text: str,
    session_id: str,
    language: str = "ru",
    existing_entries: list[dict] | None = None,
) -> tuple[str, list[dict], list[str], list[dict]]:
    """
    Маскирует ПД в тексте по гибридной схеме (суррогаты + метки).

    Суррогатные типы (цифры/коды) заменяются реалистичными тестовыми значениями;
    marker-типы (имена/орг/адреса) — инертными метками вида "[Имя 1]".

    При передаче existing_entries восстанавливает счётчики меток и обратный индекс,
    переиспользуя тот же out для того же оригинала (идемпотентность в пределах сессии).

    Args:
        text: исходный текст.
        session_id: ID сессии (входит в сид суррогатов).
        language: 'ru' или 'en' (определяет подписи меток и pipeline детекции).
        existing_entries: записи mapping из предыдущих /mask этой сессии.

    Returns:
        masked_text   — текст с суррогатами/метками вместо ПД
        entries       — полный список записей mapping (включая существующие)
        entity_types  — канонические типы, найденные в этом вызове
        masked_spans  — диапазоны out в ИТОГОВОМ masked_text (code points)
    """
    if not _models_loaded:
        raise RuntimeError("Модели не загружены. Вызовите load_models() при старте.")

    entries: list[dict] = list(existing_entries or [])
    entity_types_found: list[str] = []

    # Восстанавливаем счётчики меток и обратный индекс из существующих записей
    counters: dict[str, int] = {}
    reverse: dict[tuple[str, str], str] = {}
    occupied: set[str] = set()
    for e in entries:
        reverse[(e["type"], e["original"])] = e["out"]
        occupied.add(e["out"])
        if e.get("kind") == "marker":
            counters[e["type"]] = max(counters.get(e["type"], 0), e.get("index", 0))

    labels = settings.marker_labels(language)
    marker_set = set(settings.marker_types)

    def _make_marker(ctype: str) -> tuple[str, int]:
        """Создаёт уникальную метку, обходя коллизии с текстом и занятыми out."""
        label = labels.get(ctype, ctype)
        while True:
            counters[ctype] = counters.get(ctype, 0) + 1
            candidate = f"{settings.marker_open}{label} {counters[ctype]}{settings.marker_close}"
            if candidate not in text and candidate not in occupied:
                return candidate, counters[ctype]

    def _allocate(canonical_type: str, original: str) -> str:
        """Возвращает существующий out для original или создаёт новый."""
        existing = reverse.get((canonical_type, original))
        if existing:
            return existing
        if canonical_type in marker_set:
            out, index = _make_marker(canonical_type)
            entries.append({"out": out, "original": original,
                            "type": canonical_type, "kind": "marker", "index": index})
        else:
            out = generate_surrogate(canonical_type, original, session_id, text, occupied)
            entries.append({"out": out, "original": original,
                            "type": canonical_type, "kind": "surrogate"})
        occupied.add(out)
        reverse[(canonical_type, original)] = out
        return out

    def _note_type(ctype: str) -> None:
        if ctype not in entity_types_found:
            entity_types_found.append(ctype)

    # --- Шаг 1: Natasha NER (Person + Org для русского) ---
    natasha_spans: list[tuple[int, int, str, str]] = []  # (start, end, canonical, out)

    if language == "ru":
        doc = Doc(text)
        doc.segment(_segmenter)
        doc.tag_ner(_ner_tagger)

        for span in doc.spans:
            if span.type in ("PER", "ORG"):
                s, e = _clip_natasha_span(text, span.start, span.stop)
                if e <= s:
                    continue
                span_text = text[s:e]
                if _is_natasha_fp(span_text):
                    continue
                ctype = _CANONICAL_TYPE.get(span.type, span.type)
                out = _allocate(ctype, span_text)
                natasha_spans.append((s, e, ctype, out))
                _note_type(ctype)

    # --- Шаг 2: Presidio ---
    results = _presidio_analyzer.analyze(
        text=text,
        language=language,
        entities=_presidio_entities(language),
    )
    results = _resolve_presidio_overlaps(results)

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
        ctype = _CANONICAL_TYPE.get(result.entity_type, result.entity_type)
        out = _allocate(ctype, original)
        presidio_spans.append((result.start, result.end, ctype, out))
        _note_type(ctype)

    # --- Шаг 3: Сборка слева-направо + masked_spans (позиции в итоговом тексте) ---
    all_spans = sorted(natasha_spans + presidio_spans, key=lambda x: x[0])

    parts: list[str] = []
    masked_spans: list[dict] = []
    cursor = 0
    new_pos = 0
    for start, end, ctype, out in all_spans:
        if start < cursor:
            continue  # защита от перекрытий (не должно случаться после резолва)
        segment = text[cursor:start]
        parts.append(segment)
        new_pos += len(segment)
        masked_spans.append({"start": new_pos, "end": new_pos + len(out), "type": ctype})
        parts.append(out)
        new_pos += len(out)
        cursor = end
    parts.append(text[cursor:])
    masked = "".join(parts)

    return masked, entries, entity_types_found, masked_spans


# ----------------------------------------------------------
# Функция демаскирования
# ----------------------------------------------------------

def entries_to_out_map(entries: list[dict]) -> dict[str, str]:
    """Строит словарь {out: original} из записей mapping."""
    return {e["out"]: e["original"] for e in entries}


def unmask_text(text: str, out_to_original: dict[str, str]) -> tuple[str, int]:
    """
    Восстанавливает оригинальные значения из суррогатов/меток.

    Однопроходный re.sub: исключает каскад (восстановленный original не
    пересматривается) и даёт точный счётчик вхождений.

    Returns:
        unmasked_text   — восстановленный текст
        tokens_replaced — количество фактических вхождений (не число ключей)
    """
    if not out_to_original:
        return text, 0

    # Альтернатива отсортирована longest-first: длинный out матчится раньше,
    # чтобы короткий out-префикс не «съел» начало длинного.
    pattern = re.compile(
        "|".join(re.escape(o) for o in sorted(out_to_original, key=len, reverse=True))
    )
    count = 0

    def _repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        return out_to_original[m.group(0)]

    result = pattern.sub(_repl, text)
    return result, count
