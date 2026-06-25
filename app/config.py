# ============================================================
# config.py — конфигурация через переменные окружения
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict


# ----------------------------------------------------------
# Канонические типы сущностей
# ----------------------------------------------------------
# Детекторы (natasha / Presidio) выдают разные entity_type для одной сущности
# (PER/PERSON, PHONE_RU/PHONE_NUMBER, CARD/CREDIT_CARD). Каноника сводит их к
# единому типу, на котором работают routing (surrogate/marker), сид суррогата,
# mapping и генераторы. См. _CANONICAL_TYPE в masker.py.
ALL_CANONICAL_TYPES = frozenset({
    "PERSON", "PHONE", "INN", "OGRN", "ORG", "EMAIL", "PASSPORT",
    "SNILS", "ADDRESS", "CARD", "VIN", "ACCOUNT", "BIK", "KPP",
})


class Settings(BaseSettings):
    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # Маппинг TTL (секунды)
    mapping_ttl: int = 300           # 5 минут по умолчанию

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 6111
    api_workers: int = 2
    api_key: str = ""                # если пусто — авторизация отключена
    api_root_path: str = ""          # subpath при размещении за reverse-proxy, напр. /ai/masking

    # Логирование
    log_level: str = "INFO"

    # --------------------------------------------------------
    # Маскирование — гибрид: суррогаты (цифры/коды) + метки (имена/орг/адреса)
    # --------------------------------------------------------
    # Секрет для HMAC-сида суррогатов. ОБЯЗАТЕЛЕН: при пустом значении сервис
    # падает на старте (см. validate_config). Случайный секрет недопустим —
    # сервис запускается с несколькими worker-процессами, и у каждого был бы
    # свой секрет, из-за чего один PII дал бы разные суррогаты → round-trip
    # сломался бы. Генерация: openssl rand -hex 32
    surrogate_secret: str = ""

    # Канонические типы, которые маскируются РЕАЛИСТИЧНЫМИ СУРРОГАТАМИ
    # (синтаксически правдоподобные тестовые значения; LLM воспроизводит дословно).
    surrogate_types: list[str] = [
        "PHONE", "INN", "OGRN", "EMAIL", "PASSPORT",
        "SNILS", "CARD", "VIN", "ACCOUNT", "BIK", "KPP",
    ]
    # Канонические типы, которые маскируются ИНЕРТНЫМИ МЕТКАМИ вида "[Имя 1]"
    # (склоняемые/текстовые — суррогат сломал бы точный round-trip из-за падежей).
    marker_types: list[str] = ["PERSON", "ORG", "ADDRESS"]

    # Скобки метки
    marker_open: str = "["
    marker_close: str = "]"

    # Подписи меток по языкам: canonical_type -> подпись.
    # Для en реально применяется только PERSON (см. оговорку про английский
    # pipeline в masker._presidio_entities); ORG/ADDRESS заданы на будущее.
    marker_labels_ru: dict[str, str] = {
        "PERSON": "Имя",
        "ORG": "Организация",
        "ADDRESS": "Адрес",
    }
    marker_labels_en: dict[str, str] = {
        "PERSON": "Name",
        "ORG": "Organization",
        "ADDRESS": "Address",
    }

    # Подсказка для LLM про метки. Возвращается отдельным полем `hint`
    # (НЕ вшивается в masked_text — иначе ломается прямой round-trip).
    # Внутри подсказки используется обобщённое "[Имя N]" / "[Name N]".
    hint_text_ru: str = (
        "Примечание: фрагменты в квадратных скобках вида [Имя N] — это "
        "обезличенные реальные данные. Сохраняйте их в ответе дословно, "
        "без изменений."
    )
    hint_text_en: str = (
        "Note: bracketed fragments like [Name N] are de-identified real data. "
        "Keep them verbatim in your response, without changes."
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --------------------------------------------------------
    # Производные / валидация
    # --------------------------------------------------------
    def marker_labels(self, language: str) -> dict[str, str]:
        return self.marker_labels_en if language == "en" else self.marker_labels_ru


def validate_config(s: "Settings") -> None:
    """Fail-fast валидация конфигурации. Вызывается при старте приложения.

    Бросает RuntimeError с понятным сообщением при любой проблеме.
    """
    if not s.surrogate_secret:
        raise RuntimeError(
            "SURROGATE_SECRET не задан. Это обязательный секрет для генерации "
            "суррогатов (один на все воркеры). Сгенерируйте: openssl rand -hex 32"
        )

    surr = set(s.surrogate_types)
    mark = set(s.marker_types)

    overlap = surr & mark
    if overlap:
        raise RuntimeError(
            f"surrogate_types и marker_types пересекаются: {sorted(overlap)}"
        )

    covered = surr | mark
    unknown = covered - ALL_CANONICAL_TYPES
    if unknown:
        raise RuntimeError(
            f"Неизвестные канонические типы в конфиге: {sorted(unknown)}. "
            f"Допустимые: {sorted(ALL_CANONICAL_TYPES)}"
        )

    missing = ALL_CANONICAL_TYPES - covered
    if missing:
        raise RuntimeError(
            f"Канонические типы не отнесены ни к surrogate, ни к marker: "
            f"{sorted(missing)}"
        )

    for ctype in mark:
        if ctype not in s.marker_labels_ru:
            raise RuntimeError(f"Нет ru-подписи метки для типа {ctype}")
        if ctype not in s.marker_labels_en:
            raise RuntimeError(f"Нет en-подписи метки для типа {ctype}")


settings = Settings()
