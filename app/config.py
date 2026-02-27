# ============================================================
# config.py — конфигурация через переменные окружения
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Логирование
    log_level: str = "INFO"

    # Параметры маскирования
    # Префиксы токенов (можно переопределить)
    token_person: str = "PERSON"
    token_phone: str = "PHONE"
    token_inn: str = "INN"
    token_ogrn: str = "OGRN"
    token_org: str = "ORG"
    token_email: str = "EMAIL"
    token_passport: str = "PASSPORT"
    token_snils: str = "SNILS"
    token_address: str = "ADDRESS"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
