# tests/test_config.py — fail-fast валидация конфигурации
import pytest

from app.config import Settings, validate_config


def _ok_settings(**over) -> Settings:
    s = Settings(surrogate_secret="x")
    for k, v in over.items():
        setattr(s, k, v)
    return s


def test_valid_config_passes():
    validate_config(_ok_settings())  # не бросает


def test_empty_secret_fails():
    with pytest.raises(RuntimeError, match="SURROGATE_SECRET"):
        validate_config(_ok_settings(surrogate_secret=""))


def test_overlapping_groups_fail():
    s = _ok_settings()
    s.surrogate_types = list(s.surrogate_types) + ["PERSON"]  # PERSON и в marker
    with pytest.raises(RuntimeError, match="пересекаются"):
        validate_config(s)


def test_unknown_type_fails():
    s = _ok_settings()
    s.surrogate_types = list(s.surrogate_types) + ["FOOBAR"]
    with pytest.raises(RuntimeError, match="Неизвестные"):
        validate_config(s)


def test_uncovered_type_fails():
    s = _ok_settings()
    s.surrogate_types = ["PHONE"]   # остальные типы не отнесены ни к одной группе
    with pytest.raises(RuntimeError, match="не отнесены"):
        validate_config(s)


def test_missing_marker_label_fails():
    s = _ok_settings()
    # переносим PHONE в маркеры, но без подписи → должна быть ошибка про подпись
    s.surrogate_types = [t for t in s.surrogate_types if t != "PHONE"]
    s.marker_types = list(s.marker_types) + ["PHONE"]
    with pytest.raises(RuntimeError, match="подпис"):
        validate_config(s)


def test_marker_labels_localized():
    s = _ok_settings()
    assert s.marker_labels("ru")["PERSON"] == "Имя"
    assert s.marker_labels("en")["PERSON"] == "Name"
