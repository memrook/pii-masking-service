# tests/conftest.py
import pytest
from app import masker
from tests.helpers import FakeDoc, FakeNERTagger, FakePresidioAnalyzer


@pytest.fixture
def setup_stubs(monkeypatch):
    """Patches masker globals so mask_text can run without real NLP models."""
    monkeypatch.setattr(masker, "_models_loaded", True)
    monkeypatch.setattr(masker, "_segmenter", object())   # non-None sentinel
    monkeypatch.setattr(masker, "Doc", FakeDoc)
    monkeypatch.setattr(masker, "_ner_tagger", FakeNERTagger())
    monkeypatch.setattr(masker, "_presidio_analyzer", FakePresidioAnalyzer())
