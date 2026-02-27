# tests/helpers.py
from presidio_analyzer import RecognizerResult


class FakeSpan:
    """Mimics natasha DocSpan."""
    def __init__(self, start: int, stop: int, type_: str, text: str):
        self.start = start
        self.stop = stop
        self.type = type_
        self.text = text


class FakeNERTagger:
    """Callable stub — sets predefined spans on the doc when called."""
    def __init__(self, spans=None):
        self.spans = spans or []

    def __call__(self, doc):
        doc.spans = self.spans


class FakeDoc:
    """Replaces natasha.Doc — segment() is a no-op, tag_ner() calls the tagger."""
    def __init__(self, text: str):
        self.text = text
        self.spans = []

    def segment(self, segmenter):
        pass

    def tag_ner(self, tagger):
        tagger(self)


class FakePresidioAnalyzer:
    """Stub for AnalyzerEngine — returns a fixed list of RecognizerResult."""
    def __init__(self, results=None):
        self.results = results or []
        self.last_entities_requested: list[str] = []

    def analyze(self, text, language, entities):
        self.last_entities_requested = entities
        return self.results
