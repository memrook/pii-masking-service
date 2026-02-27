# ============================================================
# Dockerfile — PII Masking Service
# ============================================================

FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------------------------------------------------
# Зависимости
# ----------------------------------------------------------
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем spaCy модели для NER
RUN python -m spacy download ru_core_news_lg && \
    python -m spacy download en_core_web_lg

# Прогреваем Natasha-эмбеддинги (~220MB при первом вызове)
RUN python -c "from natasha import NewsEmbedding, NewsNERTagger; NewsNERTagger(NewsEmbedding())"

# ----------------------------------------------------------
# Финальный образ
# ----------------------------------------------------------
FROM deps AS final

COPY app/ ./app/

RUN useradd -m -u 1000 piiuser && chown -R piiuser:piiuser /app
USER piiuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Healthcheck — обращаемся к /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:6111/health || exit 1

EXPOSE 6111

CMD ["python", "-m", "app.main"]
