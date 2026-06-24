#!/usr/bin/env bash
# ============================================================
# deploy.sh — локальный запуск PII Masking Service
#
# Сценарий:
#   1. Проверка .env (создаёт из .env.example если нет)
#   2. Прогон тестов (pytest)
#   3. docker compose up -d --build
#   4. Ожидание healthy
#   5. При фейле — показывает логи контейнера
#
# Использует docker-compose.yml (build из исходников).
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

COMPOSE_FILE="docker-compose.yml"
SERVICE="pii-masking"
HEALTH_TIMEOUT=120  # секунд на старт моделей

log() { echo "[deploy] $*"; }
fail() { echo "[deploy] ❌ $*" >&2; exit 1; }

# 1. .env
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        log "Создан .env из .env.example"
        log "⚠️  Отредактируй .env (особенно REDIS_PASSWORD) и запусти снова"
        exit 1
    else
        fail ".env и .env.example не найдены"
    fi
fi

# 2. Тесты
log "Прогон тестов..."
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

if ! command -v pytest >/dev/null 2>&1; then
    fail "pytest не найден. Активируй venv или установи зависимости: pip install -r requirements.txt"
fi

pytest -q tests/ || fail "Тесты упали"
log "✅ Тесты пройдены"

# 3. Build + up
log "docker compose up -d --build..."
docker compose -f "$COMPOSE_FILE" up -d --build

# 4. Health wait
log "Ожидание healthy (до ${HEALTH_TIMEOUT}s)..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$SERVICE" 2>/dev/null || echo "missing")
    case "$status" in
        healthy)
            log "✅ $SERVICE healthy"
            log "UI:      http://localhost:6111/"
            log "Swagger: http://localhost:6111/docs"
            log "Health:  http://localhost:6111/api/health"
            exit 0
            ;;
        unhealthy)
            log "❌ Контейнер unhealthy. Логи:"
            docker compose -f "$COMPOSE_FILE" logs --tail=80 "$SERVICE"
            exit 1
            ;;
    esac
    sleep 2
done

log "❌ Таймаут healthcheck (${HEALTH_TIMEOUT}s). Логи:"
docker compose -f "$COMPOSE_FILE" logs --tail=80 "$SERVICE"
exit 1
