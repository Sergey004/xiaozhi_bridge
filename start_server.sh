#!/bin/bash
# Запуск xiaozhi_server (HTTP :8000 + WebSocket :8765).
# Конфиг — .env рядом со скриптом (см. env.example).
set -eu
cd "$(dirname "$0")"
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi
exec python xiaozhi_server.py "$@"
