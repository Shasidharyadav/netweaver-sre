#!/bin/bash
APP_PORT=${PORT:-7860}
if [ "$APP_PORT" != "8000" ]; then
    socat TCP-LISTEN:8000,fork,reuseaddr TCP:localhost:${APP_PORT} &
fi
exec uvicorn server.app:app --host 0.0.0.0 --port ${APP_PORT}
