set -a
source .env
set +a

cd ml_service
uvicorn main:app --host 0.0.0.0 --port ${APP_PORT} --reload