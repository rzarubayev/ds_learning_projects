cd $(dirname "$0")
echo Переход в папку $(dirname "$0")
APP_PORT=$(grep APP_PORT .env | cut -d "=" -f2)
export APP_NAME=$(grep APP_NAME .env | cut -d "=" -f2)
export FLATS_PRICE_MODEL=$(grep FLATS_PRICE_MODEL .env | cut -d "=" -f2)
export FLATS_PRICE_RESULT_COUNT=$(grep FLATS_PRICE_RESULT_COUNT .env | cut -d "=" -f2)
export FLATS_PRICE_RESULT_BUCKETS=$(grep FLATS_PRICE_RESULT_BUCKETS .env | cut -d "=" -f2)
export MODEL_DIR="../models"
cd ml_service
uvicorn main:app --host 0.0.0.0 --port ${APP_PORT} --reload