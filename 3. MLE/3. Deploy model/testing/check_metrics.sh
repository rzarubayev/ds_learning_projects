cd $(dirname "$0")
ENV_FILE="../services/.env"
APP_PORT=$(grep APP_PORT $ENV_FILE | cut -d "=" -f2)
echo Метрики доступны GET запросом на http://localhost:$APP_PORT/metrics
echo Результат запроса:
echo

curl http://localhost:$APP_PORT/metrics

echo