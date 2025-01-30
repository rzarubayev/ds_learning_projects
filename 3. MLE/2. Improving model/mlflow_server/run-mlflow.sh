#!/bin/sh
cd "$(dirname "$0")"
cd ..
if [ -f ".env" ]; then
    while IFS='=' read -r key value; do
        if [ "${key#\#}" = "$key" ] && [ -n "$value" ]; then
            export "$key"="$value"
        fi
    done < .env
    echo "Переменные окружения добавлены"
    echo "Запуск MLflow"
    mlflow server -h $MLFLOW_HOST -p $MLFLOW_PORT         --backend-store-uri postgresql://$PG_USER:$PG_PASS@$PG_HOST:$PG_PORT/$PG_DB         --default-artifact-root s3://$S3_BUCKET         --no-serve-artifacts
else
    echo ".env файл не найден!"
fi 