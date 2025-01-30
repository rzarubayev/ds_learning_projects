cd $(dirname "$0")
ENV=".env"
APP_PORT=$(grep APP_PORT $ENV | cut -d "=" -f2)
MODEL_DIR=$(grep MODEL_DIR $ENV | cut -d "=" -f2)
docker build . -f Dockerfile_ml_service -t ml_service:latest
docker run -d -rm \
    -p $APP_PORT:$APP_PORT \
    --env-file .env \
    -v ./models:$MODEL_DIR \
    --name ml_service \
    ml_service:latest
docker image prune -f