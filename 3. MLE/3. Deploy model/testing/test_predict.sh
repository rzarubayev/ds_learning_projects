cd $(dirname "$0")
ENV_FILE="../services/.env"
APP_PORT=$(grep APP_PORT $ENV_FILE | cut -d "=" -f2)
USER_ID=$(uuidgen)

echo POST запрос к микросервису для предсказания цены квартиры, в качестве параметров 
echo запроса случайная запись из тестовой выборки, сохраненная в JSON-файл. 
echo В качестве user_id - сгенерированный случайно UUID: $USER_ID.

curl -i -X POST "http://localhost:$APP_PORT/flats_price/predict?user_id=$USER_ID" \
    -H "Content-Type: application/json" \
    -d @test.json

echo
echo