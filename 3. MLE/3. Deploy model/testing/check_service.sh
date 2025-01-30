cd $(dirname "$0")
ENV_FILE="../services/.env"
APP_PORT=$(grep APP_PORT $ENV_FILE | cut -d "=" -f2)
FLATS_PRICE_MODEL=$(grep FLATS_PRICE_MODEL $ENV_FILE | cut -d "=" -f2)
FLATS_PRICE_INPUT=$(grep FLATS_PRICE_INPUT $ENV_FILE | cut -d "=" -f2)
clear
echo 1. Проверка работы сервиса по адресу http://localhost:$APP_PORT/
echo Результат запроса:

curl -i http://localhost:$APP_PORT/

echo
echo "[Нажмите любую клавишу]"
read -rsn1
clear

echo 2. Проверка статуса загрузки модели предсказания стоимости квартир
echo GET запросом адресу http://localhost:$APP_PORT/flats_price
echo Результат запроса:

curl -i http://localhost:$APP_PORT/flats_price

echo
echo "[Нажмите любую клавишу]"
read -rsn1
clear

echo 3. Перезагрузка модели производится GET запросом по адресу
echo http://localhost:$APP_PORT/flats_price/reload
echo 

echo Можно указать другую директорию с моделями, если не передавать 
echo параметры - используется последняя корректно загруженная модель. 
echo При первом запуске загружаются модель из следующей директории:
echo model_dir = $FLATS_PRICE_MODEL
echo
echo Результат запроса для модели в папке flats_price_new:

curl -i --get "http://localhost:$APP_PORT/flats_price/reload" \
    -d "model_dir=flats_price_new"
    

echo
echo "-----------------------------------------------------------------"
echo
echo "Результат запроса без параметров (загрузка предыдущих файлов)":

curl -i "http://localhost:$APP_PORT/flats_price/reload"

echo
echo