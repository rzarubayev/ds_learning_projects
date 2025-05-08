import os
import logging

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("../.env")

RANDOM_STATE = 42

# Общие настройки
PORT = os.getenv("APP_PORT")
SERVICE_URL = f"http://localhost:{PORT}/"

PREFIX = "/bank_products"
LOAD_URI = "/load_model"
USERS_URI = "/get_product_clients"
ITEMS_URI = "/get_client_products"
METRICS_URI = "/get_metrics"

TIMEOUT = 10
MODEL_TIMEOUT = 30

ITEMS_K = 7
USERS_K = 2000
USER_RAND =0.1
THOLDS = (0.5, 0.8)
TOP_K = (3, 7)

def join_url(*parts):
    return "/".join(part.strip("/") for part in parts)


# Настройка логирования
logging.basicConfig(
    filename="test_service.log", # Файл логов
    filemode="w", # Перезапись при каждом запуске
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

logging.info(f"Фиксация random seed: {RANDOM_STATE}")
np.random.seed(RANDOM_STATE)

# Получение датасета
logging.info(f"Загрузка тестового датасета")
y_true = pd.read_parquet("test_data.parquet")
print("Датасет загружен!")

# Получение ТОП продуктов по популярности
logging.info(f"Загрузка ТОП-{ITEMS_K} продуктов")
pop_items = y_true.sum().sort_values(ascending=False) \
    .head(ITEMS_K).index.to_numpy()
print(f"ТОП-{ITEMS_K} продуктов:\n", pop_items)

logging.info("Получение дополнительных данных")

# Получение списка пользователей
user_ids = y_true.index.to_numpy()

# Добавление рандомных пользователей 
# для получения случайных ошибок 
user_ids = np.unique(np.concatenate([
    user_ids, 
    np.random.randint(
        y_true.index.min(), 
        y_true.index.max(),
        int(len(user_ids) * USER_RAND))
]))

# Проверка доступности сервиса -----------------------
while True:
    print()
    inf = "Проверка доступности сервиса"
    print(inf)
    logging.info(inf)
    try: 
        resp = requests.get(
            join_url(SERVICE_URL, PREFIX), 
            timeout=TIMEOUT)
        if resp.status_code == 200:
            print(resp.json())
            logging.info("Сервис доступен")
            break
        else:
            inf = (f"Ошибка {resp.status_code}, ответ "
                   f"сервера {resp.json()}")
            print(inf)
            logging.error(inf)
    except requests.exceptions.RequestException as e:
        inf = f"Ошибка : {e}"
        logging.error(inf)
        print(inf)
    print("Произошла ошибка, попробуйте снова...")
    while True:
        print()
        print("1 - Повторить проверку")
        print("2 - Выйти")
        choice = input("Выберите действие: ")
        if choice == "1":
            break
        elif choice == "2":
            exit(0)
        print("Неверный ввод")

# Проведение тестирования ----------------------------

for item in pop_items:
    # Получение пользователей для ТОП продуктов ------
    thold = np.round(np.random.uniform(*THOLDS), 2)
    inf = (f"Запрос списка клиентов по продукту {item}, "
           f"граница для включения - {thold:.2f}")
    logging.info(inf)
    print()
    print(inf)
    params = dict(
        product_name=item,
        threshold=thold)
    try:
        resp = requests.post(
            join_url(SERVICE_URL, PREFIX, USERS_URI),
            timeout=TIMEOUT, params=params)
        if resp.status_code == 200:
            users = resp.json()
            inf = f"Получены {len(users)} клиентов"
            logging.info(inf)
            print(inf)
        else:
            inf = (f"Ошибка {resp.status_code}, ответ "
                   f"сервера {resp.json()}")
            print(inf)
            logging.error(inf)
    except requests.exceptions.RequestException as e:
        inf = f"Ошибка : {e}"
        print(inf)
        logging.error(inf)

    # Получение продуктов для рандомных пользователей-

    inf = "Запрос рекомендаций продуктов для клиента"
    print(inf)
    logging.info(inf)

    items_ok = 0
    items_bad = 0

    # Получение случайных пользователей
    users = np.random.choice(
        user_ids, size=USERS_K, replace=False)

    for user_id in tqdm(users, desc="Запрос по клиентам:"):
        params = dict(
            user_id=int(user_id),
            top_k=np.random.randint(*TOP_K))
        try:
            resp = requests.post(
                join_url(SERVICE_URL, PREFIX, ITEMS_URI),
                timeout=TIMEOUT, params=params)
            if resp.status_code == 200:
                items_ok += 1
            else:
                items_bad += 1
        except requests.exceptions.RequestException as e:
            inf = f"Ошибка запроса: {e}"
            print(inf)
            logging.error(inf)
            break
    
    inf = (
        f"Из {USERS_K} запросов получено {items_ok} "
        f"успешных ответа, {items_bad} с ошибкой")
    print(inf)
    logging.info(inf)

    # Получение метрик по рандомным клиентам ---------

    users = np.random.choice(
        [True, False], size=len(y_true))
    
    inf = f"Запрос метрик по случайным {users.sum()} клиентам"
    print(inf)
    logging.info(inf)

    # Получение данных вида index: [val1, ..., valn]
    user_items = y_true[users].astype(int).to_dict(orient="index")
    user_items = {k: list(v.values()) 
                  for k, v in user_items.items()}
    try:
        resp = requests.put(
            join_url(SERVICE_URL, PREFIX, METRICS_URI),
            timeout=MODEL_TIMEOUT, json=user_items)
        if resp.status_code == 200:
            inf = f"Метрики получены: {resp.json()}"
            print(inf)
            logging.info(inf)
        else:
            inf = f"Ошибка сервера {resp.status_code}: {resp.json()}"
            print(inf)
            logging.info(inf)
    except requests.exceptions.RequestException as e:
        inf = f"Ошибка запроса: {e}"
        print(inf)
        logging.error(inf)

    # Проверка перезагрузки модели -------------------

    inf = "Перезагрузка модели"
    print(inf)
    logging.info(inf)
    # Передача пустого json для перезагрузки текущей модели
    try:
        resp = requests.post(
            join_url(SERVICE_URL, PREFIX, LOAD_URI),
            timeout=MODEL_TIMEOUT, json={})
        if resp.status_code == 200:
            inf = f"Модель загружена: {resp.json()}"
            print(inf)
            logging.info(inf)
        else:
            inf = f"Ошибка сервера {resp.status_code}: {resp.json()}"
            print(inf)
            logging.info(inf)
    except requests.exceptions.RequestException as e:
        inf = f"Ошибка запроса: {e}"
        print(inf)
        logging.error(inf)

# Запрос метрик по всему датасету --------------------

inf = "Запрос метрик по всему датасету"
print()
print(inf)
logging.info(inf)

user_items = y_true.astype(int).to_dict(orient="index")
user_items = {k: list(v.values()) 
                for k, v in user_items.items()}
try:
    resp = requests.put(
        join_url(SERVICE_URL, PREFIX, METRICS_URI),
        timeout=MODEL_TIMEOUT, json=user_items)
    if resp.status_code == 200:
        inf = f"Метрики получены: {resp.json()}"
        print(inf)
        logging.info(inf)
    else:
        inf = f"Ошибка сервера {resp.status_code}: {resp.json()}"
        print(inf)
        logging.info(inf)
except requests.exceptions.RequestException as e:
    inf = f"Ошибка запроса: {e}"
    print(inf)
    logging.error(inf)

inf = "Тестирование завершено"
print()
print(inf)
logging.info(inf)
