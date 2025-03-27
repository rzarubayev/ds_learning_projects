import os
import logging

import joblib
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

pd.options.mode.chained_assignment = 'raise'

# Настройка логирования
logging.basicConfig(
    filename="test_service.log", # Файл логов
    filemode="w", # Перезапись при каждом запуске
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Настройки сервиса
load_dotenv()
EVENTS_STORE_URL = os.getenv("EVENTS_STORE_URL")
RECSYS_URL = os.getenv("RECSYS_URL")
SPLIT_DATE = pd.to_datetime(os.getenv("SPLIT_DATE"))
RECS_COUNT = int(os.getenv("RECS_COUNT"))

# Путь к файлам с рекомендациями
S3_PATH = f"s3://{os.getenv('AWS_S3_BUCKET')}/recsys/{os.getenv('TEST_DATA')}"

# Настройки хранилища для загрузки файлов
STORAGE_OPTIONS = {
    "endpoint_url": os.getenv("AWS_ENDPOINT_URL"), 
    "key": os.getenv("AWS_ACCESS_KEY_ID"), 
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs":{
        "region_name": os.getenv("AWS_REGION")},
    "config_kwargs": {
        "signature_version": os.getenv("AWS_SIGV")}
}

# Загрузка списка горячих пользователей
with open("models/id_encoders.pkl", "rb") as fd:
    encoders = joblib.load(fd)
    hot_users = encoders["user"].classes_
    warm_users = encoders["warm"].classes_

class TestService:

    def __init__(
            self, 
            event_store_url: str, 
            recsys_url: str, 
            recs_count: int):
        self.options = {
            "1": self.settings,
            "2": self.hot_user_recs,
            "3": self.warm_user_recs,
            "4": self.new_user_recs,
            "5": self.all_users_recs,
            "6": self.exit
        }
        self.event_store_url = event_store_url
        self.recsys_url = recsys_url
        self.recs_count = recs_count
        self.data = None
        self.running = True

    def load_data(self, path, split_date, hot_users, warm_users, **kwargs):
        """
        Загрузка датасета с тестовыми событиями
        """
        
        logging.info("Loading dataset")
        print("Загрузка датасета")
        
        # Получение тестового датасета
        self.data = pd.read_parquet(
            path, filters=[("started_at", ">=", split_date)], **kwargs)
        
        # Добавления признака горячего пользователя
        self.data = self.data.merge(
            pd.DataFrame({"user_id": hot_users, "hot": True}),
            how="left", on="user_id"
        )
        self.data["hot"] = self.data["hot"].fillna(False)
        
        # Добавление признака теплого пользователя
        self.data = self.data.merge(
            pd.DataFrame({"user_id": warm_users, "warm": True}),
            how="left", on="user_id"
        )
        self.data["warm"] = self.data["warm"].fillna(False)
        
        # Ранжирование треков
        target_ids = self.data.groupby("user_id")["item_seq"].rank(
            method="dense", ascending=True)
        
        # Получение списка событий для исключения
        target_ids = target_ids <= np.ceil(
            self.data.groupby("user_id")["item_id"].transform("count") / 2)
        target_ids = target_ids & self.data["hot"]

        # Исключение событий и сортировка по дате
        self.data = self.data[~target_ids].copy()
        self.data = self.data.sort_values(by=["started_at", "item_seq"])

        # Добавление признака о тестировании
        self.data.loc[:, "tested"] = False
        self.data = self.data.reset_index(drop=True).copy()
        
        logging.info(f"Loaded {len(self.data)} test events")
        print()
        print("Датасет загружен")
        print("Всего событий:", len(self.data))
        print("Количество пользователей:", self.data["user_id"].nunique())
        print("Количество горячих пользователей:",
              self.data[self.data["hot"]]["user_id"].nunique())
        print("Количество теплых пользователей:",
              self.data[self.data["warm"]]["user_id"].nunique())
        print("Количество новых пользователей:",
              self.data[
                  (~self.data["hot"]) & (~self.data["warm"])
              ]["user_id"].nunique())
        print()
        
    def show_menu(self):
        print()
        print("Меню для тестирования микросервиса:")
        print("1 - Изменить настройки")
        print("2 - Рекомендации для горячего пользователя (один пользователь)")
        print("3 - Рекомендации для теплого пользователя (один пользователь)")
        print("4 - Рекомендации для нового пользователя (один пользователь)")
        print("5 - Рекомендации для всех пользователей (введите количество)")
        print("6 - Завершить")

    def check_services(self) -> bool:
        """
        Проверка доступности сервисов
        """
        result = True
        for url in [self.recsys_url, 
                    self.event_store_url]:
            try:
                logging.info(f"Checking '{url}")
                response = requests.get(url, timeout=5) 
                if response.status_code == 200:
                    print(f"{url} доступен")
                    logging.info(f"{url} is up")
                else:
                    print(f"{url} недоступен (код {response.status_code})")
                    logging.warning(f"{url} status code {response.status_code}")
                    result = False
            except requests.exceptions.RequestException as e:
                print(f"{url} недоступен. Ошибка: {e}")
                logging.warning(f"Error: {e}")
                result = False
        return result

    def settings(self):
        """
        Изменение настроек тестового скрипта
        """
        logging.info("Script settings")
        while True:
            print()
            print("Изменить настройки скрипта:")
            print("1 - URL сервиса рекомендаций")
            print("2 - URL хранилища событий")
            print(f"3 - Количество рекоменаций ({self.recs_count})")
            print("4 - Проверить и выйти")
            choice = input("Выберите действие: ")
            if choice == "1":
                self.recsys_url = input(
                    "Введите URL сервиса рекомендаций: ")
                logging.info(f"Recsys URL changed: '{self.recsys_url}'")
            elif choice == "2":
                self.event_store_url = input(
                    "Введите URL хранилища событий: ")
                logging.info(f"Event Store URL changed: '{self.event_store_url}'")
            elif choice == "3":
                while True:
                    try:
                        self.recs_count = int(input("Введите целое число больше нуля: "))
                        if self.recs_count > 0: 
                            logging.info(
                                f"Recommendations count changed: {self.recs_count}")
                            break
                    except:
                        print("Ошибка ввода, повторите")
            elif choice == "4":
                if self.check_services():
                    logging.info("Back to main menu")
                    break
            else:
                print("Некорректный ввод, попробуйте снова.")

    def test(self, user_id, item_id) -> bool:
        """
        Тестирование сервиса
        """
        result = True
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        event_resp = requests.post(
            self.event_store_url + "/put",
            headers=headers, timeout=10,
            params={"user_id": user_id, "item_id": item_id})
        if event_resp.status_code == 200:
            print(f"Трек {item_id} пользователя {user_id} добавлен в хранилище событий")
            logging.info(f"user {user_id} with {item_id} stored in event store")
            recs_resp = requests.get(
                self.recsys_url + f"/{user_id}",
                headers=headers, timeout=10,
                params={"k": self.recs_count}
            )
            if recs_resp.status_code == 200:
                recs = recs_resp.json()
                recs = recs["recs"]
                print(f"Для пользователя {user_id} получены рекомендации: {recs}")
                logging.info(f"User {user_id} recommendations: {recs}")
            else:
                result = False
                print("Рекомендации не получены, код:", 
                      recs_resp.status_code)
                logging.warning(
                    f"Recsys status code: {recs_resp.status_code}")
        else:
            result = False
            print("Не удалось добавить событие в хранилище, код:", 
                  event_resp.status_code)
            logging.warning(
                f"Event store status code: {event_resp.status_code}")

        return result
    
    def one_user_recs(self, hot: bool = False, warm: bool = False):
        """
        Тестирование одного пользователя
        """
        new = False
        if hot and warm:
            logging.warning("Both 'hot' and 'warm' are True. Processing 'hot' user.")
        if hot:
            event = self.data[self.data["hot"] & (~self.data["tested"])].index[:1]
        elif warm:
            event = self.data[self.data["warm"] & (~self.data["tested"])].index[:1]
        else:
            event = self.data[
                (~self.data["hot"]) & 
                (~self.data["warm"]) &
                (~self.data["tested"])
            ].index[:1]
            new = True
        if len(event) > 0:
            event = event[0]
            user_id, item_id = tuple(self.data.loc[event][["user_id", "item_id"]])
            self.data.loc[event, "tested"] = self.test(user_id, item_id)
            if new:
                self.data.loc[self.data["user_id"] == user_id, "warm"] = True
            print(f"Обработано событий: {self.data['tested'].sum()} " +
                  f"({self.data['tested'].mean():.2%})")
        else:
            logging.info("There are no events left")
            print("Событий не осталось")

    def hot_user_recs(self):
        """
        Обработка события горячего пользователя
        """
        logging.info("Hot user recommendations selected")
        self.one_user_recs(hot=True)

    def warm_user_recs(self):
        """
        Обработка события теплого пользователя
        """
        logging.info("Warm user recommendations selected")
        self.one_user_recs(warm=True)

    def new_user_recs(self):
        """
        Обработка события нового пользователя
        """
        logging.info("New user recommendations selected")
        self.one_user_recs()

    def all_users_recs(self):
        """
        Обработка всех событий
        """
        logging.info("All users recommendations selected")
        print()
        # Ограничение количества событий
        while True:
            try:
                max_events = int(input("Введите количество событий для тестирования: "))
                break
            except:
                print("Введено некорректное число, попробуйте снова.")
        # Получение тестового датасета
        test_df = self.data[~self.data["tested"]].iloc[:max_events][["user_id", "item_id"]]
        print(f"Обработка {len(test_df)} событий")
        logging.info(f"Will be tested {len(test_df)} events")
        
        # Тестирование в несколько потоков
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            logging.info(f"Process {len(test_df)} events with {executor._max_workers} workers")
            tested = list(executor.map(self.test, test_df["user_id"], test_df["item_id"]))
        logging.info(f"Processed {len(tested)} recommendations")
        
        self.data.loc[test_df.index, "tested"] = tested
        print()
        print(f"Обработано событий: {self.data['tested'].sum()} " +
              f"({self.data['tested'].mean():.2%})")
        
    def exit(self):
        """
        Завершение работы скрипта
        """
        logging.info("Stop service selected")
        print("Выход")
        self.running = False

    def run(self):
        if self.check_services():
            print("Количество рекомендаций на пользователя:",
                  self.recs_count)
        else:
            print("Проверка микросервисов не пройдена, проверьте настройки")
            self.settings()
        
        while self.running:
            self.show_menu()
            choice = input("Выберите действие: ")
            action = self.options.get(choice)
            if action:
                action()
            else:
                print("Некорректный ввод, попробуйте снова.")


logging.info("Starting")
test_service = TestService(EVENTS_STORE_URL, RECSYS_URL, RECS_COUNT)
# Загрузка тестового датасета
test_service.load_data(
    S3_PATH, SPLIT_DATE, 
    hot_users, warm_users,
    engine="pyarrow",
    storage_options=STORAGE_OPTIONS)
# Запуск сервиса тестирования
test_service.run()