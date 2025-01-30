import os
import asyncio
import json
import random
import time

# Импорт внешних модулей и их функций
import dotenv
import httpx
from uuid import uuid4

# Объявление констант
dotenv.load_dotenv("../services/.env")
APP_PORT = os.environ.get("APP_PORT")
APP_URL = f"http://localhost:{APP_PORT}/"
SERVICE_URL = APP_URL + "flats_price"
PREDICT_URL = SERVICE_URL + "/predict?user_id="


# Количество одновременных потоков
COUNCURENCY = 4

# Максимальное время задержки после получения запроса
SLEEP_MAX_TIME = 2

# Фиксация random seed для повторяемости запросов
random.seed(42)

# Функция отправки запроса и получения ответа
async def send_request(client: httpx.AsyncClient, method: str, url: str, post_data: str | None = None) -> str:
    # Добавление user_id к запросу
    
    # Определение времени задержки
    sleep_time = SLEEP_MAX_TIME * random.random()
    try:
        # Запрос к микросервису
        if method == "post":
            response = await client.post(url=url, json=post_data, timeout=20)
        elif method == "get":
            response = await client.get(url=url)
        else:
            raise ValueError("Передан некорректный тип запроса, используйте только 'get' или 'post'")
        if response.status_code == 200:
            print(f"Получен ответ: {response.text}. Ожидание {sleep_time:.2f} секунд.")
            await asyncio.sleep(sleep_time)
        else:
            print(f"Ошибка {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Ошибка запроса: {e}")

# Worker для выполнения запросов
async def worker(queue: asyncio.Queue):
    while True:
        # Получение задачи из очереди
        task_info = await queue.get()
        if task_info is None:
            break
        await send_request(*task_info)
        queue.task_done()

async def main():
    # Загрузка json файла
    with open("test_stress.json", "r") as fd:
        data = json.load(fd)
    # Перевод в список кортежей
    data = list(data.items())
    # Создание очереди для задач
    queue = asyncio.Queue()
    async with httpx.AsyncClient() as client:
        # Определение списка переменных для выполнения задач
        tasks = [(
            client, "post", PREDICT_URL+user_id, post_data
        ) for user_id, post_data in data]
        print(f"Подготовлено {len(tasks)} запросов на предсказание.")
        # Добавление некорректных запросов на предсказание
        tasks += [(
            client, "post", PREDICT_URL+str(uuid4()), 
            {"Некорректный запрос": random.random()}
        ) for _ in range(70)]
        print(f"Добавлены некорректные запросы, общее количество {len(tasks)}")
        # Добавление GET запросов к другим эндпоинтам
        tasks += [(client, "get", SERVICE_URL) for _ in range(15)]
        tasks += [(client, "get", APP_URL) for _ in range(12)]
        tasks += [(client, "get", SERVICE_URL+"/reload") for _ in range(3)]
        print(f"Добавлены GET запросы, общее количетсов запросов {len(tasks)}")
        # Перемешивание списка задач
        random.shuffle(tasks)
        # Заполнение очереди
        for task_info in tasks:
            queue.put_nowait(task_info)
        print(f"Очередь из {queue.qsize()} запросов готова к запуску.")
        # Создание нескольких worker'ов 
        print(f"Запуск {COUNCURENCY} конкурирующих процессов с задержкой \
              до {SLEEP_MAX_TIME} секунд после успешного получения запроса.")
        workers = [asyncio.create_task(worker(queue)) for _ in range(COUNCURENCY)]
        # Ожидание завершения всех задач в очереди
        await queue.join()
        # Завершение workers
        for worker_task in workers:
            worker_task.cancel()

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    start_time = time.time() - start_time
    print(f"Тестирование завершено за {start_time:.2f} секунд")