import logging
from collections import defaultdict, deque
import pandas as pd

logger = logging.getLogger("uvicorn.error")

class EventStore:

    def __init__(self, max_len: int = 100):
        self.max_len = max_len
        self.events = defaultdict(lambda: deque(maxlen=max_len))

    def load(self, path, **kwargs):
        """
        Загружает последние события из файла
        """
        logger.info("Loading last events")
        df = pd.read_parquet(path, **kwargs)
        grouped = df.set_index("user_id")["item_id"].apply(list)
        # Преобразование в list[int]
        self.events = defaultdict(lambda: deque(maxlen=self.max_len), {
            k: deque(map(int, v), maxlen=self.max_len) 
            for k, v in grouped.items()
        })
        logger.info("Loaded")
        
    def put(self, user_id, item_id):
        """
        Сохраняет событие
        """
        logger.info(f"Adding {item_id} to user {user_id}")
        self.events[user_id].appendleft(item_id)
        logger.info(f"New user events: {self.events[user_id]}")

    def get(self, user_id: int, k: int | None = None):
        """
        Возвращает события пользователя
        """
        user_events = list(self.events.get(user_id, deque()))
        logger.info(f"User {user_id} events: {user_events}")
        return user_events[:k] if k else user_events