import logging
import pandas as pd

logger = logging.getLogger("uvicorn.error")

class EventStore:

    def __init__(self):
        self.events = {}

    def load(self, path, **kwargs):
        """
        Загружает последние события из файла
        """
        logger.info("Loading last events")
        df = pd.read_parquet(path, **kwargs)
        df = df.set_index("user_id")["item_id"].to_dict()
        # Преобразование в list[int]
        self.events = {k: v.astype(int).tolist() for k, v in df.items()}
        logger.info("Loaded")
        
    def put(self, user_id, item_id):
        """
        Сохраняет событие
        """
        logger.info(f"Adding {item_id} to user {user_id}")
        user_events = self.events.get(user_id, [])
        self.events[user_id] = [item_id] + user_events
        logger.info(f"New user events: {self.events[user_id]}")

    def get(self, user_id: int, k: int | None = None):
        """
        Возвращает события пользователя
        """
        user_events = self.events.get(user_id, [])
        logger.info(f"User {user_id} events: {user_events}")
        if k: user_events = user_events[:k]
        return user_events