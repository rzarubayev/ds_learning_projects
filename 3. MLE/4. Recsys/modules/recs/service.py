import logging
import pandas as pd

logger = logging.getLogger("uvicorn.error")

class Recommendations:

    def __init__(self):

        self._recs = {"personal": None, "default": None}
        self._stats = {
            "request_personal_count": 0,
            "request_default_count": 0,
        }

    def load(self, type, path, **kwargs):
        """
        Загружает рекомендации из файла
        """
        logger.info(f"Loading recommendations, type: {type}")
        self._recs[type] = pd.read_parquet(path, **kwargs)
        if type == "personal":
            self._recs[type] = self._recs[type].set_index("user_id")
        logger.info(f"Loaded")

    def get(self, user_id: int, k: int=10, exclude: list=[]):
        """
        Возвращает список рекомендаций для пользователя
        """
        try:
            logger.info("Getting offline recommendations")
            recs = self._recs["personal"].loc[user_id]
            recs = recs[
                ~recs["item_id"].isin(exclude)
            ]["item_id"].to_list()[:k]
            self._stats["request_personal_count"] += 1
        except KeyError:
            recs = self._recs["default"]
            recs = recs[
                ~recs["item_id"].isin(exclude)
            ]["item_id"].to_list()[:k]
            self._stats["request_default_count"] += 1
        except:
            logger.error("No recommendations found")
            recs = []

        return recs

    def stats(self):
        logger.info("Stats for recommendations")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value} ") 

class SimilarItems:

    def __init__(self):
        self._similar_items = None

    def load(self, path, **kwargs):
        """
        Загружает данные из файла
        """
        logger.info("Loading similar items")
        self._similar_items = pd.read_parquet(path, **kwargs)
        self._similar_items = self._similar_items.set_index("item_id")
        logger.info("Loaded")

    def get(self, item_id: int, k: int=10, exclude: list=[]):
        """
        Возвращает список похожих треков
        """
        try:
            logger.info("Getting similar items")
            i2i = self._similar_items.loc[item_id]
            i2i = i2i[~i2i["similar_id"].isin(exclude)].head(k)
            i2i = i2i[["similar_id", "score"]].to_dict(orient="list")
        except KeyError:
            logger.error("No recommendations found")
            i2i = {"similar_id": [], "score": []}
        
        return i2i