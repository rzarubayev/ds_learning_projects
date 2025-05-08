from typing import Dict, List, Tuple
from logging import Logger

import yaml
import pandas as pd
import numpy as np
import mlflow
from mlflow.exceptions import MlflowException

import requests

class Recommendations:

    def __init__(self, settings_file: str, logger: Logger):
        with open(settings_file, "r") as fd:
            settings = yaml.safe_load(fd)
        self.target_cols = settings["target_cols"]
        self.date_col = settings["date_col"]
        self.user_col = settings["user_col"]
        self._lgr = logger
        self.is_ready = False
        self.status = "Модель и данные не загружены!"
    
    def _check_status(self):
        # Проверяем готовность сервиса
        if not self.is_ready:
            raise RuntimeError(self.status)

    @staticmethod
    def _prepare_data(
            y_true, y_pred, k: int | None = None
        ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Валидирует данные и возвращает в подходящем формате.

        Returns:
            y_true, y_pred, k
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        assert y_true.shape == y_pred.shape, (
            "y_true и y_pred должны иметь одинаковый размер")
        assert y_pred.ndim == 2, (
            "Ожидается двумерный массив (n_samples, n_items)")
        if k: k = min(k, y_pred.shape[1])
        else: k = y_pred.shape[1]

        return y_true, y_pred, k

    @staticmethod
    def _map_k(y_true: np.ndarray, y_pred: np.ndarray, 
               k: int) -> float:
        "Возвращает MAP@k (Mean Average Precision)"
        avg_precisions = []
        for true_row, pred_row in zip(y_true, y_pred):
            top_k = np.argsort(-pred_row)[:k]
            rel = true_row[top_k]
            precisions = np.cumsum(rel) / (np.arange(k) + 1)
            if np.sum(rel) > 0:
                ap = np.sum(precisions * rel) / np.sum(rel)
            else:
                ap = 0.0
            avg_precisions.append(ap)
        return np.mean(avg_precisions)

    @staticmethod
    def _precision_k(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     k: int) -> float:
        "Возвращает Precision@k"
        top_k = np.argsort(-y_pred, axis=1)[:, :k]
        rel = np.take_along_axis(y_true, top_k, axis=1)
        return np.mean(np.sum(rel, axis=1) / k)

    @staticmethod
    def _recall_k(y_true: np.ndarray, 
                  y_pred: np.ndarray, 
                  k: int) -> float:
        "Возвращает Recall@k"
        recalls = []
        top_k = np.argsort(-y_pred, axis=1)[:, :k]
        rel = np.take_along_axis(y_true, top_k, axis=1)
        for true_row, rel_row in zip(y_true, rel):
            n_relevant = np.sum(true_row)
            if n_relevant == 0:
                continue
            recalls.append(np.sum(rel_row) / n_relevant)
        return np.mean(recalls) if recalls else 0.0

    def load(self, data_path: str, storage_options: dict | None, 
             model_name: str, flavor: str, tracking_uri: str, 
             stage: str = "Production"):
        """
        Загружает последюню версию модели и данные за текущий месяц
        """
        self.is_ready = False
        self.status = "Модель и данные не загружены!"
        
        if self._lgr: self._lgr.info(
            f"Загрузка модели {model_name}")
        
        try:
            resp = requests.get(f"{tracking_uri}/version")
            if self._lgr: self._lgr.info((
                "MLflow сервер доступен, версия сервера: "
                f"{resp.text.strip()}, версия клиента: "
                f"{mlflow.__version__}"))
        except Exception as e:
            raise ValueError(
                f"Проблема с сервером MLflow: {e}")
        
        mlflow.set_tracking_uri(tracking_uri)
        
        # Получение функции загрузки по flavor
        try:
            load_fn = getattr(mlflow, flavor).load_model
        except AttributeError:
            raise ValueError(
                f"Flavor {flavor} не поддерживается MLflow")
        
        # Загрузка модели
        try:
            self.model = load_fn(f"models:/{model_name}/{stage}")
        except MlflowException as e:
            raise ValueError(
                f"Ошибка загрузки модели '{model_name}': {e}")

        if self._lgr: self._lgr.info((
            f"Последняя версия модели {model_name} "
            f"в стадиии {stage} загружена"))

        # Загрузка датасета
        if self._lgr: self._lgr.info("Загрузка датасета")
        try:
            self.data = pd.read_parquet(
                data_path, storage_options=storage_options)
            self.data = self.data.set_index(self.user_col)
            assert self.data.index.is_unique, (
                "Датасет должен содержать данные за месяц без "
                "дубликатов идентификаторов клиента")
        except Exception as e:
            raise ValueError(f"Не удалось загрузить датасет: {e}")

        # Получение предсказаний по всему датасету
        if self._lgr: self._lgr.info(
            "Датасет загружен, получение предсказаний")
        try:
            self.probas = pd.DataFrame(
                self.model.predict_proba(self.data),
                index=self.data.index,
                columns=self.target_cols)
        except Exception as e:
            raise ValueError(f"Модель не может обработать датасет: {e}")
        
        if self._lgr: self._lgr.info("Предсказания получены, модель готова")
        
        self.is_ready = True
        self.status = "Модель и данные загружены"

    def get_metrics(self, user_item_true: Dict[int, List[int]], k: int | None):
        """
        Рассчитвыает метрики модели, исходя из факта 
        использованния продуктов пользователями в 
        текущем месяце, в виде бинарных значений 0/1 
        для каждого продукта: 
        
        `true = {
            user_1: [item_1, item_2,..,item_n], 
            ...,
            user_m: [item_1, item_2,...,item_n]
        }`

        Возвращает метрики:
        - MAP@k_all (Mean Average Precision) - для 
        всего датасета;
        - MAP@k - для клиентов с продуктами;
        - Precision@k_all - для всего датасета
        - Precision@k - для клиентов с продуктами;
        - Recall@k
        """
        self._check_status()

        # Преобразование в pandas DataFrame, 
        # заполняем пропуски на всякий случай
        y_true = pd.DataFrame.from_dict(
            user_item_true, orient="index",
            columns=self.target_cols) \
                .fillna(0).astype(int)
        # Пересечение двух датасетов, на случай получения 
        # данных не по всем пользователям
        comb = pd.concat(
            [y_true, self.probas],
            axis=1, join="inner",
            keys=["true", "pred"])
        # Оценка наличия продуктов у клиента
        mask = comb["true"].any(axis=1)
        if mask.sum() == 0:
            raise ValueError(
                "У клиентов нет ни одного продукта")

        # Все клиенты
        y_true_all, y_pred_all, k = self._prepare_data(
            comb["true"], comb["pred"], k)

        # Активные клиенты
        y_true_act, y_pred_act, k = self._prepare_data(
            comb.loc[mask, "true"], comb.loc[mask, "pred"], k)
        
        result = {}
        
        # расчет метрик
        result["MAP@k_all"] = self._map_k(
            y_true_all, y_pred_all, k)
        result["MAP@k"] = self._map_k(
            y_true_act, y_pred_act, k)
        result["Precision@k_all"] = self._precision_k(
            y_true_all, y_pred_all, k)
        result["Precision@k"] = self._precision_k(
            y_true_act, y_pred_act, k)
        result["Recall@k"] = self._recall_k(
            y_true_act, y_pred_act, k)
        
        return result

    def get_user_items(self, user_id: int, 
                          top_k: int | None = None):
        """
        Возвращает `TOP-k` продуктов клиента, 
        с вероятностью их использования в 
        следующем месяце.
        """
        self._check_status()

        if self._lgr: self._lgr.info(
                f"Получение рекомендаций по клиенту '{user_id}'")
        
        # Получаем данные по клиенту
        try:
            user = self.data.loc[user_id, :]
        except KeyError as e:
            raise ValueError(
                f"Клиент id '{user_id}' не найден в датасете")
        # Получаем предсказания
        recs = pd.Series(
            self.model.predict_proba(user),
            index=self.target_cols).sort_values(ascending=False)
        # Ограничиваем предсказания
        if top_k:
            recs = recs[:top_k]

        return recs.to_dict()

    def get_item_users(self, item_name: str,
                          thres: float = 0.5,
                          round: int = 2):
        """
        Возвращает список пользователей, у которых 
        вероятность использования продуктов в 
        следующем месяце не ниже `thres`.
        """
        self._check_status()

        # Получаем данные по продукту
        try:
            users = self.probas[item_name]
        except KeyError as e:
            raise ValueError(
                f"Неизвестный продукт: {item_name}")
        
        # Оставляем пользователей не ниже порога
        users = users[users >= thres].round(round)

        return users.to_dict()