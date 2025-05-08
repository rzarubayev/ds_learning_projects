from typing import List
import numpy as np

from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str #= "Recsys"

    # Настройки для подключения к S3
    AWS_ENDPOINT_URL: str #= "https://storage.yandexcloud.net"
    AWS_REGION: str #= "ru-central1"
    AWS_S3_BUCKET: str #= <YOUR_BUCKET_NAME>
    AWS_ACCESS_KEY_ID: str #= <YOUR_ACCESS_KEY>
    AWS_SECRET_ACCESS_KEY: str #= <YOUR_SECRET_KEY>
    AWS_SIGV: str #= "s3v4"
    
    # Пути к файлам с рекомендациями
    RECS_DIR: str #= "recsys/recommendations"
    PERSONAL_RECS_FILE: str #= "personal_als.parquet"
    POPULAR_RECS_FILE: str #= "top_popular.parquet"
    SIMILAR_RECS_FILE: str #= "similar.parquet"
    LAST_EVENTS_FILE: str #= "last_events.parquet"

    # Соотношение последних треков
    LAST_TRACK_RATIO: List[float] #= [0.5, 0.3, 0.2]

    # Соотношение оффлайн рекомендаций
    ONLINE_RATIO: float #= 0.5

    # Количество рекомендаций по умолчанию
    RECS_COUNT: int #= 10

    # URL сервиса Events store
    EVENTS_STORE_URL: str #= "http://localhost:8080/events_store"

    EVENTS_STORE_MAX_LEN: int #= 100

    class Config:
        env_file = ".env"
        # Разрешаем наличие других параметров
        extra = "allow"
    
    # Приведение соотношений последних треков к 1 в сумме, 
    # если указано некорректное соотношение, типа [3, 2, 1]
    @field_validator("LAST_TRACK_RATIO")
    @classmethod
    def check_ltr(cls, v):
        v = np.array(v)
        if np.sum(v) != 1:
            return v / np.sum(v)
        return v
    
    # Проверка корректности ONLINE_RATIO
    @field_validator("ONLINE_RATIO")
    @classmethod
    def chec_or(cls, v):
        if (v > 1) or (v < 0):
            raise ValueError("ONLINE_RATIO должен быть от 0 до 1")
        return v

# Получение настроек    
settings = Settings()