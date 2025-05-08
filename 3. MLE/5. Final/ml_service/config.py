from typing import List
import numpy as np

from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str #= "Recsys"
    APP_SETTINGS_DIR: str #= "settings"

    # Настройки для подключения к S3
    S3_BUCKET: str #= <YOUR_BUCKET_NAME>
    MLFLOW_S3_ENDPOINT_URL: str #= "https://storage.yandexcloud.net"
    AWS_REGION: str #= "ru-central1"
    AWS_ACCESS_KEY_ID: str #= <YOUR_ACCESS_KEY>
    AWS_SECRET_ACCESS_KEY: str #= <YOUR_SECRET_KEY>
    AWS_SIGV: str #= "s3v4"
    
    # Модель и датасет Bank Products
    BP_DATA_FILE: str #= "bank_products/test_data.parquet"
    BS_S3_STORAGE: bool #= True
    BP_REG_MODEL_NAME: str #= "bank_products_3PX"
    BP_FLAVOR: str #= "catboost"
    BP_TRACKING_URI: str #= "http//mlflow-server:5000"
    BP_STAGE: str #= 'Production'
    BP_SETTINGS: str #= "settings.yaml"
    # Порог по умолчанию
    BP_THRESHOLD: float #= 0.5
    # Количество ТОП предсказаний для метрик
    BP_TOP_K: int #= 3

# Получение настроек    
settings = Settings()