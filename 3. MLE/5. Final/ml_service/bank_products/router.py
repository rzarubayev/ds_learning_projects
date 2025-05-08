import os
import logging
from typing import Dict, List
from contextlib import asynccontextmanager

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from config import settings
from monitoring import monitoring_metrics
from .service import Recommendations

# Общие настройки
PREFIX = "/bank_products"
MODEL_NAME = "Bank_Products"
LOAD_URI = "/load_model"
USERS_URI = "/get_product_clients"
ITEMS_URI = "/get_client_products"
METRICS_URI = "/get_metrics"

# Настройки хранилища для загрузки файлов
if settings.BS_S3_STORAGE:
    DATA_FILE = f"s3://{settings.S3_BUCKET}/{settings.BP_DATA_FILE}"
    STORAGE_OPTIONS = {
        "endpoint_url": settings.MLFLOW_S3_ENDPOINT_URL, 
        "key": settings.AWS_ACCESS_KEY_ID, 
        "secret": settings.AWS_SECRET_ACCESS_KEY,
        "client_kwargs":{
            "region_name": settings.AWS_REGION},
        "config_kwargs": {
            "signature_version": settings.AWS_SIGV}}
else:
    DATA_FILE = settings.BP_DATA_FILE
    STORAGE_OPTIONS = None

SETTINGS_FILE = os.path.join(
    settings.APP_SETTINGS_DIR,
    settings.BP_SETTINGS)

# Модель для загрузки датасета
class ModelData(BaseModel):
    data_path: str = DATA_FILE
    storage_options: str | None = STORAGE_OPTIONS
    reg_model_name: str = settings.BP_REG_MODEL_NAME
    flavor: str = settings.BP_FLAVOR
    tracking_uri: str = settings.BP_TRACKING_URI
    stage: str = settings.BP_STAGE

# Получение логера
logger = logging.getLogger("uvicorn.error")

# Создание сервиса рекомендаций
bank_products = Recommendations(
    settings_file=SETTINGS_FILE,
    logger=logger)

# Загрузка модели при запуске
@asynccontextmanager
async def lifespan(app):
    print(settings.BP_TRACKING_URI)
    print()
    # Загружаем модель
    await load_model(ModelData())
    yield

# Создание FastAPI роутера
router = APIRouter(
    prefix=PREFIX, 
    tags=[MODEL_NAME],
    lifespan=lifespan)
    
# Эндпоинт загрузки модели
@router.post(LOAD_URI)
async def load_model(json_data: ModelData):
    try:
        bank_products.load(
            data_path=json_data.data_path, 
            storage_options=json_data.storage_options,
            model_name=json_data.reg_model_name, 
            flavor=json_data.flavor,
            tracking_uri=json_data.tracking_uri, 
            stage=json_data.stage)
    except (ValueError, AssertionError) as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(
            status_code=503,
            detail=str(re))
    except Exception as e:
        logger.error((
            "Ошибка загрузки модели "
            f"{json_data.reg_model_name}: {e}"))
        raise HTTPException(
            status_code=500,
            detail=str(e))
    try:
        # Обновление метрик
        monitoring_metrics["model_reqs"].labels(
            model_name=MODEL_NAME,
            endpoint=LOAD_URI).inc()
    except Exception as e:
        logger.error(
            f"Ошибка при обновлении метрик: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e))
    # Отправка результата
    return {"status": bank_products.status}

# Эндпоинт рекомендаций клиентов по продукту
@router.post(USERS_URI)
async def product_clients(
        product_name: str, 
        threshold: float = settings.BP_THRESHOLD,
        round: int = 2):
    try:
        result = bank_products.get_item_users(
            item_name=product_name,
            thres=threshold,
            round=round)
    except (ValueError, AssertionError) as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(
            status_code=503,
            detail=str(re))
    except Exception as e:
        logger.error((
            "Ошибка при получении пользователей "
            f"по продукту '{product_name}: {e}'"))
        raise HTTPException(
            status_code=500,
            detail=str(e))
    try:
        # Обновление метрик
        monitoring_metrics["model_reqs"].labels(
            model_name=MODEL_NAME,
            endpoint=USERS_URI).inc()
    except Exception as e:
        logger.error(
            f"Ошибка при обновлении метрик: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e))
    # Отправка результата
    return result

# Эндпоинт рекомендаций продуктов по клиенту
@router.post(ITEMS_URI)
async def client_products(
        user_id: int, top_k: int = settings.BP_TOP_K):
    try:
        result = bank_products.get_user_items(
            user_id, top_k)
    except (ValueError, AssertionError) as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(
            status_code=503,
            detail=str(re))
    except Exception as e:
        logger.error((
            "Ошибка при получении продуктов "
            f"по пользователю '{user_id}: {e}'"))
        raise HTTPException(
            status_code=500,
            detail=str(e))
    try:
        # Обновление метрик
        monitoring_metrics["model_reqs"].labels(
            model_name=MODEL_NAME,
            endpoint=ITEMS_URI).inc()
    except Exception as e:
        logger.error(
            f"Ошибка при обновлении метрик: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e))
    # Отправка результата
    return result

# Эндпоинт для расчета метрик
@router.put(METRICS_URI)
async def metrics(
        top_k: int = settings.BP_TOP_K,
        y_true: Dict[str, List[int]] = Body(...)):
    try:
        y_true = {int(k): v for k, v in y_true.items()}
        result = bank_products.get_metrics(
            user_item_true=y_true,
            k=top_k)
    except (ValueError, AssertionError) as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(
            status_code=503,
            detail=str(re))
    except Exception as e:
        logger.error(
            f"Ошибка при получении метрик: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e))
    try:
        # Обновление метрики запроса
        monitoring_metrics["model_reqs"].labels(
            model_name=MODEL_NAME,
            endpoint=METRICS_URI).inc()
        # Обновление метрик модели
        for metric, val in result.items():
            monitoring_metrics["recsys_metrics"].labels(
                model_name=MODEL_NAME,
                metric=metric).set(val)
    except Exception as e:
        logger.error(
            f"Ошибка при обновлении метрик: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e))
    # Отправка результата
    return result

# Доступность сервиса
@router.get("/")
@router.get("")
async def status():
    if not bank_products.is_ready:
        raise HTTPException(
            status_code=503, 
            detail=bank_products.status)
    try:
        # Обновление метрик
        monitoring_metrics["model_reqs"].labels(
            model_name=MODEL_NAME,
            endpoint=PREFIX).inc()
    except Exception as e:
        logger.error(
            f"Ошибка при обновлении метрик: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e))
    return {"status": "ok"}