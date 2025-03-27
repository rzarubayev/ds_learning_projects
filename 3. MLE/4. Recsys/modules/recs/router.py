from fastapi import APIRouter
import httpx
import numpy as np

from config import settings
from .service import Recommendations, SimilarItems, logger

# Путь к файлам с рекомендациями
S3_PATH = f"s3://{settings.AWS_S3_BUCKET}/{settings.RECS_DIR}/"

# Настройки хранилища для загрузки файлов
STORAGE_OPTIONS = {
    "endpoint_url": settings.AWS_ENDPOINT_URL, 
    "key": settings.AWS_ACCESS_KEY_ID, 
    "secret": settings.AWS_SECRET_ACCESS_KEY,
    "client_kwargs":{
        "region_name": settings.AWS_REGION},
    "config_kwargs": {
        "signature_version": settings.AWS_SIGV}
}

# Создание FastAPI роутера
router = APIRouter(prefix="/recs", tags=["Recommendations"])

# Загрузка оффлайн рекомендаций
rec_store = Recommendations()
rec_store.load(
    "personal", S3_PATH+settings.PERSONAL_RECS_FILE,
    engine="pyarrow", storage_options=STORAGE_OPTIONS)
rec_store.load(
    "default", S3_PATH + settings.POPULAR_RECS_FILE,
    engine="pyarrow", storage_options=STORAGE_OPTIONS)

# Загрузка похожих объектов
similar = SimilarItems()
similar.load(
    S3_PATH + settings.SIMILAR_RECS_FILE,
    engine="pyarrow", storage_options=STORAGE_OPTIONS)

# Функция для получения событий из Events Store
async def get_last_events(user_id: int, k: int | None = None):
    """
    Загружает k последних событий из Event Store
    """
    url = settings.EVENTS_STORE_URL + "/get"
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    params = {"user_id": user_id}
    if k:
        params[k] = k
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.post(url, headers=headers, params=params)
            resp.raise_for_status()
            result = resp.json()
            return result["events"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error: {e.response.status_code}")
            logger.info(f"Response: {resp.json()}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Request Error: {str(e)}")
            return []

# Функция удаления дублирующихся идентификаторов
def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids

# Доступность сервиса
@router.get("/")
@router.get("")
async def status():
    return {"status": "ok"}

# Предоставление рекомендаций
@router.get("/{user_id}")
async def recommend(user_id: int, k: int = settings.RECS_COUNT):
    """
    Предоставляет комбинированные рекомендации
    """
    k_online = round(k * settings.ONLINE_RATIO)
    # Получаем все последние события пользователя
    events = await get_last_events(user_id)
    # Получаем офлайн рекомендации
    recs_offline = await recommend_offline(
        user_id=user_id, k=k, events=events
    )
    recs_offline = recs_offline["recs"]

    # Получаем онлайн рекомендации
    recs_online = await recommend_online(
        user_id=user_id, k=k_online, events=events
    )
    recs_online = recs_online["recs"]

    # Комбинируем рекомендации
    recs_blended = []
    min_len = min(len(recs_offline), len(recs_online))
    for i in range(min_len):
        recs_blended.append(recs_offline[i])
        recs_blended.append(recs_online[i])
    if len(recs_offline) > min_len:
        recs_blended += recs_offline[min_len:]
    if len(recs_online) > min_len:
        recs_blended += recs_online[min_len:]
    
    # Очищаем дубликаты и оставляем k первых
    recs_blended = dedup_ids(recs_blended)[:k]

    return {"recs": recs_blended}
    

# Предоставление оффлайн рекомендаций
@router.get("/offline/{user_id}")
async def recommend_offline(
        user_id: int, k: int = settings.RECS_COUNT, 
        events: list | None = None):
    """
    Предоставляет только offline рекомендации
    """
    # Загрузка событий, если не переданы
    if events is None:
        events = await get_last_events(user_id)

    # Получение оффлайн рекомендаций
    recs = rec_store.get(user_id=user_id, k=k, exclude=events)
    return {"recs": recs}

# Предоставление онлайн рекомендаций
@router.get("/online/{user_id}")
async def recommend_online(
        user_id: int, k: int = settings.RECS_COUNT, 
        events: list | None = None):
    """
    Предоставляет только online рекомендации
    """
    # Загрузка событий, если не переданы
    if events is None:
        events = await get_last_events(user_id)

    # Получение последних событий и количества рекомендаций
    events_count = settings.LAST_TRACK_RATIO
    # Сокращение рекомендаций, если событий меньше
    if len(events) < len(events_count):
        events_count = events_count[:len(events)]
        events_count = events_count / np.sum(events_count)
    events_count = np.round(events_count * k).astype(int).tolist()
    last_events = dict(zip(events[:len(events_count)], events_count))

    # Получение онлайн рекомендаций
    recs = []
    for item_id, k in last_events.items():
        sim_item = similar.get(item_id=item_id, k=k, exclude=events)
        logger.info(f"{sim_item}")
        # Исключаем уже полученные рекомендации
        events = list(set(events + recs))

    return {"recs": recs}
