from fastapi import APIRouter

from config import settings
from .service import EventStore

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

# Создание FastAPI router
router = APIRouter(prefix="/events_store", tags=["Events"])

# Загрузка последних событий пользователей
event_store = EventStore(
    max_len=settings.EVENTS_STORE_MAX_LEN)

event_store.load(
    S3_PATH + settings.LAST_EVENTS_FILE, 
    engine="pyarrow", storage_options=STORAGE_OPTIONS)

# Доступность сервиса
@router.get("/")
@router.get("")
async def status():
    return {"status": "ok"}

# Добавление нового события
@router.post("/put")
async def put(user_id: int, item_id: int):
    """
    Добавляет новое событие в хранилище
    """
    event_store.put(user_id, item_id)
    return {"result": "ok"}

# Предоставление событий
@router.post("/get")
async def get(user_id: int, k: int | None = None):
    """
    Возвращает k последних событий пользователя 
    из хранилища. Если `k=None`, возвращает все события.
    """
    events = event_store.get(user_id, k)
    return {"events": events}