import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager

from config import settings
from modules.recs.router import router as recs_router
from modules.event_store.router import router as evst_router

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting")
    yield
    logger.info("Stopping")
    
# создаём приложение FastAPI
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# Добавление роутеров модулей микросервиса
app.include_router(recs_router)
app.include_router(evst_router)

# Статус микросервиса
@app.get("/")
async def status():
    """
    Возвращает статус микросервиса
    """
    return {"status": "OK"}