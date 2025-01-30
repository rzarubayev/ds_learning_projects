import sys
import os
from fastapi import FastAPI
from core.config import settings
from core.monitoring import instrumentator

# В данном проекте использована модульная архитектура FastAPI приложения 
# для отделения различных моделей обучения, которые могут подключаться 
# отдельными модулями под каждую задачу. Модуль для задачи предсказания
# стоимости квартир находится в папке flats_price.

# Импорт роутера с моделью предсказания стоимости квартир
from flats_price.router import router as router_flats_price

# Приложение FastAPI
app = FastAPI(title=settings.APP_NAME)

# Добавление роутера для предсказания стоимости квартир
app.include_router(router_flats_price)

# Инициализация и запуск экспортера метрик
instrumentator.instrument(app).expose(app)

# Проверка статуса работы сервиса FastAPI
@app.get("/")
async def get_status():
    return {"status": "OK"}