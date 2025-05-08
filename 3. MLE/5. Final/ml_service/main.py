import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from config import settings
from monitoring import instrumentator
from bank_products.router import router as bank_products_router

logger = logging.getLogger("uvicorn.error")

    
# создаём приложение FastAPI
app = FastAPI(title=settings.APP_NAME)

# Добавление роутеров микросервиса
app.include_router(bank_products_router)

# Инициализация и запуск экспортера метрик
instrumentator.instrument(app).expose(app)

# Вывод ошибок в json
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail})

# Статус микросервиса
@app.get("/")
async def status():
    """
    Возвращает статус микросервиса
    """
    return {"status": "OK"}