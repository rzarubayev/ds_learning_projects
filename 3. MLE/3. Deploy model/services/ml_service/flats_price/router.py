import os

from fastapi import APIRouter, HTTPException, Query
from core.config import settings
from flats_price.schemas import FlatsPriceParams

# Импорт модулей текущего проекта
from flats_price.handler import FlatsPriceHandler
from core.monitoring import prediction_median, prediction_hist_flats_price


# Объявление констант
MODEL_NAME = "flats_price"
MODEL_DIR = os.path.join(settings.MODEL_DIR, settings.flats_price.FLATS_PRICE_MODEL)
RESULT_COUNT = settings.flats_price.FLATS_PRICE_RESULT_COUNT


# Создание экземпляра APIRouter
router = APIRouter(
    prefix=f"/{MODEL_NAME}",
    tags=["Предсказание цен квартир"]
)

# Создание экземпляра обработчика модели предсказания стоимости квартир
handler = FlatsPriceHandler(
    model_dir=MODEL_DIR, 
    result_count=RESULT_COUNT
)


# Эндпоинты роутера ------------------------------------------------------------------------------

# Статус модели
@router.get("", description="Вывод статуса модели")
async def get_model_status():
    result = handler.get_status()
    if result["status_code"] == 200:
        return {"status": "OK"}
    else:
        raise HTTPException(
            status_code=result["status_code"],
            detail=result["detail"]
        )

# Перезагрузка модели из файла
@router.get("/reload", description="Перезагрузка модели из директории")
async def reload_model( 
    model_dir: str = Query(
        default=None, 
        description="Относительный путь к директории модели от основной, если не указан - используется предыдущая модель.")
):
    if model_dir is not None:
        model_dir = os.path.join(settings.MODEL_DIR, model_dir)
    result = handler.load(
        model_dir=model_dir
    )
    if result["status_code"] == 200:
        return {"status": result["detail"]}
    else: 
        raise HTTPException(
            status_code=result["status_code"],
            detail=result["detail"]
        )

# Предсказание модели
@router.post("/predict", description="Получение предсказания модели")
async def get_model_prediction(user_id: str, params: FlatsPriceParams):
    result = handler.handle(params.model_dump())
    if result["status_code"] == 200:
        prediction_hist_flats_price.observe(result["detail"])
        prediction_median.labels(
            model=MODEL_NAME, results_count=RESULT_COUNT
        ).set(handler.get_agg("median"))
        return {"user_id": user_id, "prediction": result["detail"]}
    else:
        raise HTTPException(
            status_code=result["status_code"],
            detail=result["detail"]
        )