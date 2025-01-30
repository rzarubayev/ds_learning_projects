from typing import List

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str #= "ML-service"
    MODEL_DIR: str #= "/models"

    # Настройки для модели предсказания квартир
    class FlatsPriceSettings(BaseSettings):
        FLATS_PRICE_MODEL: str #= "flats_price"
        FLATS_PRICE_RESULT_COUNT: int #= 60
        FLATS_PRICE_RESULT_BUCKETS: List[float] #= [2.5e6, 5e6, 7.5e6, 1e7, 1.25e7, 1.5e7, 2e7, 2.5e7, 3e7, 5e7, 1.5e8]

    flats_price: FlatsPriceSettings = FlatsPriceSettings()
    
settings = Settings()