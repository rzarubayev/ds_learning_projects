from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator.metrics import Info
from fastapi import Request
from prometheus_client import Histogram, Gauge
from core.config import settings

instrumentator = Instrumentator(
    should_instrument_requests_inprogress=True
)

# Метрики прикладного уровня моделей обучения -------------------------------------------------------

# Метрика медианного значения регрессионных моделей машинного обучения
prediction_median = Gauge(
    name="prediction_median",
    documentation="Median value of a ML model predictions.",
    labelnames=["model", "results_count"]
)

# Гистограмма значений модели предсказания стоимости квартир
prediction_hist_flats_price = Histogram(
    name="prediction_hist_flats_price",
    documentation="Values of a flats price model predictions.",
    buckets=settings.flats_price.FLATS_PRICE_RESULT_BUCKETS
)