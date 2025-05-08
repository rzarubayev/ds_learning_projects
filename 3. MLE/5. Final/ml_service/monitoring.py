from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter

instrumentator = Instrumentator(
    should_instrument_requests_inprogress=True
)

# Метрика медианного значения регрессионных моделей машинного обучения

monitoring_metrics = {}

monitoring_metrics["model_reqs"] = Counter(
    name="model_success_requests",
    documentation="Успешно обработанные запросы к модели",
    labelnames=["model_name", "endpoint"])

monitoring_metrics["recsys_metrics"] = Gauge(
    name="models_metric",
    documentation="Метрики моделей",
    labelnames=["model_name", "metric"])