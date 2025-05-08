import pandas as pd
import json
from mlflow.pyfunc import PythonModel

# Трансформер в виде модели pyfunc для получения выборки, очищенной от записей 
# за последний месяц, так как по ним нет целевых значений, а также от записей, 
# у которых пропуски в месяцах, так как запись перед пропуском также не имеет 
# целевых значений - следующий месяц отсутствует.
class BankProductTargetExtractor(PythonModel):    
    def load_context(self, context):
        with open(context.artifacts["params"], "r") as fd:
            params = json.load(fd)
        self.user_col = params["user_col"]
        self.date_col = params["date_col"]
        self.rel_col = params["rel_col"]
        self.target_cols = params["target_cols"]

    def transform(self, X):
        # Получение маски записей, количество дней со следующей более 31 дня
        mask = X.groupby(self.user_col)[self.date_col] \
            .diff(-1).abs().fillna(pd.Timedelta(days=32)) \
            .dt.days < 32
        # Добавление клиентов, по которым закончились отношения, чтобы модель 
        # знала, что у таких клиентов не будет новых продуктов в следующем 
        # месяце - по ним все записи целевых значений будут False. Исключаются 
        # клиенты, которые закончили отношения в последнем месяце, так как он 
        # не используется в датасете.
        last_month = X[self.date_col].max()
        mask = (X[self.date_col] != last_month) & \
            ~X[self.rel_col] | mask
        # Получение таргетов путем сравнения неиспользованных в текущем месяце 
        # продуктов и использованных продуктов со сдвигом записей на 1 месяц
        y = X.groupby(self.user_col)[self.target_cols] \
            .shift(-1) & (~X[self.target_cols])
        return X[mask].copy(), y[mask]

    def predict(self, context, model_input):
        return self.transform(model_input)