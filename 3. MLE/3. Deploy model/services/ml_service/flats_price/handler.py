import os

import json
import yaml
import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Функция для обработки столбцов, аналогично обработчику выбросов
# без самой обработки выбросов 
def columns_selector(df: pd.DataFrame, drop_cols: list | None = None, change_col: list | None = None):
    """
    Функция поизводит преобразования столбцов, аналогичные обработке выбросов - 
    удаление и преименование столбцов, изменение типов данных.
    """
    result = df.copy()
    if drop_cols is not None:
        result.drop(columns=drop_cols, inplace=True)
    if change_col is not None:
        for col in change_col:
            if "astype" in col:
                result[col["name"]] = result[col["name"]].astype(col["astype"])
            if "rename" in col:
                result.rename(columns={col["name"]: col["rename"]}, inplace=True)
    return result

# Функция для отбора признаков
def features_selector(df: pd.DataFrame, feat_proc: Pipeline, columns: list):
    """
    Функция для добавления новых признаков трансформером и их отбор.
    """
    # Столкнулся с проблемой autofeat - не добавлет столбцы для бинарных признаков, если 
    # нет противоположных значений, продублировал строку и инвертировал значения.
    # Вообще столкнулся с большим количеством сложностей именно при работе с autofeat 
    # в прошлом задании.
    new_df = pd.concat([df, df], ignore_index=True)
    new_df.loc[1, ["has_elevator", "is_apartment"]] = ~df[["has_elevator", "is_apartment"]].iloc[0]
    
    return pd.DataFrame(
        feat_proc.transform(new_df),
        columns=feat_proc["newfeat"].get_feature_names_out()
    )[columns].iloc[0]

# Класс обработчик модели предсказания цены квартиры для FastAPI сервиса
class FlatsPriceHandler:
    """
    Класс FloorPriceHandler загружает дамп модели предсказания стоимости квартир,
    а также входных параметров модели из файла input_example.json для модели.
    
    Parameters
    ---------

    model_dir : строка, обязательный параметр. 
        Относительный путь к директории модели предсказания стоимости квартир из основной директории для моделей. 
        В директории в обязательном порядке должны находиться следующие файлы:
        - `model.cb` - файл модели CatBoost;
        - `outlier_params.yaml` - параметры для обработчика выбросов, используется для проведения аналогичных преобразований 
        столбцов, для подготовки данных к дальнейшей обработке трансформерами;
        - `feat_processor.pkl` - дамп трансформера для добавления новых признаков;
        - `input_example.json` - файл MLflow для получения списка признаков по результатам их отбора.


    result_count : целочисленный, обязательный параметр. 
        Количество значений результата для рассчета статистики.
    """
    def __init__(
            self, 
            model_dir: str,
            result_count: int
        ) -> None:
        self.model = None
        self.model_dir = model_dir
        self.files = {}
        self.file_types = ["model", "outl_proc", "feat_proc", "columns"]
        self.last_load = {}
        self.result_count = result_count
        self.results = pd.Series()
        self.load(model_dir)

    def __load_file(self, file_type: str, file_path: str) -> str:
        if file_type not in self.file_types:
            raise ValueError(f"Указан неверный тип, используйте один из {self.file_types}.")
        try:
            if not os.path.exists(file_path):
                return None, f"[{file_type}]: Отсутствует файл '{file_path}'."
            _, ext = os.path.splitext(file_path)
            match ext:
                case ".cb":
                    inst = CatBoostRegressor()
                    inst.load_model(file_path)
                case ".pkl":
                    with open(file_path, "rb") as fd:
                        inst = joblib.load(fd)
                case ".json":
                    with open(file_path, "r") as fd:
                        inst = json.load(fd)[file_type]
                case ".yaml":
                    with open(file_path, "r") as fd:
                        inst = yaml.safe_load(fd)
                case other:
                    return None, f"Неизвестное расширение файла '{file_path}' - '{other}' "
            return inst, f"Файл '{file_path}' загружен."
        except Exception as e: 
            return None, f"[{file_type}]: Не удалось загрузить файл '{file_path}': {e}."

    def load(self, model_dir: str | None) -> dict:
        """
        Функция загрузки модели и входных параметров.

        Parameters
        ----------

        model_dir : строка, по умолчанию None 
            При значении None перезагружается последняя использованная модель.
            
            > **Важно!!!** Необходимо контролировать корректное наименование всех файлов 
            > модели и их содержимое. В случае проблем с загрузкой отдельных файлов 
            > будет использоваться последняя удачно загруженная модель.
        """
        if model_dir is None:
            if self.model_dir is None:
                return self.get_status()
            else:
                model_dir = self.model_dir
        new_files = {
            "model": os.path.join(model_dir, "model.cb"), 
            "outl_proc": os.path.join(model_dir, "outlier_params.yaml"),
            "feat_proc": os.path.join(model_dir, "feat_processor.pkl"),
            "columns": os.path.join(model_dir, "input_example.json")}
        file_stat = {}
        file_inst = {}
        load_status = True 
        for file_type in new_files:
            # Загрузка файла и получение статуса
            file_inst[file_type], file_stat[file_type] = self.__load_file(
                file_type, new_files[file_type])
            if file_inst[file_type] is None:
                load_status = False
        
        # если все файлы загружены, готовим пайплайн модели
        if load_status:
            # Добавление функции предварительной обработки столбцов датасета, как при обработке выбросов
            self.model = Pipeline([
                ("outl_proc", FunctionTransformer(
                    columns_selector,
                    kw_args={
                        "drop_cols": file_inst["outl_proc"]["drop_cols"] if "drop_cols" in file_inst["outl_proc"] else None,
                        "change_col": file_inst["outl_proc"]["change_col"] if "change_col" in file_inst["outl_proc"] else None,
                    }
                )),
                # Добавление функции отбора признаков с трансформером для добавления новых признаков
                ("feat_proc", FunctionTransformer(
                    features_selector,
                    kw_args={
                        "feat_proc": file_inst["feat_proc"], 
                        "columns": file_inst["columns"]
                    }
                )),
                ("model", file_inst["model"])
            ])

        # Сохранение статуса последней загрузки
        for file_type in self.file_types:
            self.last_load[file_type] = file_stat[file_type]

        return self.get_status(status=200 if load_status else 400)
    
    def get_status(self, status: int = None) -> dict:
        """
        Возвращает статус загрузки модели и параметров.
        
        Parameters
        -----------
        last_load : логический, по умолчанию `False`.
            Получение статуса последней загрузки файлов, вместо текущего статуса.
        """
        if self.model is None:
            status_code = 500
            status = self.last_load
        elif status is not None:
            status_code = status
            status = self.last_load
        else:
            status_code = 200
            status = "OK"
        return {
            "status_code": status_code,
            "detail": status
        }

    def __save_result(self, result):
        if self.results.shape[0] == self.result_count:
            self.results.shift(1)
            self.results[0] = result
        if self.results.shape[0] > 0:
            self.results = pd.concat([pd.Series(result), self.results], ignore_index=True)
        else:
            self.results = pd.Series(result)

    def handle(self, params: dict):
        result = self.get_status()
        if result["status_code"] == 200:
            result["detail"] = self.model.predict(pd.DataFrame(params, index=[0]))
            self.__save_result(result=result["detail"])
        return result
        
    def get_agg(self, func="median") -> float:
        """Применяет к списку результатов аггрегирующую функцию pandas."""
        return self.results.aggregate(func)
    
    def get_quant(self, q: float) -> float:
        """Возвращает указанный квантиль из списка результатов."""
        return self.results.quantile(q)
