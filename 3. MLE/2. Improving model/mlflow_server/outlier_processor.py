import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Функция для обработки выбросов, подготовленная при EDA
def is_outlier(
    df: pd.DataFrame, target: str, divider: str | None = None,  out_val: int | float | None = None,
    by: list | None = None, by_round: dict | None = None, min_val: int | float | None = None, 
    max_val: int | float | None = None, threshold: float | None = 1.5, verbose = False,
) -> pd.Series:
    """
    Функция производит поиск выбросов посредством метода IQR для целевого признака 
    или его относительное значение к другому признаку, в том числе в разрезе других 
    признаков исходного датасета. Возвращает pandas Series с логическими значениями 
    True/False для каждой строки исходного датасета. 

    Параметры:
    ----------

    df : DataFrame библиотеки pandas 
        Исходный датасет, содержащий целевой признак, а также другие признаки, 
        необходимые для обработки выбросов.

    target : строка
        Целевой признак, для которого осуществляется поиск выбросов.

    divider : строка или None, по умолчанию None
        Признак, относительно которого рассчитывается целевой - target / divider, 
        если указано None, то target используется без изменений.

    out_val : числовой или None, по умолчанию None
        Указывается явно значение выброса, чтобы исключить из рассчета квантилей, 
        например, значение 0.
    
    by : список или None, по умолчанию None 
        Применяется при необходимости рассчета выбросов в разрезе групп признаков.

    by_round : словарь или None, по умолчанию None
        Применяется для округления значений признаков в списке by до определенного 
        количества знаков после запятой. Словарь вида {"col_name": 2}.
    
    min_val : числовой или None, по умолчанию None
        Применяется для ограничения минимального значения для признания выбросом, 
        если установлено, то все значения ниже min_val признаются выбросами.
    
    max_val : числовой или None, по умолчанию None
        Применяется для ограничения максимального значения для признания выбросом, 
        если установлено, то все значения выше max_val признаются выбросами.

    threshold : с плавающей точкой или None, по умолчанию 1.5
        Коэффициент IQR, за пределами которого значение признается выбросом. Если 
        указано None, то обязательно нужно указать min_val и max_val, функция не 
        будет рассчитывать IQR и определит выбросы по крайним значениям.

    verbose : логический, по умолчанию False
        Определяет вывод логов.
    """
    df = df.copy()
    if divider is not None:
        df[target] = df[target] / df[divider]
    if by is None:
        if verbose: print("Выбросы без группировки")
        if threshold is not None:
            Q1 = df[target].quantile(.25) if out_val is None else df.query(f"{target} != {out_val}").quantile(.25)
            Q3 = df[target].quantile(.75) if out_val is None else df.query(f"{target} != {out_val}").quantile(.75)
            min_val = max(Q1 - threshold * (Q3 - Q1), min_val) if min_val else Q1 - threshold * (Q3 - Q1)
            max_val = min(threshold * (Q3 - Q1) + Q3, max_val) if max_val else Q3 + threshold * (Q3 - Q1)
        else:
            if (min_val is None) or (max_val is None):
                raise ValueError("Если не используется аргумент 'threshold' необходимо обязательно указать 'min_value' и 'max_value'")
        if verbose: print("Минимальное значения для признания выбросом:", min_val)
        if verbose: print("Максимальное значения для признания выбросом:", max_val)
    else:
        if verbose: print(f"Выбросы с группировкой по {by}")
        # Округляем значения столбцов
        if by_round is not None:
            for col, val in by_round.items():
                df[col] = df[col].round(val)
        # Получение межквартильного отклонения для каждой группы
        vals = df.groupby(
            by=by)[target].quantile([.25, .75]).to_frame().unstack().droplevel(0, axis=1).apply(
                lambda x:
                pd.Series({
                    "min_val": x[0.25] - threshold * (x[0.75] - x[0.25]),
                    "max_val": x[0.75] + threshold * (x[0.75] - x[0.25])
                }), axis=1
        ) if out_val is None else df.query(f"{target} != {out_val}").groupby(
            by=by)[target].quantile([.25, .75]).to_frame().unstack().droplevel(0, axis=1).apply(
                lambda x:
                pd.Series({
                    "min_val": x[0.25] - threshold * (x[0.75] - x[0.25]),
                    "max_val": x[0.75] + threshold * (x[0.75] - x[0.25])
                }), axis=1
        )
        # Добавление ограничений минимального и максимального значения
        if min_val is not None: vals.loc[vals["min_val"] < min_val, "min_val"] = min_val
        if max_val is not None: vals.loc[vals["max_val"] > max_val, "max_val"] = max_val
        if verbose:
            print("Значения для признания выбросом:")
            print(vals.T)
        # Определение ограничений каждой записи датасета
        vals = df.join(vals, on=by)[["min_val", "max_val"]]
        min_val = vals["min_val"]
        max_val = vals["max_val"]
    return (df[target] < min_val) | (df[target] > max_val) | (df[target] == out_val)

# Функция для рассчета медианы
def get_agg_values(
    df: pd.DataFrame, target: str, divider: str | None = None,
    by: list | None = None, round_n: int | None = 2, func: str = "median"
) -> pd.Series:
    """
    Функция возвращает аггрегированные значения целевого признака или его 
    относительное значение к другому признаку в датасете в разрезе других 
    столбцов исходного датасета. Возвращает Series для каждой строки
    исходного датасета с аггрегированными значениями целевого признака.

    Параметры:
    ---------

    df : DataFrame библиотеки pandas
        Исходный датасет, содержащий целевой признак, а также другие 
        признаки, применяемые для рассчета аггрегированных значений.

    target : строка
        Целевой признак, по которому производится аггрегация.

    divider : строка или None, по умолчанию None
        Признак, относительно которого рассчитывается целевой target / divider, 
        если не указан, то используется target.

    by : список или None, по умолчанию None
        Список признаков для группировки, если не указан - группировка 
        не производится (аналог `df[target].transform(func)`).

    round_n : целочисленной или None, по умолчанию 2
        Количество знаков после запятой для округления, None - без округления.

    func : строка, функция, либо список, по умолчанию 'median'
        Аггрегирующая функция pandas.

    """
    result = df[target].copy()
    if by is None:
        result = result.transform(func=func)
    else:
        if divider is not None:
            result = result / df[divider]
        result = df[by].join(result.rename("result"))
        result = result.groupby(by).transform(func=func)["result"]
        if round_n is not None:
            result = round(result * df[divider], round_n)
    return result



# Трансформер для проведения предобработки выбросов ------------------------------------------------
class OutlierProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_params=None, dropna=True, drop_duplicates=True, verbose=False):
        self.outlier_params = outlier_params
        self.dropna = dropna
        self.drop_duplicates=drop_duplicates
        self.verbose = verbose
    
    # Добавление ранее объявленных методов как статические
    is_outlier = staticmethod(is_outlier)
    get_agg_values = staticmethod(get_agg_values)

    # Метод fit не производит никаких вычислений
    def fit(self, X, y=None):
        # Хорошей практикой было бы добавить в этот метод поиск и сохранение всех границ значений,
        # а в методе transform проводить сравнение и обработку значений для определения выбросов,
        # но это бы сильно поменяло логику ранее подготовленной мной функции для DAG.
        # Поэтому не стал сильно переделывать, как обычно отстаю.
        return self
    
    # Метод transform производит обработку выбросов
    def transform(self, X, y=None):
        # Работа с копией полученных данных
        result = X.copy()
        if self.verbose: print("Размер датасета:", result.shape)
        # Обработка признаков
        if self.outlier_params is not None:
            # Удаление неиспользуемых признаков
            if "drop_cols" in self.outlier_params:
                result.drop(columns=self.outlier_params["drop_cols"], inplace=True)
                if self.verbose: 
                    print("Удалены признаки", self.outlier_params["drop_cols"])
            # Удаление дубликатов перед обработкой выбросов
            if self.drop_duplicates: 
                if self.verbose: print("Количество дубликатов:", result.duplicated().sum())
                result.drop_duplicates(keep="first", inplace=True)
                if self.verbose: print("Размер датасета после удаления дубликатов:", result.shape)
            # Обработка выбросов
            if "out_cols" in self.outlier_params:
                for col in self.outlier_params["out_cols"]:
                    if self.verbose: print("Обработка признака", col["name"])
                    result.loc[self.is_outlier(
                        result, target=col["name"], 
                        divider=col["divider"] if "divider" in col else None,
                        out_val=col["out_val"] if "out_val" in col else None,
                        by=col["by"] if "by" in col else None, 
                        by_round=col["by_round"] if "by_round" in col else None,
                        min_val=col["min_val"] if "min_val" in col else None,
                        max_val=col["max_val"] if "max_val" in col else None,
                        threshold=col["threshold"] if "threshold" in col else None,
                        verbose=self.verbose
                    ), col["name"]] = np.nan
                    if self.verbose: 
                        print(f"Исключены выбросы в признаке {col['name']}: {result[col['name']].isna().sum()}")
            # Заполнение пропусков медианой
            if "fill_cols" in self.outlier_params:
                for col in self.outlier_params["fill_cols"]:
                    if self.verbose: print("Заполнение пропусков признака", col["name"])
                    outs = result[col["name"]].isna()
                    if self.verbose: print("Количество пропусков:", outs.sum())
                    result.loc[outs, col["name"]] = self.get_agg_values(
                        result, target=col["name"],
                        divider=col["divider"] if "divider" in col else None,
                        by=col["by"] if "by" in col else None,
                        round_n=col["round"] if "round" in col else None,
                        func=col["func"] if "func" in col else "median",
                    )[outs]
                    if self.verbose: print("Пропусков осталось:", result[col["name"]].isna().sum())
            if "change_col" in self.outlier_params:
                for col in self.outlier_params["change_col"]:
                    if "astype" in col:
                        result[col["name"]] = result[col["name"]].astype(col["astype"])
                    if "rename" in col:
                        result.rename(columns={col["name"]: col["rename"]}, inplace=True)
        # Удаление пропусков, в том числе выбросов, которым присвоено np.nan
        if self.dropna:
            if self.verbose: 
                print("Количесво пропусков:")
                print(result.isna().sum())
            result.dropna(inplace=True)
            if self.verbose: print("Размер датасета после удаления пропусков:", result.shape)

        # Удаление дубликатов после обработки выбросов
        if self.drop_duplicates: 
            if self.verbose: print("Количество дубликатов:", result.duplicated().sum())
            result.drop_duplicates(keep="first", inplace=True)
            if self.verbose: print("Размер датасета после удаления дубликатов:", result.shape)
        # Возврат результатов
        return result