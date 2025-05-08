import os
import warnings
import yaml

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class BankProductTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, settings_file: str, random_state: int | None = None):
        if random_state is None:
            warnings.warn(
                "Рекомендуется установить random_state для воспроизводимости.",
                UserWarning)
        if os.path.exists(settings_file):
            with open(settings_file) as fd:
                settings = yaml.safe_load(fd)
        else:
            raise FileNotFoundError(
                f"Файл не найден: {settings_file}")
        # Общие признаки и значения
        self.target_cols = settings["target_cols"]
        self.date_col = settings["date_col"]
        self.user_col = settings["user_col"]
        self.country_col = settings["country_col"]
        self.residence_col = settings["residence_col"]
        self.province_col = settings["province_col"]
        self.country_code = settings["country_code"]
        self.start_date_col = settings["start_date_col"]
        self.seniority_col = settings["seniority_col"]
        self.rel_col = settings["rel_col"]
        self.end_date_col = settings["end_date_col"]
        self.age_col = settings["age_col"]
        # Данные для обработки пропусков
        self.missing_col = settings["missing_col"]
        self.ffill_cols = settings["ffill_cols"]
        self.bfill_cols = settings["bfill_cols"]
        self.val_na_cols = settings["val_na_cols"]
        self.depend_val_cols = settings["depend_val_cols"]
        self.mode_fill_cols = settings["mode_fill_cols"]
        self.na1to0_cols = settings["na1to0_cols"]
        self.val_fill_cols = settings["val_fill_cols"]
        self.del_val_bfill_cols = settings["del_val_bfill_cols"]
        # Данные для обработки аномалий
        self.last_val_cols = settings["last_val_cols"]
        self.del_other_val_cols = settings["del_other_val_cols"]
        self.mod2_cols = settings["mod2_cols"]
        self.fix_residence = settings["fix_residence"]
        self.fix_seniority = settings["fix_seniority"]
        self.fix_val_order = settings["fix_val_order"]
        self.fix_rel_col = settings["fix_rel_col"]
        self.fix_bin_cols = settings["fix_bin_cols"]
        self.fix_age_vals = settings["fix_age_vals"]
        self.fix_age_outs = settings["fix_age_outs"]
        self.age_model_age_agg = settings["age_model_age_agg"]
        self.age_model_prods_agg = settings["age_model_prods_agg"]
        self.age_model_ohe_cols = settings["age_model_ohe_cols"]
        self.age_model_num_cols = settings["age_model_num_cols"]
        self.age_model_std_thres = settings["age_model_std_thres"]
        # Столбцы, которые остаются в датасете (кроме целевых)
        self.output_cols = settings["output_cols"]

        self.age_ohe = OneHotEncoder(drop="first", sparse_output=False)
        self.age_model = RandomForestRegressor(random_state=random_state)
        
    def _check_is_dataframe(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
        
    def _fix_residence(self, data: pd.DataFrame):
        # Исправление значений по индикатору резиденции
        country_ids = data[self.country_col] == self.country_code
        data[self.residence_col] = country_ids
        # Исправление значений провинции
        data.loc[
            ~country_ids, self.province_col
        ] = self.val_fill_cols[self.province_col]
        # Исправление ранее заполненных значений 
        # для резидентов с пропусками
        mask = data[self.province_col] == self \
            .val_fill_cols[self.province_col]
        data.loc[country_ids & mask, self.province_col] = np.nan
        data.loc[
            country_ids, self.province_col
        ] = data.loc[country_ids].groupby(
            self.user_col)[self.province_col].ffill()
        data.loc[
            country_ids, self.province_col
        ] = data.loc[country_ids].groupby(
            self.user_col)[self.province_col].bfill()
        data[self.province_col] = data[self.province_col] \
            .fillna(self.val_fill_cols[self.province_col])

        return data
    
    def _month_diff(self, data):
        return (data[self.date_col].dt.to_period("M").astype(int) - 
                data[self.start_date_col].dt.to_period("M").astype(int))

    def _check_month(self, data):
        return data[self.seniority_col] == self._month_diff(data)
    
    def _fix_double_dates(self, data):
        # Получение клиентов, у которых дата регистрации 
        # больше даты первой выгрузки
        mask = data[self.user_col].map((
            data.groupby(self.user_col)[self.date_col].min() + 
            pd.offsets.MonthEnd() -
            data.groupby(self.user_col)[self.start_date_col].max()
        ).dt.days < 0)
        # Установка меньшей даты
        if mask.sum() > 0:
            data.loc[mask, self.start_date_col] = data[mask] \
                .groupby(self.user_col)[self.start_date_col].transform("min")

        # Получение клиентов с 2 датами регистрации
        mask = data[self.user_col].map(
            data.groupby(self.user_col)[self.start_date_col].nunique() > 1)
        # Удаление клиентов без совпадений
        mask = mask & data[mask] \
            .groupby(self.user_col)[self.user_col].transform(
                lambda x: self._check_month(data[
                    data[self.user_col].isin(x.unique())
                ]).any())
        # Установка совпадающей даты
        if mask.sum() > 0:
            data.loc[mask, self.start_date_col] = data.loc[
                mask, self.user_col].map(
                    data.loc[mask].groupby(self.user_col).apply(
                        lambda x: next(iter(
                            x.loc[self._check_month(x), 
                                  self.start_date_col].tail(1)), 
                            np.nan), include_groups=False)).dropna()

        # Получение оставшихся клиентов с 2-мя совпадениями
        mask = data[self.user_col].map(
            data.groupby(self.user_col)[self.start_date_col].nunique() > 1)
        # Замена оставшихся значений на последнее
        if mask.sum() > 0:
            data.loc[mask, self.start_date_col] = data[mask] \
                .groupby(self.user_col)[self.start_date_col].transform("last")
        return data

    @staticmethod
    def _check_order(ser: pd.Series):
        res = np.full(len(ser), True)
        cur = len(res)
        for i in range(len(res)-1, -1, -1):
            if ser.iloc[i] <= cur:
                res[i] = False
                cur = ser.iloc[i]
        return pd.Series(res, index=ser.index)

    @staticmethod
    def _fix_age(group):
        out = group.to_numpy()
        for i in range(len(out)-2, -1, -1):
            out[i] = out[i] if out[i] == out[i + 1] - 1 else out[i + 1]
        return pd.Series(out, index=group.index)
    
    def _get_age_model_data(self, data):
        # Целевой признак для модели предсказания возраста
        y_age = data.groupby(self.user_col)[self.age_col] \
            .agg(self.age_model_age_agg)
        # Получение таргетов как признаков для модели
        X_age = data.groupby(self.user_col)[self.target_cols] \
            .agg(self.age_model_prods_agg)
        # Добавление числовых признаков для модели
        X_age[list(self.age_model_num_cols.keys())] = data.groupby(
            self.user_col)[list(self.age_model_num_cols.keys())] \
                .agg(self.age_model_num_cols)
        # Добавление категориальных признаков для модели
        X_age[self.age_ohe.get_feature_names_out()] = pd.DataFrame(
            self.age_ohe.fit_transform(
                data.groupby(self.user_col)[list(
                    self.age_model_ohe_cols.keys()
                )].agg(self.age_model_ohe_cols)),
            columns=self.age_ohe.get_feature_names_out(),
            index=X_age.index)
        
        return X_age, y_age
    
    def _fix_age_outliers(self, data):
        # Получение данных и обучение модели
        X_age, y_age = self._get_age_model_data(data)
        self.age_model.fit(X_age, y_age)
        
        # Получение остатков
        res = self.age_model.predict(X_age)
        res = y_age - res
        
        # Получение границ для выбросов
        lo_th = res.mean() - (
            self.age_model_std_thres["lower"] * res.std())
        up_th = res.mean() + (
            self.age_model_std_thres["upper"] * res.std())
        outs = (res < lo_th) | (res > up_th)
        
        # Исключение остатков в границах и округление
        outs = res.astype(int) * outs
        
        # Изменение выбросов по возрасту
        data.loc[:, self.age_col] = data[self.age_col] - data[
            self.user_col].map(outs)
        
        return data
    
    def _fix_missing(self, data: pd.DataFrame):
        # Удаление пропусков по столбцу fecha_alta
        data = data[data[self.missing_col].notna()].copy()

        # Удаление одиночных записей
        mask = data[self.user_col].map(
            data.groupby(self.user_col)[self.user_col] \
                .count() > 1)
        data = data[mask].copy()
        
        # Заполнение пропусков по целевым столбцам
        data[self.target_cols] = data[self.target_cols] \
            .fillna(0).astype(bool)
        
        # Замена значений при пропусках во 2-й с 1-й строкой
        for col in self.na1to0_cols:
            # получаем маску 2-й строки
            mask = data.groupby(self.user_col).cumcount() == 1
            # если они содержат пропуски
            mask = mask & data[col].isna()
            # добавляем сдвиг на одну строку назад
            mask = mask | mask.shift(-1)
            # получаем обратно отсортированные значения первых 2-х 
            # строк по группе
            data.loc[mask, col] = data[mask].groupby(self.user_col)[col] \
                .transform(lambda x: pd.Series(
                    list(x)[::-1], index=x.index))
        
        # Заполнение пропусков методом ffill
        data[self.ffill_cols] = data.groupby(
            self.user_col)[self.ffill_cols].ffill()
        
        # Заполнение пропусков методом bfill
        data[self.bfill_cols] = data.groupby(
            self.user_col)[self.bfill_cols].bfill()
        
        # Исключение некорректных значений
        for col, val in self.val_na_cols.items():
            mask = ~data[col].isin(val)
            data.loc[mask, col] = np.nan

        # Заполнение зависимых признаков
        for col, dep in self.depend_val_cols.items():
            for dep_col, val_map in dep.items():
                data[col] = data[col].fillna(
                    data[dep_col].map(val_map))
        
        # Заполнение пропусков модой
        for col in self.mode_fill_cols:
            data[col] = data[col].fillna(data[col].mode().iloc[0])
        
        # Заполнение пропусков значением
        for col, val in self.val_fill_cols.items():
            data[col] = data[col].fillna(val)
        
        # Удаление значений с заполнением bfill
        for col, val in self.del_val_bfill_cols.items():
            data[col] = data[col].replace(val, np.nan)
            data[col] = data[col].bfill()

        return data

    def _fix_anomalies(self, data):
        # Замена значений на последние
        for col in self.last_val_cols:
            data[col] = data.groupby(self.user_col)[col] \
                .transform("last")
            
        # Удаление строк с неизвестным значением
        for col, val in self.del_other_val_cols.items():
            data = data[data[col].isin(val)].copy()
        
        # Остаток от деления на 2
        for col in self.mod2_cols:
            data[col] = pd.to_numeric(data[col]) % 2

        # Исправление данных о резиденции
        if self.fix_residence:
            data = self._fix_residence(data)
        
        # Исправление данных о стаже
        if self.fix_seniority:
            data = self._fix_double_dates(data)
            data[self.seniority_col] = self._month_diff(data)
            # Исправление расхождений в 1 месяц
            mask = data[self.user_col].map(
                data.groupby(self.user_col)[self.seniority_col].agg(
                    lambda x: x.iloc[0] == 1))
            data.loc[mask, self.seniority_col] = data.loc[
                mask, self.seniority_col] - 1
        
        # Исправление некорректного порядка значений
        for col, val in self.fix_val_order.items():
            # Получение словаря с корректным порядком
            order_map = {v: i for i, v in enumerate(val)}
            # Получение маски клиентов с 2 и более значениями
            mask = data[self.user_col].map(
                data.groupby(self.user_col)[col].nunique() > 1)
            # Получение датасета с порядковыми номерами значений
            col_idx = data.loc[mask, [self.user_col]]
            col_idx[col] = data.loc[mask, col].map(order_map)
            # Получение маски по всем значениям
            mask = col_idx.groupby(self.user_col)[col].transform(
                self._check_order).reindex(mask.index) == True
            data.loc[mask, col] = np.nan
            data[col] = data.groupby(self.user_col)[col].bfill()

        # Исправление значений отношения с клиентом
        if self.fix_rel_col:
            mask = data[self.end_date_col].notna()
            if len(data.loc[mask, self.rel_col].unique()) > 1:
                raise ValueError(f"Имеются несколько значений {self.rel_col}"+
                                 " для ушедших клиентов")
            val = data.loc[~mask, self.rel_col].unique()
            if len(val) > 1:
                raise ValueError(f"Имеются несколько значений {self.rel_col}: "+
                                 f"{val} для активных клиентов")
            data.loc[mask, self.rel_col] = data[mask] \
                .groupby([self.user_col, self.end_date_col])[self.rel_col] \
                    .transform(
                    lambda x: x.where(x.index == x.index[-1], val[0]))
        
        # Изменение бинарных признаков
        for col, val in self.fix_bin_cols.items():
            data[col] = data[col].eq(val).astype(bool)

        # Исправление значений возраста (увеличение на 1)
        if self.fix_age_vals:
            data[self.age_col] = data.groupby(
                self.user_col)[self.age_col].transform(self._fix_age)
            
        # Обработка выбросов моделью
        if self.fix_age_outs:
            data = self._fix_age_outliers(data)

        return data
        
    def fit(self, X, y=None):
        self._check_is_dataframe(X)
        # При обучении ничего не делаем, так как для обучения модели 
        # определения выбросов по возрасту необходимо предварительно 
        # привести в порядок все данные, что производит transform
        return self

    def transform(self, X: pd.DataFrame):
        self._check_is_dataframe(X)

        # Сортировка значений по пользователю и дате
        result = X.sort_values(
            by=[self.user_col, self.date_col]).copy()
        
        # Заполнение пропусков с проверкой заполнения
        result = self._fix_missing(result)
        if result[self.output_cols].isna().sum().sum() > 0:
            cols = result[self.output_cols].isna().sum() \
                .loc[lambda x: x > 0].index
            raise ValueError(
                f"Не все пропуски устранены, проверьте столбцы: {cols}")
        
        # Исправление аномалий
        result = self._fix_anomalies(result)

        return result[self.output_cols + self.target_cols]