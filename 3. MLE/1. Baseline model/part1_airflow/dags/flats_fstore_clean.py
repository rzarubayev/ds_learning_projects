import pendulum
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task, dag
from steps.messages import send_telegram_success_message, send_telegram_failure_message

@dag(
    dag_id="flats_fstore_clean",
    schedule="@once",
    start_date=pendulum.datetime(2024, 9, 14, tz="UTC"),
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message,
    tags=["flats_feature_store_clean"]
)
def prepare_flats_fstore_clean():
    import numpy as np
    import pandas as pd
    import os
    TBL_DST = os.environ.get("TBL_DST")
    TBL_DST_CLEAN = os.environ.get("TBL_DST_CLEAN")

    @task()
    def create_table():
        hook = PostgresHook("destination_db")
        conn = hook.get_sqlalchemy_engine()
        from sqlalchemy import MetaData
        metadata = MetaData()
        from sqlalchemy import Table, Column, Integer, Numeric, Float, Boolean, String, UniqueConstraint, inspect
        flats_fstore = Table(
            TBL_DST_CLEAN, metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),  
            Column("rooms", Float),
            Column("total_area", Float),
            Column("kitchen_area", Float),
            Column("living_area", Float),
            Column("floor", Integer),  
            Column("is_apartment", Boolean),
            Column("building_type", String),  
            Column("build_year", Integer),  
            Column("latitude", Float),
            Column("longitude", Float),
            Column("ceiling_height", Float),
            Column("flats_count", Integer),  
            Column("floors_total", Integer),  
            Column("has_elevator", Boolean),
            Column("price", Numeric),
            Column("first_floor", Boolean),
            Column("second_floor", Boolean),
            Column("last_floor", Boolean),
            UniqueConstraint('id', name='unique_flats_clean_id_constraint')
        )
        if not inspect(conn).has_table(flats_fstore.name): 
            metadata.create_all(conn)
        conn.dispose()

    @task()
    def extract() -> pd.DataFrame:
        hook = PostgresHook("destination_db")
        sql_query = f"SELECT * FROM {TBL_DST}" 
        conn = hook.get_conn()
        data = pd.read_sql(sql_query, conn)
        conn.close()
        return data
        
    @task()
    def transform(data: pd.DataFrame) -> pd.DataFrame:
        # Функция поиска выбросов
        def is_outlier(
            df: pd.DataFrame, target: str, by: list | None = None, divider: str | None = None, 
            min_val: int | None = None, max_val: int | None = None, threshold = 1.5, verbose = False,
        ) -> pd.Series:
            df = df.copy()
            if divider is not None:
                df[target] = df[target] / df[divider]
            if by is None:
                if verbose: print("Выбросы без группировки")
                Q1 = df[target].quantile(0.25)
                Q3 = df[target].quantile(0.75)
                min_val = max(Q1 - threshold * (Q3 - Q1), min_val) if min_val else Q1 - threshold * (Q3 - Q1)
                if verbose: print("Минимальное значения для признания выбросом:", min_val)
                max_val = min(threshold * (Q3 - Q1) + Q3, max_val) if max_val else Q3 + threshold * (Q3 - Q1)
                if verbose: print("Максимальное значения для признания выбросом:", max_val)
            else:
                if verbose: print(f"Выбросы с группировкой по {by}")
                # Получение межквартильного отклонения для каждой группы
                vals = df.groupby(
                    by=by)[target].quantile([.25, .75]).to_frame().unstack().droplevel(0, axis=1).apply(
                    lambda x:
                    pd.Series({
                        "min_val": x[0.25] - threshold * (x[0.75] - x[0.25]),
                        "max_val": x[0.75] + threshold * (x[0.75] - x[0.25])
                    }), axis=1
                )
                # Добавление ограничений минимального и максимального значения
                if min_val is not None: vals.loc[
                    vals["min_val"] < min_val, "min_val"
                ] = min_val
                if max_val is not None: vals.loc[
                    vals["max_val"] > max_val, "max_val"
                ] = max_val
                # Добавление ограничений к каждой записи датасета by
                vals = df.join(vals, on=by)[["min_val", "max_val"]]
                min_val = vals["min_val"]
                if verbose: print("Минимальное значение выброса по всем группам:",
                    min_val.min())
                max_val = vals["max_val"]
                if verbose: print("Максимальное значение выброса по всем группам:",
                    max_val.max())
            return (df[target] < min_val) | (df[target] > max_val)
        # --------------------------------------------------------------------------------------
        
        # Функция для рассчета медианы
        def get_median_values(
            data: pd.DataFrame, target: str, by: list | None = None, 
            divider: str | None = None, round_n = 2
        ) -> pd.Series:
            result = data[target].copy()
            if by is None:
                result = result.transform("median")
            else:
                if divider is not None:
                    result = result / data[divider]
                result = data[by].join(result.rename("result"))
                result = result.groupby(by).transform("median")["result"]
                if divider is not None:
                    result = round(result * data[divider], round_n)
            return result
        # -------------------------------------------------------------------------------------
        
        # Удаление дубликатов
        result = data.drop_duplicates(
            subset=data.drop(columns=["id"]).columns.to_list(),
            keep="first" # Это значение по умолчанию, поэтому не добавлял явно
        ).copy()
        # Удаление лишних признаков
        result.drop(columns="studio", inplace=True)

        # Список признаков с параметрами для поиска выбросов
        import yaml
        with open("dags/flats_fstore_clean.yaml", "r") as fd:
            params = yaml.safe_load(fd)

        # Сброс значений для выбросов
        for col in params["out_cols"]:
            result.loc[is_outlier(
                result, target=col["name"], by=col["by"], divider=col["divider"], 
                min_val=col["min_val"], max_val=col["max_val"], 
                threshold=col["threshold"], verbose=False
            ), col["name"]] = col["fill"] if col["fill"] else np.nan
        
        # Удаление пропусков, в том числе выбросов
        result = result.dropna()

        # Заполнение нулевых значений медианой
        for col in params["median_cols"]:
            zeros = result[col] == 0
            result.loc[zeros, col] = np.nan
            result.loc[zeros, col] = get_median_values(result, col, params["median_by_cols"], "total_area")[zeros]

        # Удаление некорректных значений
        result = result.query("(floor > 0) and (floor <= floors_total)").copy()
        result = result.query("(ceiling_height >= 2) and (ceiling_height <= 10)").copy()

        # Преобразование типа здания
        result["building_type_int"] = result["building_type_int"].astype("str")
        result.rename(columns={"building_type_int": "building_type"}, inplace=True)

        # Добавление новых признаков
        result["first_floor"] = result["floor"] == 1
        result["second_floor"] = result["floor"] == 2
        result["last_floor"] = result["floor"] == result["floors_total"]

        # Возврат результата
        return result

    @task()
    def load(data: pd.DataFrame):
        hook = PostgresHook("destination_db")
        hook.insert_rows(
            table=TBL_DST_CLEAN,
            replace=True,
            target_fields=data.columns.tolist(),
            replace_index=['id'],
            rows=data.values.tolist()
        )
    
    create_table()
    data = extract()
    data = transform(data)
    load(data)
prepare_flats_fstore_clean()