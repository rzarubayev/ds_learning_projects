
import pendulum
from airflow.decorators import task, dag
from steps.messages import send_telegram_success_message, send_telegram_failure_message

@dag(
    dag_id="flats_fstore",
    schedule="@once",
    start_date=pendulum.datetime(2024, 9, 14, tz="UTC"),
    on_success_callback=send_telegram_success_message,
    on_failure_callback=send_telegram_failure_message,
    tags=["flats_feature_store"]
)
def prepare_flats_fstore():
    import pandas as pd
    import os
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    TBL_DST = os.environ.get("TBL_DST")
    TBL_FLATS = os.environ.get("TBL_FLATS")
    TBL_BUILDINGS = os.environ.get("TBL_BUILDINGS")

    @task()
    def create_table():
        hook = PostgresHook("destination_db")
        conn = hook.get_sqlalchemy_engine()
        from sqlalchemy import MetaData
        metadata = MetaData()
        from sqlalchemy import Table, Column, Integer, Numeric, Float, Boolean, UniqueConstraint, inspect
        flats_fstore = Table(
            TBL_DST, metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("rooms", Integer),
            Column("total_area", Float),
            Column("kitchen_area", Float),
            Column("living_area", Float),
            Column("floor", Integer),
            Column("studio", Boolean),
            Column("is_apartment", Boolean),
            Column("building_type_int", Integer),
            Column("build_year", Integer),
            Column("latitude", Float),
            Column("longitude", Float),
            Column("ceiling_height", Float),
            Column("flats_count", Integer),
            Column("floors_total", Integer),
            Column("has_elevator", Boolean),
            Column("price", Numeric),
            UniqueConstraint('id', name='unique_flat_id_constraint')
        )
        if not inspect(conn).has_table(flats_fstore.name): 
            metadata.create_all(conn)
        conn.dispose()

    @task()
    def extract() -> pd.DataFrame:
        hook = PostgresHook("destination_db")
        sql_query = f"""
        SELECT 
            f.id, f.rooms, f.total_area, f.kitchen_area, f.living_area, 
            f.floor, f.studio, f.is_apartment, b.building_type_int, 
            b.build_year, b.latitude, b.longitude, b.ceiling_height, 
            b.flats_count, b.floors_total, b.has_elevator, f.price
        FROM {TBL_FLATS} AS f
        LEFT JOIN {TBL_BUILDINGS} AS b ON b.id = f.building_id
        """
        conn = hook.get_conn()
        data = pd.read_sql(sql_query, conn)
        conn.close()
        return data
        
    @task()
    def transform(data: pd.DataFrame) -> pd.DataFrame:
        return data

    @task()
    def load(data: pd.DataFrame):
        hook = PostgresHook("destination_db")
        hook.insert_rows(
            table=TBL_DST,
            replace=True,
            target_fields=data.columns.tolist(),
            replace_index=['id'],
            rows=data.values.tolist()
        )
    
    create_table()
    data = extract()
    transformed_data = transform(data)
    load(transformed_data)

prepare_flats_fstore()