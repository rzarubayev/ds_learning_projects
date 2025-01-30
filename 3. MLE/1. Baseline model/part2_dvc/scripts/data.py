import os
import yaml
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

def create_connection():
    #load_dotenv()
    host = os.environ.get('DB_DESTINATION_HOST')
    port = os.environ.get('DB_DESTINATION_PORT')
    db = os.environ.get('DB_DESTINATION_NAME')
    username = os.environ.get('DB_DESTINATION_USER')
    password = os.environ.get('DB_DESTINATION_PASSWORD')
    print(f"postgresql://{username}:{password}@{host}:{port}/{db}")
    conn = create_engine(f"postgresql://{username}:{password}@{host}:{port}/{db}")
    return conn

def get_data():
    # Загрузка параметров
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
    
    # Получение данных из базы
    conn = create_connection()
    table = os.environ.get('TBL_DST_CLEAN')
    data = pd.read_sql(f"SELECT * FROM {table}", conn, index_col=params["index_col"])
    conn.dispose()
    # Сохранение результатов
    os.makedirs('data', exist_ok=True)
    data.to_csv("data/initial_data.csv", index=None)

if __name__ == '__main__':
    load_dotenv()
    get_data() 