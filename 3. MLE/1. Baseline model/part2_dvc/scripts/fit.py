import yaml
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import CatBoostEncoder
from catboost import CatBoostRegressor


def fit_model():
    # Загрузка параметров
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
    
    # Загрузка данных
    data = pd.read_csv('data/initial_data.csv')

    # Подготовка модели
    target = params["target"]
    bin_features = data.select_dtypes(include="boolean").columns.to_list()
    cat_features = data.select_dtypes(include="object").columns.to_list()
    #num_features = [x for x in data.columns if x not in [target, *bin_features, *cat_features]]
    # Пайплайн
    pipe = Pipeline([
        # Препроцессор
        ("preprocessor", ColumnTransformer(
            [
                ("bin", OneHotEncoder(
                    drop=params["one_hot_drop"], sparse_output=False
                ), bin_features),
                ("cat", CatBoostEncoder(return_df=False), cat_features)
                #("num", StandardScaler(), num_features)
            ], remainder="passthrough",
            verbose_feature_names_out=False
        )),
        # Модель
        ("model", CatBoostRegressor(
            iterations=params["iterations"], 
            learning_rate=params["learning_rate"], 
            depth=params["depth"],
            verbose=0, random_seed=params["random_state"])
        )
    ])

    # Обучение модели
    pipe.fit(data.drop(columns=target), data[target])

    # Сохранение модели
    os.makedirs('models', exist_ok=True) 
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipe, fd)

if __name__ == '__main__':
    fit_model() 