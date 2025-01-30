import yaml
import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_model():
    # Загрузка параметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd) 
    
    # Загрузка данных и модели
    data = pd.read_csv('data/initial_data.csv')
    with open('models/fitted_model.pkl', 'rb') as fd:
        pipe = joblib.load(fd) 
    
    # Кросс-валидация 
    cv_strategy = StratifiedKFold(n_splits=params['n_splits'])
    cv_res = cross_validate(
        pipe,
        data.drop(columns=params["target"]),
        data[params["target"]],
        cv=cv_strategy,
        n_jobs=params['n_jobs'],
        scoring=params['metrics']
        )
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3) 
    
    # Сохранение результата кросс-валидации
    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as fd:
        json.dump(cv_res, fd, indent=4)

if __name__ == '__main__':
    evaluate_model() 