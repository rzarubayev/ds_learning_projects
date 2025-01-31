# Мои проекты обучения по специализации Data Science / Machine Learning

В данном репозитории представлены мои проекты по программе обучения "Специалист по Data Science" и "Инженер машинного обучения" в Яндекс Практикум. Не включал первые проекты по DS, которые проверялись автоматически, либо не представляют интереса в качестве портфолио.

---
# Специалист по Data Science
Файлы Jupyter Notebook для проектов расположены в директории `1. Data Science`. Содержимое директории представлено в таблице (со ссылками на файлы):
<table>
    <thead>
        <tr>
            <th>Проект</th>
            <th>Задача</th>
            <th>Технологии</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1. <a href="1.%20Data%20Science/01.%20Game%20sales.ipynb">Продажи игр</a></td>
            <td>анализ</td>
            <td>pandas, matplotlib, ttest</td>
        </tr>
        <tr>
            <td>2. <a href="1.%20Data%20Science/02.%20Milk%20farm.ipynb">Молочная ферма</a></td>
            <td>регрессия, классификация</td>
            <td>phik, sklearn linear models</td>
        </tr>
        <tr>
            <td>3. <a href="1.%20Data%20Science/03.%20Customer%20activity.ipynb">Покупательская активность</a></td>
            <td>классификация</td>
            <td>+ sklearn pipeline, logreg, kNN, SVM, Decision Tree, SHAP</td>
        </tr>
        <tr>
            <td>4. <a href="1.%20Data%20Science/04.%20HR%20analytics.ipynb">HR-аналитика</a></td>
            <td>регрессия, классификация</td>
            <td>+ imblearn</td>
        </tr>
        <tr>
            <td>5. <a href="1.%20Data%20Science/05.%20Oil%20location.ipynb">Выбор локации для скважины</a></td>
            <td>регрессия</td>
            <td>linreg, bootstrap</td>
        </tr>
        <tr>
            <td>6. <a href="1.%20Data%20Science/06.%20Car%20price.ipynb">Стоимость автомобилей</a></td>
            <td>регрессия</td>
            <td>Optuna, CatBoost, LightGBM, Decision Tree, linreg</td>
        </tr>
        <tr>
            <td>7. <a href="1.%20Data%20Science/07.%20Taxi%20orders.ipynb">Заказы такси</a></td>
            <td>временные ряды</td>
            <td>Statmodels, CatBoost, LightGBM, XGBoost</td>
        </tr>
        <tr>
            <td>8. <a href="1.%20Data%20Science/08.%20Toxic%20comments.ipynb">Токсичность текста</a></td>
            <td>классификация</td>
            <td>BERT, transformers, CatBoost, LightGBM, Random Forest</td>
        </tr>
        <tr>
            <td>9. <a href="1.%20Data%20Science/09.%20Computer%20vision.ipynb">Определение возраста по фото</a></td>
            <td>регрессия</td>
            <td>CNN, ResNet50</td>
        </tr>
        <tr>
            <td>10. <a href="1.%20Data%20Science/10.%20Customer%20churn.ipynb">Отток клиентов</a></td>
            <td>классификация</td>
            <td>Seaborn, Optuna, CatBoost, Random Forest, Ridge, SHAP</td>
        </tr>
    </tbody>
</table>

---
# SQL
Добавил проект по SQL из программы "Специалист по Data Science" в директорию `2. SQL`.
- Описание базы данных и ER-диаграмма в файле '[1.0. SQL project.pdf](2.%20SQL/1.0.%20SQL%20project.pdf)';
- Запросы по заданиям в проекте в файле '[1.1. SQL project.md](2.%20SQL/1.1.%20SQL%20project.md)'. 

> Стоит отметить, что задания в теоретической части были существенно сложнее, чем задания в самостоятельном проекте, в том числе оконные функции и применяемые в рамках оконных функции ранжирования, смещения и аггрегации. В рамках данной темы также было ознакомился с библиотекой PySpark.

Также добавил 10 заданий из дополнительной практики по SQL этой же программы для обучения:

1. Задания по базе данных Northwind, описание базы в файле '[2.0. Northwind description.md](2.%20SQL/2.0.%20Northwind%20description.md)', файлы с 5 последними заданиями в файле '[2.1. Northwind tasks.md](2.%20SQL/2.1.%20Northwind%20tasks.md)'.
2. Задания по базе данных AdventureWorks, описание базы в файле ['3.0. AdventureWorks description.md](2.%20SQL/3.0.%20AdventureWorks%20description.md)', файлы с 5 последними заданиями в файле '[3.1. AdventureWorks tasks.md](2.%20SQL/3.1.%20AdventureWoks%20tasks.md)'.

---
# Инженер машинного обучения
В настоящее время прохожу обучение по новой программе "Инженер машинного обучения" в Яндекс Практикуме, в директорию `3. MLE` выкладываю завершенные проекты по этой программе. 

<table>
    <thead>
        <tr>
            <th>Проект</th>
            <th>Задача</th>
            <th>Технологии</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1. <a href="3.%20MLE/1.%20Baseline%20model">Стоимость квартир (Базовая модель)</a></td>
            <td>регрессия</td>
            <td>airflow, DVC-pipeline, SQLAlchemy, Docker, S3</td>
        </tr>
        <tr>
            <td>2. <a href="3.%20MLE/2.%20Improving%20model">Стоимость квартир (Улучшение модели)</a></td>
            <td>регрессия</td>
            <td>ML-flow, Plotly, AutoFeat, MLxtend, Optuna</td>
        </tr>
        <tr>
            <td>3. <a href="3.%20MLE/3.%20Deploy%20model">Стоимость квартир (Деплой модели)</a></td>
            <td>регрессия</td>
            <td>FastAPI, Uvicorn, Docker compose, Prometheus, Grafana</td>
        </tr>
        <tr>
            <td>4. Рекомендация музыкальных треков</td>
            <td>рекомендательные системы</td>
            <td>(В процессе)</td>
        </tr>
    </tbody>
</table>
