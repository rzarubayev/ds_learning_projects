## Содержание директории flats_price

Для работы модели в директории должны находиться в обязательном порядке следующие файлы:

- `model.cb` - дамп модели CatBoost;
- `feat_processor.pkl` - дамп трансформера для добавления новых признаков;
- `input_example.json` - файл с примером входных данных для получения списка признаков по результатам их отбора.

## Модель удалена!

Был вынужден исключить дамп модели, размер которой составляет порядка ~80Mb, так как не получается отправить проект на проверку в Практикуме.