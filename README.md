# Fashion MNIST classification (MLOps pipeline)

Задача классификации одежды на наборе данных [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

[Репозиторий курса](https://github.com/girafe-ai/mlops)

## Запуск hw1

- Настройка окружения:
```bash
poetry install
```

- Обучение:
```bash
cd fashion_mnist
poetry run python train.py
```

- Инференс:
```bash
poetry run python infer.py
```

- Запуск `pre-commit`:
```bash
pre-commit install
pre-commit run --all-files
```

## Hw2

[Google Drive с данными для DVC](https://drive.google.com/drive/u/6/folders/1pseSLm5GJNShatTFvCr5DsU9K61fVr5R)

- Загрузка данных:
```bash
dvc pull
```
