# Fashion MNIST classification

Задача классификации одежды на наборе данных [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

[Репозиторий курса](https://github.com/girafe-ai/mlops)

## Запуск

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
poetry run python infet.py
```

- Запуск `pre-commit`:
```bash
pre-commit install
pre-commit run --all-files
```
