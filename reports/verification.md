# Проверка лабораторной №3

Проверена сеть по мотивам LeNet: ядра 5×5, Tanh, AvgPool, SGD с momentum и Нестеровым. Обучение выполнено **17 ноября 2026 года**. Веса и результаты находятся в `results/lenet_sgd/`.

## Окружение

Windows 11, Python 3.13.7, PyTorch 2.9.1+cpu, Torchvision 0.24.1+cpu, NumPy 2.2.6. Обучение выполнено на CPU с четырьмя потоками. Полные параметры сохранены в [config.json](../results/lenet_sgd/config.json), зависимости — в [requirements-lock-cpu.txt](../requirements-lock-cpu.txt).

## Проверки кода

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip check
```

Результат: **17 passed in 18.88s**. Конфликтов зависимостей нет: `No broken requirements found.`

Проверены:

- Разбиение MNIST: 50000/10000, пропорции классов, отсутствие пересечений и повторов, сохранение индексов и контрольной суммы меток.
- Выход сети `[N,10]`, конечные оценки и изменение весов после шага SGD.
- Загрузка весов в режиме `eval`, точное совпадение ответов до и после сохранения, отклонение несовместимой архитектуры.
- Цикл обучения, выбор лучшей эпохи по validation loss и работа StepLR: пятая эпоха использует вдвое меньший шаг.
- Обработка рисунков: размеры, нормализация, цвет фона, центрирование, прозрачность и пустой холст.
- CSV-разметка, уникальность рисунков, минимум пять примеров на класс, недопустимые пути и изображения без меток.
- Accuracy, macro F1 и усреднение loss по всем изображениям, включая неполный пакет.
- Оценка собственного набора: режимы `independent` и `repeated`, сохранение меток, контрольных сумм и результатов.
- Распознавание одного файла, десять вероятностей с суммой 1 и сохранение подготовленного изображения.

Изображения в автоматических тестах — временные синтетические примеры для проверки кода. Они не входят в набор рисунков пользователя.

## Обучение и повторная загрузка

MNIST разделён на 50000 обучающих и 10000 валидационных изображений с `seed=42`. Собственные рисунки в обучении не используются.

```powershell
.\.venv\Scripts\python.exe -m digit_classifier train --epochs 12 --batch-size 128 --lr 0.03 --seed 42 --threads 4 --device cpu --data-dir data --split splits/mnist_seed42.json --output results/lenet_sgd
.\.venv\Scripts\python.exe -m digit_classifier evaluate --checkpoint results/lenet_sgd/best.pt --dataset validation --data-dir data --device cpu --output results/lenet_sgd/validation_recheck
```

Архитектура содержит **61706 параметров**. SGD использует `momentum=0.9`, `nesterov=True`, `weight_decay=0.0001`. StepLR уменьшает learning rate после каждых четырёх эпох: `0.03 → 0.015 → 0.0075`. Dropout и аугментации не применяются.

Обучение завершено за **182,27 с**, лучшая эпоха — **10**. На 10000 валидационных изображений получены **9887 верных ответов**, accuracy **98,87%**, macro F1 **0,98861**, loss **0,03653**.

Отдельная загрузка `best.pt` воспроизвела метрики, матрицу ошибок и все предсказания. Веса включены в Git и доступны по ссылке [best.pt](../results/lenet_sgd/best.pt).

SHA-256 опубликованного файла:

```text
18366c0d9f141759b02588831c0e700a3e4d7ef7d1be0820ea4dc9abbeecf53d
```

## Повторная оценка рисунков

Проверены все 50 рисунков пользователя из `custom_test/`, по 5 на класс. Проверки читаемости, непустоты, уникальности и разметки прошли.

```powershell
.\.venv\Scripts\python.exe -m digit_classifier evaluate --checkpoint results/lenet_sgd/best.pt --dataset custom --custom-dir custom_test --custom-role repeated --device cpu --output results/lenet_sgd/custom
```

Результат: **46/50**, accuracy **92%**, macro F1 **0,91793**, loss **0,34249**. Сохранены [метрики](../results/lenet_sgd/custom/metrics.json), [предсказания](../results/lenet_sgd/custom/predictions.json) и [конфигурация с SHA-256 рисунков](../results/lenet_sgd/custom/evaluation_config.json).

Набор служит для повторных проверок. Результат на нём не считается независимой итоговой оценкой. Веса выбраны по валидации MNIST; для независимой оценки нужен отдельный набор рисунков.

## Ограничения

Графики обучения, матрицы ошибок и примеры ошибок просмотрены. Команда `predict` проверена на исходном рисунке с опубликованными весами. CUDA в этом окружении не проверялась; точное воспроизведение чисел зависит от оборудования и библиотек. Для окна рисования нужен обычный PowerShell с доступом к Tcl/Tk.
