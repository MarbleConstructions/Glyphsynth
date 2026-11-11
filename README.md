# GlyphSort

Классификатор рукописных цифр 0–9 (28×28, grayscale) на PyTorch.

## Установка

    pip install -r requirements.txt

## Обучение

    python glyphsort.py train

Создаёт `artifacts/glyphsort.pt`.

## Предсказание

    python glyphsort.py predict --demo
    python glyphsort.py predict path/to/img.png

## Файлы

- `settings.py` — dataclass `Settings` + чтение `config.yaml`
- `synth.py`    — генерация синтетических цифр
- `network.py`  — модель `GlyphNet` (Conv + BatchNorm + FC)
- `engine.py`   — обучение, early stopping, загрузка чекпойнта
- `glyphsort.py`— CLI-точка входа

Конфиг читается из `config.yaml`; если файла нет — используются
дефолты из `settings.py`. Устройство выбирается автоматически
(CUDA → CPU).