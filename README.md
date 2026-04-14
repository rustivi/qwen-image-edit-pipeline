# Qwen Image Edit Pipeline

Пайплайн для автоматического редактирования изображений с помощью модели [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit).

## Описание

Скрипт обрабатывает 28 фотографий природы с уникальными промптами - 
меняет освещение, погоду, сезон и стиль изображений. 
Результаты сохраняются в `results.json`.

## Модель

- **Модель:** Qwen/Qwen-Image-Edit
- **Тип:** image-to-image
- **Инфраструктура:** NVIDIA A100 80GB

## Структура проекта
```plaintext
├── images/
│   ├── original/     # исходные изображения (28 фото)
│   └── edited/       # отредактированные изображения
├── results.json      # результаты обработки
├── run.py            # основной скрипт
└── README.md
```

## Формат results.json

```json
[
  {
    "id": 1,
    "source_image_id": "img_001",
    "source_image_path": "images/original/img_001.jpg",
    "prompt": "Add heavy snowstorm and blizzard to the mountain scene",
    "result_image_id": "img_001_edited",
    "result_image_path": "images/edited/img_001_edited.jpg",
    "status": "success"
  }
]
```

## Примеры промптов

| Фото | Промпт |
|------|--------|
| Горы Доломиты | Add heavy snowstorm and blizzard to the mountain scene |
| Лесная тропинка | Transform the forest to autumn with orange and red falling leaves |
| Водопад | Transform to winter — freeze the waterfall with ice and snow |
| Тропический пляж | Add a dramatic tropical storm with dark clouds and heavy rain |

## Установка и запуск

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate pillow diffusers
python3 run.py
```

## Требования

- Python 3.10+
- CUDA GPU с VRAM ≥ 80 ГБ
- ~60 ГБ свободного места на диске
