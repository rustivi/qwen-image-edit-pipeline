'EOF'
import os
import json
import torch
from PIL import Image
from diffusers import QwenImageEditPipeline

INPUT_DIR = "images/original"
OUTPUT_DIR = "images/edited"
RESULTS_FILE = "results.json"
MODEL_NAME = "Qwen/Qwen-Image-Edit"

PROMPTS = {
    "img_001": "Add heavy snowstorm and blizzard to the mountain scene",
    "img_002": "Transform the forest to autumn with orange and red falling leaves",
    "img_003": "Change the forest lighting to a magical moonlit night scene",
    "img_004": "Add glowing mushrooms and fairy lights to the dark forest",
    "img_005": "Transform to a winter scene with snow covering all the branches",
    "img_006": "Add a dramatic tropical storm with dark clouds and heavy rain",
    "img_007": "Transform the misty forest into a cherry blossom spring scene",
    "img_008": "Change the sunset to a stunning aurora borealis northern lights",
    "img_009": "Add a dramatic lightning storm in the background sky",
    "img_010": "Transform the field to winter replace flowers with snow",
    "img_011": "Change the sky to a dramatic stormy sunset with purple clouds",
    "img_012": "Add northern lights aurora borealis reflecting in the lake",
    "img_013": "Transform the sunset to a full moon night scene over the sea",
    "img_014": "Transform to winter freeze the waterfall with ice and snow",
    "img_015": "Change the scene to a dramatic stormy evening with dark clouds",
    "img_016": "Transform the sunset to a dramatic aurora borealis night sky",
    "img_017": "Add heavy snowfall and transform the scene to a winter wonderland",
    "img_018": "Replace the red sunset with a full moon rising over the silhouette",
    "img_019": "Transform the scene to winter with snow covering the field and roof",
    "img_020": "Enhance the fog and add mysterious dark storm clouds to the sky",
    "img_021": "Change the flowers to vibrant purple lavender field in full bloom",
    "img_022": "Transform to a rainy moody scene with dark clouds and wet flowers",
    "img_023": "Change the sky to a dramatic sunset with deep red and purple colors",
    "img_024": "Transform the scene to autumn with orange and red trees on the hills",
    "img_025": "Add dramatic northern lights aurora borealis reflecting in the lake",
    "img_026": "Transform to winter cover mountains with snow and freeze the lake",
    "img_027": "Change the summer scene to a magical winter with snow on all trees",
    "img_028": "Replace the pastel sky with a dramatic stormy dark purple sky",
}


def load_pipeline() -> QwenImageEditPipeline:
    print(f"Загружаю модель {MODEL_NAME}...")
    pipeline = QwenImageEditPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")
    print("Модель загружена!\n")
    return pipeline


def process_image(pipeline, input_path: str, output_path: str, prompt: str) -> None:
    image = Image.open(input_path).convert("RGB")
    inputs = {
        "image": image,
        "prompt": prompt,
        "generator": torch.manual_seed(42),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 50,
    }
    with torch.inference_mode():
        output = pipeline(**inputs)
        output.images[0].save(output_path)


def build_record(idx: int, source_id: str, filename: str,
                 result_filename: str, prompt: str, success: bool,
                 error: str = None) -> dict:
    return {
        "id": idx,
        "source_image_id": source_id,
        "source_image_path": f"images/original/{filename}",
        "prompt": prompt,
        "result_image_id": os.path.splitext(result_filename)[0] if success else None,
        "result_image_path": f"images/edited/{result_filename}" if success else None,
        "status": "success" if success else f"error: {error}",
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipeline = load_pipeline()

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Найдено изображений: {len(image_files)}\n")
    results = []

    for idx, filename in enumerate(image_files, start=1):
        source_id = os.path.splitext(filename)[0]
        prompt = PROMPTS.get(source_id, "Transform the scene with dramatic lighting")
        input_path = os.path.join(INPUT_DIR, filename)
        result_filename = f"{source_id}_edited.jpg"
        output_path = os.path.join(OUTPUT_DIR, result_filename)

        print(f"[{idx}/{len(image_files)}] {filename}")
        print(f"  Промпт : {prompt}")

        try:
            process_image(pipeline, input_path, output_path, prompt)
            record = build_record(idx, source_id, filename, result_filename, prompt, success=True)
            print(f"  Статус : ✓ сохранено {result_filename}")
        except Exception as e:
            record = build_record(idx, source_id, filename, result_filename, prompt, success=False, error=str(e))
            print(f"  Статус : ✗ ошибка — {e}")

        results.append(record)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\nГотово: {success_count}/{len(results)} изображений обработано успешно")
    print(f"Результаты сохранены в {RESULTS_FILE}")


if __name__ == "__main__":
    main()
