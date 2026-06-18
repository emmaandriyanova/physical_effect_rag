"""
Разбиение записей базы FEText на чанки трёх типов: полная запись (record),
текстовое описание (description) и структурная тройка вход-объект-выход
(triple) — для последующей индексации в LanceDB.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import pandas as pd
from pathlib import Path
import json
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def safe_str(value) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    if value is None:
        return ""
    return str(value).strip()


def create_chunks(effect: dict) -> list[dict]:
    chunks = []

    effect_id = safe_str(effect.get("id", ""))
    effect_name = safe_str(effect.get("name", ""))
    description = safe_str(effect.get("description", ""))
    canonical_description = safe_str(effect.get("canonical_description", ""))
    description_hash = safe_str(effect.get("description_hash", ""))
    input_params = safe_str(effect.get("input_params", ""))
    output_params = safe_str(effect.get("output_params", ""))
    effect_object = safe_str(effect.get("object", ""))
    application = safe_str(effect.get("application", ""))
    literature = safe_str(effect.get("literature", ""))

    if description:
        chunks.append({
            "chunk_id": f"{effect_id}_desc",
            "effect_id": effect_id,
            "effect_name": effect_name,
            "chunk_type": "description",
            "text": f"Эффект: {effect_name}. Описание: {description}",
            "description": description,
            "canonical_description": canonical_description,
            "description_hash": description_hash,
            "input_params": input_params,
            "object": effect_object,
            "output_params": output_params
        })

    if input_params:
        chunks.append({
            "chunk_id": f"{effect_id}_input",
            "effect_id": effect_id,
            "effect_name": effect_name,
            "chunk_type": "input",
            "text": f"Эффект: {effect_name}. Входное воздействие: {input_params}",
            "description": description,
            "canonical_description": canonical_description,
            "description_hash": description_hash,
            "input_params": input_params,
            "object": effect_object,
            "output_params": output_params
        })

    if effect_object:
        chunks.append({
            "chunk_id": f"{effect_id}_object",
            "effect_id": effect_id,
            "effect_name": effect_name,
            "chunk_type": "object",
            "text": f"Эффект: {effect_name}. Объект воздействия: {effect_object}",
            "description": description,
            "canonical_description": canonical_description,
            "description_hash": description_hash,
            "input_params": input_params,
            "object": effect_object,
            "output_params": output_params
        })

    if output_params:
        chunks.append({
            "chunk_id": f"{effect_id}_output",
            "effect_id": effect_id,
            "effect_name": effect_name,
            "chunk_type": "output",
            "text": f"Эффект: {effect_name}. Выходное воздействие: {output_params}",
            "description": description,
            "canonical_description": canonical_description,
            "description_hash": description_hash,
            "input_params": input_params,
            "object": effect_object,
            "output_params": output_params
        })

    chunks.append({
        "chunk_id": f"{effect_id}_triple",
        "effect_id": effect_id,
        "effect_name": effect_name,
        "chunk_type": "triple",
        "text": (
            f"Эффект: {effect_name}. "
            f"Вход: {input_params}. "
            f"Объект: {effect_object}. "
            f"Выход: {output_params}."
        ),
        "description": description,
        "canonical_description": canonical_description,
        "description_hash": description_hash,
        "input_params": input_params,
        "object": effect_object,
        "output_params": output_params
    })

    chunks.append({
        "chunk_id": f"{effect_id}_record",
        "effect_id": effect_id,
        "effect_name": effect_name,
        "chunk_type": "record",
        "text": (
            f"Эффект: {effect_name}\n"
            f"Описание: {description}\n"
            f"Вход: {input_params}\n"
            f"Объект: {effect_object}\n"
            f"Выход: {output_params}"
        ),
        "description": description,
        "canonical_description": canonical_description,
        "description_hash": description_hash,
        "input_params": input_params,
        "object": effect_object,
        "output_params": output_params
    })

    if application:
        chunks.append({
            "chunk_id": f"{effect_id}_app",
            "effect_id": effect_id,
            "effect_name": effect_name,
            "chunk_type": "application",
            "text": f"Эффект: {effect_name}. Применение: {application}",
            "description": description,
            "canonical_description": canonical_description,
            "description_hash": description_hash,
            "input_params": input_params,
            "object": effect_object,
            "output_params": output_params
        })

    if literature:
        chunks.append({
            "chunk_id": f"{effect_id}_lit",
            "effect_id": effect_id,
            "effect_name": effect_name,
            "chunk_type": "literature",
            "text": f"Эффект: {effect_name}. Литература: {literature}",
            "description": description,
            "canonical_description": canonical_description,
            "description_hash": description_hash,
            "input_params": input_params,
            "object": effect_object,
            "output_params": output_params
        })

    return chunks


def main():
    csv_path = Path(__file__).parent.parent / "data" / "parsed_effects.csv"
    df = pd.read_csv(csv_path)

    logging.info(f"загружено {len(df)} эффектов")

    all_chunks = []
    for _, row in df.iterrows():
        chunks = create_chunks(row.to_dict())
        all_chunks.extend(chunks)

    logging.info(f"создано {len(all_chunks)} чанков")

    chunks_path = Path(__file__).parent.parent / "data" / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logging.info(f"чанки сохранены в {chunks_path}")

    stats = {}
    for chunk in all_chunks:
        stats[chunk["chunk_type"]] = stats.get(chunk["chunk_type"], 0) + 1

    print("\nстатистика по типам чанков:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()