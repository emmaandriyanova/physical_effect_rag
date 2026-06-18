"""
Прогон базовой модели без RAG-контекста на тестовой выборке из 40 патентов
с сохранением результатов извлечения (вход, объект, выход) в CSV-файл
для последующего сравнения с RAG-системой.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import csv
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LM_STUDIO_URL, LM_STUDIO_MODEL_ID, EVAL_NO_RAG_RESULTS_PATH

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
INPUT_PATH = _DATA_DIR / "patents_40_texts.txt"
OUTPUT_PATH = EVAL_NO_RAG_RESULTS_PATH


def read_texts(file_path: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8").strip()
    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    return parts


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    brace_count = 0
    end = -1

    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if end == -1:
        return None

    candidate = text[start:end + 1].strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    candidate_fixed = candidate.replace("'", '"')

    try:
        parsed = json.loads(candidate_fixed)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return None


def clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_list_or_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        items = [clean_text(x) for x in value if clean_text(x)]
        return ". ".join(items)
    return clean_text(value)


def call_model(text: str) -> dict:
    system_prompt = (
        "Ты извлекаешь физический эффект из технического текста.\n"
        "Нужно вернуть строго один JSON-объект со следующими полями:\n"
        "- input_params\n"
        "- object\n"
        "- output_params\n\n"
        "ПРАВИЛА:\n"
        "- input_params: только входные воздействия\n"
        "- object: только объект воздействия\n"
        "- output_params: только выходные воздействия\n"
        "- не добавляй пояснений, комментариев и лишнего текста\n"
        "- отвечай только JSON\n"
    )

    user_prompt = (
        f"Текст:\n{text}\n\n"
        "Верни ответ в формате:\n"
        "{\n"
        '  "input_params": "...",\n'
        '  "object": "...",\n'
        '  "output_params": "..."\n'
        "}"
    )

    chat_url = LM_STUDIO_URL.replace("/completions", "/chat/completions")

    payload = {
        "model": LM_STUDIO_MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }

    response = requests.post(chat_url, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    content = ""

    if data.get("choices"):
        choice = data["choices"][0]
        content = (
            choice.get("message", {}).get("content")
            or choice.get("text", "")
            or ""
        )

    parsed = extract_json_object(content)

    if parsed is None:
        return {
            "status": "error",
            "raw_response": content,
            "input_params": "",
            "object": "",
            "output_params": ""
        }

    return {
        "status": "ok",
        "raw_response": content,
        "input_params": normalize_list_or_str(parsed.get("input_params", "")),
        "object": normalize_list_or_str(parsed.get("object", "")),
        "output_params": normalize_list_or_str(parsed.get("output_params", "")),
    }


def main():
    texts = read_texts(INPUT_PATH)

    print(f"найдено текстов: {len(texts)}")
    print("начинаю прогон без rag...")

    rows = []

    for i, text in enumerate(texts, start=1):
        print(f"[{i}/{len(texts)}] думаю...")

        try:
            result = call_model(text)
            rows.append({
                "id": i,
                "text": text,
                "status": result["status"],
                "source": "no_rag",
                "input_params": result["input_params"],
                "object": result["object"],
                "output_params": result["output_params"],
                "raw_response": result["raw_response"],
            })
        except Exception as e:
            rows.append({
                "id": i,
                "text": text,
                "status": f"exception: {str(e)}",
                "source": "no_rag",
                "input_params": "",
                "object": "",
                "output_params": "",
                "raw_response": "",
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "text",
                "status",
                "source",
                "input_params",
                "object",
                "output_params",
                "raw_response",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"готово: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
