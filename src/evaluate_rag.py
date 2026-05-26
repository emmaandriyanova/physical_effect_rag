import csv
from pathlib import Path

from config import LM_STUDIO_URL, LM_STUDIO_MODEL_ID
from run_pipeline import RAGPipeline


def read_texts(file_path: Path) -> list[str]:
    text = file_path.read_text(encoding="utf-8").strip()

    parts = [part.strip() for part in text.split("\n\n") if part.strip()]
    return parts


def main():
    project_root = Path(__file__).parent.parent

    input_path = project_root / "data" / "patents_40_texts.txt"
    output_path = project_root / "data" / "results_rag.csv"

    texts = read_texts(input_path)

    print(f"найдено текстов: {len(texts)}")
    print("загрузка системы...")

    pipeline = RAGPipeline(
        lm_studio_url=LM_STUDIO_URL,
        model_id=LM_STUDIO_MODEL_ID
    )

    print("система готова")
    print("начинаю прогон...\n")

    rows = []

    for i, text in enumerate(texts, start=1):
        print(f"[{i}/{len(texts)}] думаю...")

        try:
            result = pipeline.run(text)

            if result.get("status") == "ok":
                final_result = result.get("result", {})
                rows.append({
                    "id": i,
                    "text": text,
                    "status": "ok",
                    "source": result.get("source", ""),
                    "input_params": final_result.get("input_params", ""),
                    "object": final_result.get("object", ""),
                    "output_params": final_result.get("output_params", ""),
                })
            else:
                rows.append({
                    "id": i,
                    "text": text,
                    "status": "error",
                    "source": "",
                    "input_params": "",
                    "object": "",
                    "output_params": "",
                })

        except Exception as e:
            rows.append({
                "id": i,
                "text": text,
                "status": f"exception: {str(e)}",
                "source": "",
                "input_params": "",
                "object": "",
                "output_params": "",
            })

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
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
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nготово")
    print(f"результаты сохранены в: {output_path}")


if __name__ == "__main__":
    main()