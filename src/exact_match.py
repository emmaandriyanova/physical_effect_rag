import pandas as pd
from pathlib import Path
import hashlib
import html
import re
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(str(text))
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def canonicalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"\s*([,.;:!?()])\s*", r"\1", text)
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class ExactMatchFinder:
    def __init__(self, csv_path: str | Path | None = None):
        if csv_path is None:
            csv_path = Path(__file__).parent.parent / "data" / "parsed_effects.csv"

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"не найден файл: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path).fillna("")

        required_columns = [
            "id",
            "name",
            "description",
            "canonical_description",
            "description_hash",
            "input_params",
            "object",
            "output_params",
        ]

        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"в parsed_effects.csv не хватает колонок: {missing}")

        logging.info(f"загружено {len(self.df)} эффектов из {self.csv_path}")

        self.embedder = SentenceTransformer("intfloat/multilingual-e5-small")
        self.embeddings_cache_path = self.csv_path.parent / "exact_match_embeddings.npy"

        if self.embeddings_cache_path.exists():
            logging.info(f"загружаем эмбеддинги из кэша: {self.embeddings_cache_path}")
            embeddings = np.load(self.embeddings_cache_path)
        else:
            logging.info("строим эмбеддинги для канона...")

            texts = [
                f"passage: {str(x)}"
                for x in self.df["canonical_description"].tolist()
            ]

            embeddings = self.embedder.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=64
            )

            np.save(self.embeddings_cache_path, embeddings)
            logging.info(f"эмбеддинги сохранены в кэш: {self.embeddings_cache_path}")

        self.df["embedding"] = list(embeddings)
        logging.info("эмбедднги готовы")

    def find(self, query_text: str) -> dict | None:
        canonical_query = canonicalize_text(query_text)
        query_hash = text_hash(canonical_query) if canonical_query else ""

        hash_matches = self.df[self.df["description_hash"] == query_hash]
        if not hash_matches.empty:
            row = hash_matches.iloc[0]
            logging.info(f"найдено соответствие по хэш: ID={row['id']}, name={row['name']}")
            return self._row_to_result(row, match_type="hash")

        canonical_matches = self.df[self.df["canonical_description"] == canonical_query]
        if not canonical_matches.empty:
            row = canonical_matches.iloc[0]
            logging.info(f"найдено соответствие по канону: ID={row['id']}, name={row['name']}")
            return self._row_to_result(row, match_type="canonical")

        logging.info("соответствие не найдено")
        return None

    @staticmethod
    def _row_to_result(row, match_type: str) -> dict:
        return {
            "match_found": True,
            "match_type": match_type,
            "effect_id": str(row["id"]),
            "effect_name": str(row["name"]),
            "description": str(row["description"]),
            "canonical_description": str(row["canonical_description"]),
            "description_hash": str(row["description_hash"]),
            "input_params": str(row["input_params"]),
            "object": str(row["object"]),
            "output_params": str(row["output_params"]),
        }

    def find_near_duplicate(self, query_text: str) -> dict | None:
        canonical_query = canonicalize_text(query_text)
        if not canonical_query:
            return None

        query_vec = self.embedder.encode(
            f"query: {canonical_query}",
            normalize_embeddings=True
        )

        best_score = -1.0
        best_row = None

        for _, row in self.df.iterrows():
            score = float(np.dot(query_vec, row["embedding"]))
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is None:
            return None

        if best_score >= 0.93:
            mode = "hard_example" if best_score >= 0.975 else "soft_example"
            return {
                "mode": mode,
                "score": best_score,
                "example": {
                    "effect_id": str(best_row["id"]),
                    "effect_name": str(best_row["name"]),
                    "description": str(best_row["description"]),
                    "input_params": str(best_row["input_params"]),
                    "object": str(best_row["object"]),
                    "output_params": str(best_row["output_params"])
                }
            }

        return None


if __name__ == "__main__":
    finder = ExactMatchFinder()

    test_text = """
    Затухание электромагнитных волн по мере их проникновения в глубь проводящей среды.
    В результате - неоднородное распределение переменного высокочастотного тока по сечению проводника.
    Ток течет, в основном, в узком поверхностном слое (скин слое) проводника и практически отсутствует в глубине.
    Чем больше частота тока, тем меньше толщина скин-слоя.
    """

    result = finder.find(test_text)

    print("\nрезультат:")
    if result:
        print(f"ID: {result['effect_id']}")
        print(f"Название: {result['effect_name']}")
        print(f"Тип совпадения: {result['match_type']}")
        print(f"Вход: {result['input_params']}")
        print(f"Объект: {result['object']}")
        print(f"Выход: {result['output_params']}")
    else:
        print("Совпадение не найдено")
