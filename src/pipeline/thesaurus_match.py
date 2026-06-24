"""
Поиск терминов в тезаурусе физических эффектов: точное совпадение по
MD5-хешу канонической формы и семантический поиск через LanceDB с
фильтрацией по роли (вход / объект / выход) и лифтингом терминов.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
from pathlib import Path
import logging
import hashlib
import html
import re

import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

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


class ThesaurusMatcher:
    def __init__(
        self,
        db_path: str = "./lancedb_data",
        table_name: str = "thesaurus",
        embedding_model: str = "intfloat/multilingual-e5-small"
    ):
        project_root = Path(__file__).parent.parent.parent
        self.db_path = project_root / db_path

        self.db = lancedb.connect(str(self.db_path))
        self.table = self.db.open_table(table_name)

        logging.info("Загружаем эмбеддинговую модель для тезауруса...")
        self.embedder = SentenceTransformer(embedding_model)
        logging.info("эмбеддинговая модель загружена")

        self.df = self.table.to_pandas().fillna("")
        logging.info(f"загружено {len(self.df)} записей тезауруса")

    @staticmethod
    def _path_parts(item: dict) -> list[str]:
        full_path = str(item.get("full_path", ""))
        return [p.strip() for p in full_path.split(">") if p.strip()]

    @staticmethod
    def _path_parts_lower(item: dict) -> list[str]:
        return [p.lower() for p in ThesaurusMatcher._path_parts(item)]

    @staticmethod
    def _is_input_branch(item: dict) -> bool:
        parts = ThesaurusMatcher._path_parts_lower(item)
        return any("воздейств" in p for p in parts)

    @staticmethod
    def _is_output_like_branch(item: dict) -> bool:
        parts = ThesaurusMatcher._path_parts_lower(item)
        keywords = [
            "состояние",
            "деформ",
            "давление",
            "ток",
            "намагнич",
            "поляризац",
            "скорость",
            "ускорение",
            "температур",
            "нагрев",
            "смещение",
        ]
        return any(any(k in p for k in keywords) for p in parts)

    @staticmethod
    def _is_object_like_branch(item: dict) -> bool:
        parts = ThesaurusMatcher._path_parts_lower(item)
        keywords = [
            "объект",
            "веществ",
            "материал",
            "сред",
            "тело",
            "металл",
            "газ",
            "жидк",
            "кристалл",
        ]
        return any(any(k in p for k in keywords) for p in parts)

    @staticmethod
    def _lift_input_term(item: dict) -> dict:
        if not item:
            return item

        path_parts = ThesaurusMatcher._path_parts(item)
        lower_parts = [p.lower() for p in path_parts]

        if not path_parts:
            return item

        for i, part in enumerate(lower_parts):
            if "физическая величина" in part and i > 0:
                lifted_name = path_parts[i - 1]
                lifted_path = " > ".join(path_parts[:i])
                new_item = dict(item)
                new_item["name"] = lifted_name
                new_item["canonical_name"] = canonicalize_text(lifted_name)
                new_item["full_path"] = lifted_path
                new_item["path"] = lifted_path
                new_item["lifted"] = True
                return new_item

        if len(path_parts) >= 4:
            for i, part in enumerate(lower_parts):
                if "воздейств" in part and i + 1 < len(path_parts):
                    lifted_name = path_parts[i + 1]
                    lifted_path = " > ".join(path_parts[:i + 2])
                    new_item = dict(item)
                    new_item["name"] = lifted_name
                    new_item["canonical_name"] = canonicalize_text(lifted_name)
                    new_item["full_path"] = lifted_path
                    new_item["path"] = lifted_path
                    new_item["lifted"] = True
                    return new_item

        return item

    def _filter_by_role(self, items: list[dict], role: str) -> list[dict]:
        filtered = []

        for item in items:

            if role == "input":
                if self._is_input_branch(item):
                    lifted = self._lift_input_term(item)
                    filtered.append(lifted)

            elif role == "object":
                if not self._is_input_branch(item) and self._is_object_like_branch(item):
                    filtered.append(item)

            elif role == "output":
                if not self._is_input_branch(item) and self._is_output_like_branch(item):
                    filtered.append(item)

        return filtered

    def find_exact(self, term: str) -> dict | None:
        canonical_term = canonicalize_text(term)
        term_hash = text_hash(canonical_term) if canonical_term else ""

        hash_matches = self.df[self.df["name_hash"] == term_hash]
        if not hash_matches.empty:
            row = hash_matches.iloc[0]
            return self._row_to_result(row, score=1.0, match_type="hash")

        canonical_matches = self.df[self.df["canonical_name"] == canonical_term]
        if not canonical_matches.empty:
            row = canonical_matches.iloc[0]
            return self._row_to_result(row, score=1.0, match_type="canonical")

        exact_name_matches = self.df[self.df["name"] == clean_text(term)]
        if not exact_name_matches.empty:
            row = exact_name_matches.iloc[0]
            return self._row_to_result(row, score=1.0, match_type="name")

        return None

    def find_similar(self, term: str, top_k: int = 5) -> list[dict]:
        query = clean_text(term)
        if not query:
            return []

        query_vector = self.embedder.encode(
            f"query: {query}",
            normalize_embeddings=True
        ).tolist()

        results = (
            self.table.search(query_vector)
            .limit(top_k)
            .to_list()
        )

        return [
            {
                "node_id": row.get("node_id", ""),
                "name": row.get("name", ""),
                "canonical_name": row.get("canonical_name", ""),
                "full_path": row.get("full_path", ""),
                "path": row.get("path", ""),
                "path_depth": row.get("path_depth", 0),
                "top_level": row.get("top_level", ""),
                "parent": row.get("parent", ""),
                "_distance": row.get("_distance", None),
                "match_type": "semantic"
            }
            for row in results
        ]

    @staticmethod
    def _pick_semantic_candidate(similar: list[dict], role: str | None = None) -> dict | None:
        if not similar:
            return None

        threshold_by_role = {
            "input": 0.80,
            "object": 0.86,
            "output": 0.82,
            None: 0.84
        }

        min_score = threshold_by_role.get(role, 0.84)

        best = None
        best_score = -1.0

        for item in similar:
            distance = item.get("_distance", None)
            if distance is None:
                continue

            score = 1.0 - float(distance)
            if score > best_score:
                best_score = score
                best = dict(item)
                best["_semantic_score"] = score

        if best and best_score >= min_score:
            return best

        return None

    def find_best(self, term: str, top_k: int = 5, role: str | None = None) -> dict:
        """
        Ищет наилучшее совпадение для термина в тезаурусе: сначала точное, затем семантическое.

        :param term: искомый термин
        :param top_k: число кандидатов для семантического поиска
        :param role: роль термина — 'input', 'object', 'output' или None (без фильтрации)
        :return: словарь с ключами query, exact_match (dict или None), similar (list[dict])
        """
        exact = self.find_exact(term)

        if exact:
            if role == "input":
                exact = self._lift_input_term(exact)

            return {
                "query": term,
                "exact_match": exact,
                "similar": []
            }

        similar = self.find_similar(term, top_k=top_k)

        if role:
            similar = self._filter_by_role(similar, role=role)

        return {
            "query": term,
            "exact_match": None,
            "similar": similar
        }

    @staticmethod
    def _row_to_result(row, score: float, match_type: str) -> dict:
        return {
            "node_id": row["node_id"],
            "name": row["name"],
            "canonical_name": row["canonical_name"],
            "full_path": row["full_path"],
            "path": row["path"],
            "path_depth": row["path_depth"],
            "top_level": row["top_level"],
            "parent": row["parent"],
            "score": score,
            "match_type": match_type
        }


if __name__ == "__main__":
    matcher = ThesaurusMatcher()

    test_term = "нагружение"
    result = matcher.find_best(test_term, top_k=10)

    print("\nрезультат:")
    print(result)
