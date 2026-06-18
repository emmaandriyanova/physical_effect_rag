"""
Векторный поиск похожих физических эффектов в базе данных LanceDB для
формирования RAG-контекста. Поддерживает ранжирование по типу чанка,
семантическому сходству и структурному совпадению полей.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
from pathlib import Path
import logging

import lancedb
from sentence_transformers import SentenceTransformer

from config import LANCEDB_PATH, LANCEDB_TABLE, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FERetriever:
    def __init__(
        self,
        db_path: str = LANCEDB_PATH,
        table_name: str = LANCEDB_TABLE,
        embedding_model: str = EMBEDDING_MODEL
    ):
        project_root = Path(__file__).parent.parent
        self.db_path = project_root / db_path

        self.db = lancedb.connect(str(self.db_path))
        self.table = self.db.open_table(table_name)

        logging.info("Загружаем embedding model для retrieval...")
        self.embedder = SentenceTransformer(embedding_model)
        logging.info("Embedding model загружена")

    def _embed_query(self, query_text: str):
        return self.embedder.encode(
            f"query: {query_text}",
            normalize_embeddings=True
        ).tolist()

    @staticmethod
    def _format_row(row: dict) -> dict:
        return {
            "chunk_id": row.get("chunk_id", ""),
            "effect_id": row.get("effect_id", ""),
            "effect_name": row.get("effect_name", ""),
            "chunk_type": row.get("chunk_type", ""),
            "text": row.get("text", ""),
            "description": row.get("description", ""),
            "canonical_description": row.get("canonical_description", ""),
            "description_hash": row.get("description_hash", ""),
            "input_params": row.get("input_params", ""),
            "object": row.get("object", ""),
            "output_params": row.get("output_params", ""),
            "_distance": float(row.get("_distance")) if row.get("_distance", None) is not None else None,
        }

    @staticmethod
    def _deduplicate_by_effect(results: list[dict]) -> list[dict]:
        unique = []
        seen = set()

        for row in results:
            key = row.get("effect_id", "")
            if key and key not in seen:
                unique.append(row)
                seen.add(key)

        return unique

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        chunk_types: list[str] | None = None
    ) -> list[dict]:
        """
        Выполняет векторный поиск похожих чанков в LanceDB.

        :param query_text: текст запроса
        :param top_k: максимальное число возвращаемых результатов
        :param chunk_types: фильтр по типам чанков (record, description, triple);
                            None — без фильтрации
        :return: список словарей с полями chunk_id, effect_id, effect_name, input_params и др.
        """
        if not query_text or not query_text.strip():
            return []

        query_vector = self._embed_query(query_text.strip())

        results = (
            self.table.search(query_vector)
            .limit(top_k * 8)
            .to_list()
        )

        if chunk_types:
            allowed = set(chunk_types)
            results = [row for row in results if row.get("chunk_type") in allowed]

        formatted = [self._format_row(row) for row in results]
        return formatted[:top_k]

    def search_examples(self, query_text: str, top_k: int = 5) -> list[dict]:
        results = self.search(
            query_text,
            top_k=top_k * 6,
            chunk_types=["record", "description", "triple"]
        )

        for row in results:
            distance = row.get("_distance", 1.0) or 1.0
            bonus = 0.0

            if row.get("chunk_type") == "record":
                bonus = 0.03
            elif row.get("chunk_type") == "description":
                bonus = 0.02
            elif row.get("chunk_type") == "triple":
                bonus = 0.01

            row["_rank_score"] = (1.0 - distance) + bonus

        best_by_effect = {}
        for row in results:
            effect_id = row.get("effect_id", "")
            if not effect_id:
                continue

            if effect_id not in best_by_effect:
                best_by_effect[effect_id] = row
            else:
                if row["_rank_score"] > best_by_effect[effect_id]["_rank_score"]:
                    best_by_effect[effect_id] = row

        ranked = list(best_by_effect.values())
        ranked.sort(key=lambda x: x.get("_rank_score", 0.0), reverse=True)
        ranked = self._rerank_by_structure(ranked, query_text)

        return ranked[:top_k]

    def _rerank_by_structure(self, results: list[dict], query_text: str) -> list[dict]:
        if not results:
            return results

        query_lower = query_text.lower()

        input_keywords = {
            "температур": "температура",
            "концентрац": "концентрация",
            "коэффициент теплопередач": "коэффициент теплопередачи",
            "магнитн": "магнитное поле",
            "электрическ поле": "электрическое поле",
            "давлен": "давление",
            "толщин": "толщина",
            "скорост охлажден": "скорость охлаждения",
            "объемн дол": "объемная доля",
        }

        output_keywords = {
            "потер": "потери",
            "коэффициент линейного расширения": "тклр",
            "тклр": "тклр",
            "намагничен": "намагниченность",
            "сопротивлен": "сопротивление",
            "теплоемкост": "теплоёмкость",
            "скорост охлажден": "скорость охлаждения",
            "вязкост": "вязкость",
            "температур кюри": "температура кюри",
            "коэрцитивн": "коэрцитивная сила",
            "прочност": "прочность",
        }

        query_input_type = next(
            (v for k, v in input_keywords.items() if k in query_lower), None
        )
        query_output_type = next(
            (v for k, v in output_keywords.items() if k in query_lower), None
        )

        for row in results:
            bonus = row.get("_rank_score", 0.0)

            inp = str(row.get("input_params", "")).lower()
            out = str(row.get("output_params", "")).lower()

            if query_input_type and query_input_type in inp:
                bonus += 0.05

            if query_output_type and query_output_type in out:
                bonus += 0.10

            row["_rank_score"] = bonus

        results.sort(key=lambda x: x.get("_rank_score", 0.0), reverse=True)
        return results

    def get_example_bundle(self, query_text: str, main_k: int = 1, aux_k: int = 2) -> dict:
        """
        Возвращает набор примеров для RAG-контекста: один главный и несколько вспомогательных.

        :param query_text: текст запроса
        :param main_k: количество главных примеров (обычно 1)
        :param aux_k: количество вспомогательных примеров из других эффектов
        :return: словарь с ключами main_example (dict или None) и aux_examples (list[dict])
        """

        examples = self.search_examples(query_text, top_k=main_k + aux_k + 3)

        if not examples:
            return {
                "main_example": None,
                "aux_examples": []
            }

        main_example = examples[0]

        aux_examples = []
        main_id = main_example.get("effect_id", "")

        for ex in examples[1:]:
            if ex.get("effect_id", "") != main_id:
                aux_examples.append(ex)
            if len(aux_examples) >= aux_k:
                break

        return {
            "main_example": main_example,
            "aux_examples": aux_examples
        }


if __name__ == "__main__":
    retriever = FERetriever()

    test_text = (
        "Магнитопластический эффект повышение пластической деформации металлов "
        "под действием магнитного поля в условиях нагружения."
    )

    bundle = retriever.get_example_bundle(test_text, main_k=1, aux_k=2)
    print(bundle)
