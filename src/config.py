"""
Централизованная конфигурация проекта: URL LM Studio, идентификаторы моделей,
пути к базам данных LanceDB и файлам оценки качества.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
from pathlib import Path

LM_STUDIO_URL = "http://79.170.162.18:1234/v1/completions"
LM_STUDIO_MODEL_ID = "qwen3-8b"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
KEYT5_MODEL_PATH = "/Users/emma/Desktop/диплом/для нейронки/keyt5-finetuned"

LANCEDB_PATH = "./lancedb_data"
LANCEDB_TABLE = "physical_effects"

EVAL_GROUND_TRUTH_PATH = Path("/Users/emma/PycharmProjects/PythonProject8/data/ground_truth_40.csv")
EVAL_RAG_RESULTS_PATH = Path("/Users/emma/PycharmProjects/PythonProject8/data/results_rag.csv")
EVAL_NO_RAG_RESULTS_PATH = Path("/Users/emma/PycharmProjects/PythonProject8/data/results_no_rag.csv")
EVAL_COMPARISON_DETAILS_OUT = Path("/Users/emma/PycharmProjects/PythonProject8/data/comparison_ranked.csv")
EVAL_COMPARISON_SUMMARY_OUT = Path("/Users/emma/PycharmProjects/PythonProject8/data/comparison_summary.csv")
