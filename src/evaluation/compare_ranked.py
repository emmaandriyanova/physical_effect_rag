"""
Сравнение результатов RAG-системы и базовой модели без RAG по ранговой
метрике (I–VI / NONE) на эталонной выборке из 40 патентов. Выводит сводную
таблицу точности и детальные примеры по каждому рангу совпадения.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import csv
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EMBEDDING_MODEL,
    EVAL_GROUND_TRUTH_PATH as GROUND_TRUTH_PATH,
    EVAL_RAG_RESULTS_PATH as RAG_RESULTS_PATH,
    EVAL_NO_RAG_RESULTS_PATH as NO_RAG_RESULTS_PATH,
    EVAL_COMPARISON_DETAILS_OUT as DETAILS_OUT,
    EVAL_COMPARISON_SUMMARY_OUT as SUMMARY_OUT,
)


print("загружаем модель эмбеддингов...")
EMBEDDER = SentenceTransformer(EMBEDDING_MODEL)
print("модель загружена\n")

SIM_FULL = 0.92
SIM_BASE = 0.80

_EMB_CACHE: dict[str, np.ndarray] = {}


def embed(text: str) -> np.ndarray:
    text = str(text or "").strip()
    if not text:
        return np.zeros(384, dtype=np.float32)
    if text in _EMB_CACHE:
        return _EMB_CACHE[text]
    vec = EMBEDDER.encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    _EMB_CACHE[text] = vec
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(a) or not np.any(b):
        return 0.0
    return float(np.dot(a, b))


def text_similarity(a: str, b: str) -> float:
    """Косинусное сходство двух строк после OCR-фикса и базовой очистки."""
    a = canonicalize(fix_ocr(str(a or "")))
    b = canonicalize(fix_ocr(str(b or "")))
    if not a or not b:
        return 0.0
    return cosine(embed(a), embed(b))


def wrap_text(text: str, width: int = 80) -> str:
    text = str(text or "").strip()
    return textwrap.fill(text, width=width)


LATIN_TO_CYRILLIC = {
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
    'X': 'Х', 'Y': 'У',
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р',
    'x': 'х', 'y': 'у',
}


def fix_ocr(text: str) -> str:
    def fix_word(word):
        has_cyr = any('\u0400' <= c <= '\u04ff' for c in word)
        has_lat = any(c in LATIN_TO_CYRILLIC for c in word)
        if has_cyr and has_lat:
            return ''.join(LATIN_TO_CYRILLIC.get(c, c) for c in word)
        return word

    tokens = re.split(r'(\s+|[^\w])', text)
    return ''.join(fix_word(t) for t in tokens)


def clean(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("ё", "е").replace("Ё", "е")
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


SYNONYMS: dict[str, str] = {
    "ферромагнетик": "ферромагнитный материал",
    "ферромагнитный материал": "ферромагнитный материал",
    "аморфный сплав": "аморфные сплавы",
    "аморфные сплавы": "аморфные сплавы",
    "металл": "металлы",
    "лазерное излучение": "электромагнитное излучение",
    "силовое (механическое воздействие)": "механическое воздействие",
    "силовое(механическое воздействие)": "механическое воздействие",
    "силовое (механическое) воздействие": "механическое воздействие",
    "упругие (акустические) волны": "ультразвук",
    "упругие(акустические)волны": "ультразвук",
}


def canonicalize(text: str) -> str:
    t = clean(text)
    return SYNONYMS.get(t, t)

def clean_pred_field(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"(?i)объект\s+воздействия\s*\([^)]*\)\s*[—\-–:]+\s*это\s*", "", text)
    text = re.sub(r"(?i)входные\s+параметры?\s*\([^)]*\)\s*[—\-–:]+\s*это?\s*", "", text)
    text = re.sub(r"(?i)выходные\s+параметры?\s*\([^)]*\)\s*[—\-–:]+\s*это?\s*", "", text)
    text = re.sub(r"(?i)(input_params|output_params|object)\s*[=:—\-–]+\s*", "", text)
    return text.strip()

def split_entrada_items(text: str) -> list[str]:
    text = str(text or "").strip()
    if not text or text.lower() in ("nan", "none", ""):
        return []

    if re.search(r"(?i)(вход|выход)\d+\s*:", text):
        parts = re.split(r"(?i)(?:вход|выход)\d+\s*:", text)
        return [p.strip() for p in parts if p.strip()]

    if "," in text and text.count(".") <= 1:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) > 1:
            return parts

    return [text.strip()]


def extract_name_and_qualities(item: str) -> tuple[str, list[str]]:
    parts = [p.strip() for p in re.split(r"[.;]", item) if p.strip()]
    if not parts:
        return "", []

    name = canonicalize(parts[0])

    quality_markers = [
        "переменн", "постоянн", "высокочастот", "низкочастот",
        "импульс", "неоднород", "однород", "увеличен", "уменьшен",
        "снижени", "повышени", "монотон", "немонотон",
    ]
    qualities = []
    for p in parts[1:]:
        pc = clean(p)
        if any(m in pc for m in quality_markers):
            qualities.append(canonicalize(p))

    return name, qualities


def parse_field(text: str) -> list[tuple[str, list[str]]]:
    items = split_entrada_items(text)
    return [extract_name_and_qualities(it) for it in items if it]


def names_match(a: str, b: str) -> bool:
    return text_similarity(a, b) >= SIM_BASE


def qualities_match(gold_qs: list[str], pred_qs: list[str]) -> bool:
    if not gold_qs:
        return True
    if not pred_qs:
        return False
    for gq in gold_qs:
        if not any(names_match(gq, pq) for pq in pred_qs):
            return False
    return True


def field_full_match(gold: str, pred: str) -> bool:
    g = canonicalize(fix_ocr(str(gold or "")))
    p = canonicalize(fix_ocr(str(pred or "")))
    if not g or not p:
        return False
    return g == p


def field_base_match(gold_parsed: list, pred_parsed: list) -> bool:
    gold_names = [name for name, _ in gold_parsed if name]
    if not gold_names:
        return False
    pred_names = [name for name, _ in pred_parsed if name]
    if not pred_names:
        return False
    return all(any(names_match(gn, pn) for pn in pred_names) for gn in gold_names)


def field_quality_match(gold_parsed: list, pred_parsed: list) -> bool:
    gold_all_q = [q for _, qs in gold_parsed for q in qs]
    if not gold_all_q:
        return True
    pred_all_q = [q for _, qs in pred_parsed for q in qs]
    return qualities_match(gold_all_q, pred_all_q)


def object_match(gold: str, pred: str) -> bool:
    return names_match(gold, pred)

def rank_match(
    gold_input: str, gold_object: str, gold_output: str,
    pred_input: str, pred_object: str, pred_output: str,
) -> str:
    gold_input  = fix_ocr(str(gold_input  or ""))
    gold_object = fix_ocr(str(gold_object or ""))
    gold_output = fix_ocr(str(gold_output or ""))

    pred_input  = clean_pred_field(str(pred_input  or ""))
    pred_object = clean_pred_field(str(pred_object or ""))
    pred_output = clean_pred_field(str(pred_output or ""))

    gi = parse_field(gold_input)
    pi = parse_field(pred_input)
    go = parse_field(gold_output)
    po = parse_field(pred_output)

    inp_full = field_full_match(gold_input, pred_input)
    inp_base = field_base_match(gi, pi)
    inp_qual = field_quality_match(gi, pi)

    out_full = field_full_match(gold_output, pred_output)
    out_base = field_base_match(go, po)
    out_qual = field_quality_match(go, po)

    obj_ok = object_match(gold_object, pred_object)

    if inp_full and obj_ok and out_full:                            return "I"
    if obj_ok and inp_base and inp_qual and out_base and out_qual:  return "II"
    if obj_ok and inp_base and out_base:                            return "III"
    if inp_full and out_full:                                       return "IV"
    if inp_base and inp_qual and out_base and out_qual:             return "V"
    if inp_base and out_base:                                       return "VI"
    return "NONE"


RANK_SCORE = {"I": 1.0, "II": 0.7, "III": 0.5, "IV": 0.3,
              "V": 0.15, "VI": 0.03, "NONE": 0.0}


def load_csv(path: Path) -> dict[int, dict]:
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                idx = int(str(row.get("id", "")).strip())
                rows[idx] = row
            except ValueError:
                pass
    return rows


def summarize(details: list[dict], prefix: str) -> dict:
    ranks  = [r[f"{prefix}_rank"]  for r in details]
    scores = [float(r[f"{prefix}_score"]) for r in details]
    cnt    = Counter(ranks)
    total  = len(details) or 1

    strong         = cnt["I"] + cnt["II"] + cnt["III"]
    weak_or_better = strong + cnt["IV"] + cnt["V"] + cnt["VI"]

    return {
        "system":             prefix,
        "total":              total,
        "I":                  cnt["I"],
        "II":                 cnt["II"],
        "III":                cnt["III"],
        "IV":                 cnt["IV"],
        "V":                  cnt["V"],
        "VI":                 cnt["VI"],
        "NONE":               cnt["NONE"],
        "strong_precision_%": round(strong / total * 100, 2),
        "weak_or_better_%":   round(weak_or_better / total * 100, 2),
        "mean_similarity":    round(sum(scores) / total, 4),
        "mean_similarity_percent": round(sum(scores) / total * 100, 2),
    }

def main():
    gt     = load_csv(GROUND_TRUTH_PATH)
    rag    = load_csv(RAG_RESULTS_PATH)
    no_rag = load_csv(NO_RAG_RESULTS_PATH)

    ids     = sorted(gt.keys())
    details = []

    for idx in ids:
        gt_row     = gt.get(idx, {})
        rag_row    = rag.get(idx, {})
        no_rag_row = no_rag.get(idx, {})

        gold_input  = gt_row.get("gold_input",  "")
        gold_object = gt_row.get("gold_object", "")
        gold_output = gt_row.get("gold_output", "")

        r_input  = rag_row.get("input_params",  "")
        r_object = rag_row.get("object",        "")
        r_output = rag_row.get("output_params", "")

        n_input  = no_rag_row.get("input_params",  "")
        n_object = no_rag_row.get("object",        "")
        n_output = no_rag_row.get("output_params", "")

        rag_rank    = rank_match(gold_input, gold_object, gold_output,
                                 r_input, r_object, r_output)
        no_rag_rank = rank_match(gold_input, gold_object, gold_output,
                                 n_input, n_object, n_output)

        details.append({
            "id":            idx,
            "gold_input":    gold_input,
            "gold_object":   gold_object,
            "gold_output":   gold_output,
            "rag_input":     r_input,
            "rag_object":    r_object,
            "rag_output":    r_output,
            "rag_rank":      rag_rank,
            "rag_score":     RANK_SCORE[rag_rank],
            "no_rag_input":  n_input,
            "no_rag_object": n_object,
            "no_rag_output": n_output,
            "no_rag_rank":   no_rag_rank,
            "no_rag_score":  RANK_SCORE[no_rag_rank],
        })

    DETAILS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DETAILS_OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    summary_rows = [summarize(details, "rag"), summarize(details, "no_rag")]
    with open(SUMMARY_OUT, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Детали:  {DETAILS_OUT}")
    print(f"Сводка:  {SUMMARY_OUT}\n")

    for row in summary_rows:
        print(f" {row['system'].upper()} ")
        print(f"  Всего: {row['total']}")
        for rank in ["I", "II", "III", "IV", "V", "VI", "NONE"]:
            print(f"  Ранг {rank:4s}: {row[rank]:3d}")
        print(f"  Сильное совпадение (I-III):   {row['strong_precision_%']}%")
        print(f"  Любое совпадение (I-VI):      {row['weak_or_better_%']}%")
        print(f"  Средний балл сходства:        {row['mean_similarity']}")
        print()

    for prefix in ("rag", "no_rag"):
        print(f"\n{'=' * 50}")
        print(f"  ID по рангам — {prefix.upper()}")
        print(f"{'=' * 50}")

        by_rank: dict[str, list] = {}
        for row in details:
            rank = row[f"{prefix}_rank"]
            by_rank.setdefault(rank, []).append(str(row["id"]))

        for rank in ["I", "II", "III", "IV", "V", "VI", "NONE"]:
            ids = by_rank.get(rank, [])
            ids_str = ", ".join(ids) if ids else "—"
            print(f"  Ранг {rank:5s} ({len(ids):2d}): {ids_str}")

    for prefix in ("rag", "no_rag"):
        print(f"\n{'=' * 60}")
        print(f" Примеры по рангам — {prefix.upper()} ")
        print(f"{'=' * 60}")

        seen_ranks: dict[str, dict] = {}
        for row in details:
            rank = row[f"{prefix}_rank"]
            if rank not in seen_ranks:
                seen_ranks[rank] = row

        for rank in ["I", "II", "III", "IV", "V", "VI", "NONE"]:
            if rank not in seen_ranks:
                continue

            row = seen_ranks[rank]

            print(f"\n--- ранг {rank} (id={row['id']}) ---")
            print("-" * 60)
            print(f"[эталон]")
            print("Вход:")
            print(wrap_text(row["gold_input"]))
            print("Объект:")
            print(wrap_text(row["gold_object"]))
            print("Выход:")
            print(wrap_text(row["gold_output"]))

            print(f"\n[итог]")
            print("Вход:")
            print(wrap_text(row[f"{prefix}_input"]))
            print("Объект:")
            print(wrap_text(row[f"{prefix}_object"]))
            print("Выход:")
            print(wrap_text(row[f"{prefix}_output"]))


if __name__ == "__main__":
    main()