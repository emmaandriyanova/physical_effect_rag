"""
Форматирование полей физического эффекта по стандарту FEText: сопоставление
терминов с тезаурусом, выбор модификаторов из контекста, нормализация
регистра и пунктуации, обработка множественных входов (Вход1/Вход2).

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import re
from typing import Optional

from thesaurus_match import ThesaurusMatcher


class FETextFormatter:

    def __init__(self, matcher: ThesaurusMatcher):
        self.matcher = matcher
        self.df = matcher.df.fillna("")

    @staticmethod
    def _clean(text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _cap(text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return ""
        if text.isupper():
            text = text.lower()
        return text[0].upper() + text[1:]

    @staticmethod
    def _split_fetext_items(text: str) -> list[str]:
        text = str(text or "").strip()
        if not text:
            return []

        if "Вход1:" in text or "Вход2:" in text or "Вход3:" in text:
            parts = re.split(r"(?:^|\s)(Вход\d+:)", text)
            items = []
            current = ""

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                if re.match(r"Вход\d+:", part):
                    if current:
                        items.append(current.strip())
                    current = part
                else:
                    current = f"{current} {part}".strip()

            if current:
                items.append(current.strip())

            cleaned = []
            for item in items:
                item = re.sub(r"^Вход\d+:\s*", "", item).strip()
                if item:
                    cleaned.append(item)
            return cleaned

        return [text]

    @staticmethod
    def _normalize_formatting(text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return text
        text = re.sub(r'(\w)\(', r'\1 (', text)
        text = re.sub(r'\.\s+([а-яёa-zA-Z])', lambda m: '. ' + m.group(1).upper(), text)
        return text

    @staticmethod
    def _looks_abbreviated(text: str) -> bool:
        t = str(text or "").strip()
        if not t:
            return True

        bad_patterns = [
            r"\bмагн\.",
            r"\bдеформиров\.",
            r"\bразн\.",
            r"\bпотенц\.",
            r"\bэлектр\.",
            r"\bграв\.",
            r"\bсуммари",
        ]
        return any(re.search(p, t.lower()) for p in bad_patterns)

    def _find_exact_node(self, term: str) -> Optional[dict]:
        result = self.matcher.find_best(term, top_k=5)
        exact = result.get("exact_match")
        if exact:
            return exact

        similar = result.get("similar", [])
        picked = self.matcher._pick_semantic_candidate(similar)
        return picked

    def _find_branch_children(self, node_name: str, branch_hint: str) -> list[str]:
        node = self._find_exact_node(node_name)
        if not node:
            return []

        node_id = str(node.get("node_id", "")).strip()
        if not node_id:
            return []

        branch_rows = self.df[
            (self.df["parent"] == node_id) &
            (self.df["name"].str.lower() == branch_hint.lower())
        ]

        if branch_rows.empty:
            return []

        branch_id = str(branch_rows.iloc[0]["node_id"])
        child_rows = self.df[self.df["parent"] == branch_id]

        names = [self._cap(str(x).strip()) for x in child_rows["name"].tolist() if str(x).strip()]
        return names

    def _choose_modifiers_from_text(self, text: str, term_name: str) -> list[str]:

        text_low = self._clean(text).lower()
        result = []

        for branch in ["Временные характеристики", "Пространственные характеристики"]:
            candidates = self._find_branch_children(term_name, branch)
            for cand in candidates:
                cand_low = cand.lower()

                if cand_low in text_low:
                    result.append(self._cap(cand))

                if "переменн" in text_low and "переменн" in cand_low:
                    result.append(self._cap(cand))
                if "постоянн" in text_low and "постоянн" in cand_low:
                    result.append(self._cap(cand))
                if "низкочастот" in text_low and "низкочастот" in cand_low:
                    result.append(self._cap(cand))
                if "высокочастот" in text_low and "высокочастот" in cand_low:
                    result.append(self._cap(cand))
                if "однородн" in text_low and "однородн" in cand_low:
                    result.append(self._cap(cand))
                if "неоднородн" in text_low and "неоднородн" in cand_low:
                    result.append(self._cap(cand))
                if "импульс" in text_low and "импульс" in cand_low:
                    result.append(self._cap(cand))

        uniq = []
        seen = set()
        for x in result:
            k = x.lower()
            if k not in seen:
                uniq.append(x)
                seen.add(k)

        return uniq

    def _canonical_term(self, text: str, role: str) -> str:
        clean_text = self._cap(text)

        if role == "input":
            return clean_text

        result = self.matcher.find_best(text, top_k=5, role=role)
        exact = result.get("exact_match")
        similar = result.get("similar", [])
        picked = exact or self.matcher._pick_semantic_candidate(similar, role=role)

        if not picked:
            return clean_text

        thes_name = self._cap(picked.get("name", ""))

        if self._looks_abbreviated(thes_name):
            return clean_text

        if len(clean_text) >= len(thes_name):
            return clean_text

        return thes_name

    def format_input_field(self, input_text: str) -> str:

        term = self._canonical_term(input_text, role="input")
        modifiers = self._choose_modifiers_from_text(input_text, term)

        parts = [term]
        parts.extend(modifiers)

        uniq = []
        seen = set()
        for p in parts:
            k = p.lower()
            if k not in seen:
                uniq.append(p)
                seen.add(k)

        return self._normalize_formatting(". ".join([p for p in uniq if p]).strip())

    def format_output_field(self, output_text: str) -> str:
        return self._normalize_formatting(self._canonical_term(output_text, role="output"))

    def format_object_field(self, object_text: str) -> str:
        return self._normalize_formatting(self._canonical_term(object_text, role="object"))

    def format_input_params(self, input_params: str) -> str:
        items = self._split_fetext_items(input_params)

        if not items:
            return ""

        if len(items) == 1:
            return self.format_input_field(items[0]).strip()

        formatted = []
        for idx, item in enumerate(items, start=1):
            formatted.append(f"Вход{idx}:{self.format_input_field(item)}")
        return " ".join(formatted).strip()

    def format_output_params(self, output_params: str) -> str:
        items = self._split_fetext_items(output_params)

        formatted = []
        for item in items:
            item = self._clean(item)
            if not item:
                continue
            formatted.append(self.format_output_field(item))

        uniq = []
        seen = set()
        for x in formatted:
            k = x.lower()
            if k not in seen:
                uniq.append(x)
                seen.add(k)

        return ". ".join(uniq).strip()

    def format_fields(self, input_params: str, object_text: str, output_params: str) -> dict:
        return {
            "input_params": self.format_input_params(input_params),
            "object": self.format_object_field(object_text),
            "output_params": self.format_output_params(output_params)
        }
