import re
from thesaurus_match import ThesaurusMatcher


class ThesaurusNormalizer:
    def __init__(self, matcher: ThesaurusMatcher):
        self.matcher = matcher

        self.input_alias_rules = [
            (r"\bмагнит\w*\s+пол\w*\b", "Магнитное поле"),
            (r"\bэлектр\w*\s+пол\w*\b", "Электрическое поле"),
            (r"\bлазер\w*\s+излуч\w*\b", "Лазерное излучение"),
            (r"\bсвет\b|\bсветов\w*\b", "Свет"),
            (r"\bультразв\w*\b", "Ультразвук"),
            (
                r"\bнагружен\w*\b|\bнагруз\w*\b|\bмеханическ\w*\b|\bсжат\w*\b|\bрастяжен\w*\b|\bсдвиг\w*\b|\bизгиб\w*\b",
                "Механическое воздействие"
            ),
            (r"\bвысок\w*\s+давлен\w*\b", "Силовое (механическое) воздействие"),
        ]

        self.output_alias_rules = [
            (r"\bпластическ\w*\s+деформ\w*\b", "Пластическая деформация"),
            (r"\bразупрочнен\w*\b", "Разупрочнение"),
            (r"\bэлектрическ\w*\s+ток\b|\bток\b", "Электрический ток"),
            (r"\bнагрев\w*\b", "Нагрев"),
            (r"\bплавлен\w*\b", "Плавление"),
            (r"\bиспарен\w*\b", "Испарение"),
            (r"\bнамагниченн\w*\b", "Намагниченность"),
            (r"\bполяризац\w*\b", "Поляризация"),
            (r"\bсмещени\w*\b", "Электрическое смещение"),
            (r"\bкавитац\w*\b", "Кавитация"),
            (r"\bфазов\w*\s+переход\w*\b", "Фазовый переход"),
            (r"\bплазм\w*\b", "Образование плазмы"),
        ]

        self.object_alias_rules = [
            (r"\bпроводящ\w*\s+сред\w*\b", "Проводящая среда"),
            (r"\bпроводник\w*\b", "Проводник"),
        ]

    @staticmethod
    def _normalize_to_list(value) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]

        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []

        return [str(value).strip()]

    @staticmethod
    def _clean(text: str) -> str:
        return str(text or "").strip()

    def _apply_alias_rules(self, text: str, rules: list[tuple[str, str]]) -> str | None:
        text_low = self._clean(text).lower()

        for pattern, canonical in rules:
            if re.search(pattern, text_low, flags=re.IGNORECASE):
                return canonical

        return None

    def _pick_candidate(self, raw_text: str, role: str) -> dict | None:
        result = self.matcher.find_best(raw_text, top_k=5, role=role)

        exact = result.get("exact_match")
        if exact:
            return {
                "raw": raw_text,
                "candidate": exact,
                "canonical_text": exact.get("name", raw_text),
                "source": "thesaurus_exact"
            }

        semantic = self.matcher._pick_semantic_candidate(result.get("similar", []), role=role)
        if semantic:
            return {
                "raw": raw_text,
                "candidate": semantic,
                "canonical_text": semantic.get("name", raw_text),
                "source": "thesaurus_semantic"
            }

        return None

    def _normalize_input(self, raw_input: str) -> dict:
        raw_input = self._clean(raw_input)

        alias = self._apply_alias_rules(raw_input, self.input_alias_rules)
        if alias:
            return {
                "raw": raw_input,
                "candidate": alias,
                "canonical_text": alias,
                "source": "alias"
            }

        picked = self._pick_candidate(raw_input, role="input")
        if picked:
            return picked

        return {
            "raw": raw_input,
            "candidate": None,
            "canonical_text": raw_input,
            "source": "raw"
        }

    def _normalize_object(self, raw_object: str) -> dict | None:
        import re
        raw_object_clean = str(raw_object or "").strip()
        if not raw_object_clean:
            return None

        has_formula = bool(re.search(
            r"\b(Fe|Co|Ni|Si|B|P|Mo|Cr|Zr|Ti|Al|Cu|Mn|Nb|W|Gd|Dy|Ho|Au|Pd|Pt)\b",
            raw_object_clean
        ))
        if has_formula:
            return {
                "raw": raw_object_clean,
                "candidate": None,
                "canonical_text": raw_object_clean,
                "source": "raw_formula"
            }

        alias = self._apply_alias_rules(raw_object_clean, self.object_alias_rules)
        if alias:
            return {
                "raw": raw_object_clean,
                "candidate": alias,
                "canonical_text": alias,
                "source": "alias"
            }

        picked = self._pick_candidate(raw_object_clean, role="object")
        if picked:
            return picked

        return {
            "raw": raw_object_clean,
            "candidate": None,
            "canonical_text": raw_object_clean,
            "source": "raw"
        }

    def _normalize_output(self, raw_output: str) -> dict:
        raw_output = self._clean(raw_output)

        alias = self._apply_alias_rules(raw_output, self.output_alias_rules)
        if alias:
            return {
                "raw": raw_output,
                "candidate": alias,
                "canonical_text": alias,
                "source": "alias"
            }

        picked = self._pick_candidate(raw_output, role="output")
        if picked:
            return picked

        return {
            "raw": raw_output,
            "candidate": None,
            "canonical_text": raw_output,
            "source": "raw"
        }

    def normalize_raw_result(self, raw_result: dict) -> dict:
        raw_inputs = self._normalize_to_list(raw_result.get("raw_inputs", []))
        raw_object = self._clean(raw_result.get("raw_object", ""))
        raw_outputs = self._normalize_to_list(raw_result.get("raw_outputs", []))

        return {
            "inputs": [self._normalize_input(x) for x in raw_inputs],
            "object": self._normalize_object(raw_object),
            "outputs": [self._normalize_output(x) for x in raw_outputs],
        }
