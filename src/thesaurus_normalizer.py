"""
Нормализация сырых полей физического эффекта через тезаурус: применение
алиас-правил для типовых воздействий и явлений, а также семантический поиск
канонических терминов для входов, объекта и выходов.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
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
            (r"\bферромагнитн\w*\s+(?:материал\w*|веществ\w*|сплав\w*)\b", "Ферромагнитный материал"),
            (r"\bаморфн\w*\s+сплав\w*\b", "Аморфные сплавы"),
            (r"\bметалл\w*\b", "Металлы"),
            (r"\bжидкост\w*\b", "Жидкость"),
            (r"\bгаз\w*\b", "Газ"),
            (r"\bкристалл\w*\b", "Кристалл"),
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

    @staticmethod
    def _extract_formula_phrase(text: str) -> str:
        formula_tokens = re.findall(
            r"\b(?:Fe|Co|Ni|Si|B|P|Mo|Cr|Zr|Ti|Al|Cu|Mn|Nb|W|Gd|Dy|Ho|Au|Pd|Pt)"
            r"(?:\([^)]{1,20}\)|\d+(?:[,.]\d+)?)?",
            text
        )
        if not formula_tokens:
            return ""

        formula = "".join(formula_tokens)
        text_low = text.lower()
        if "сплав" in text_low:
            return f"сплав {formula}"
        if "порош" in text_low:
            return f"порошок {formula}"
        return formula

    def _object_search_terms(self, text: str) -> list[str]:
        text = self._clean(text)
        if not text:
            return []

        candidates = [text]
        formula_phrase = self._extract_formula_phrase(text)
        if formula_phrase:
            candidates.append(formula_phrase)

        chunks = re.split(r"[.;]|\s+-\s+|\s+—\s+|\s+–\s+", text)
        object_keywords = (
            "материал", "веществ", "сред", "тело", "металл", "газ", "жидк",
            "кристалл", "сплав", "порош", "проводник", "диэлектрик", "полупроводник"
        )
        for chunk in chunks:
            chunk = self._clean(chunk)
            if not chunk:
                continue
            parts = [self._clean(x) for x in re.split(r",|\s+и\s+", chunk) if self._clean(x)]
            for part in [chunk, *parts]:
                part_low = part.lower()
                if any(k in part_low for k in object_keywords) or self._extract_formula_phrase(part):
                    words = part.split()
                    if len(words) <= 8:
                        candidates.append(part)

        result = []
        seen = set()
        for candidate in candidates:
            key = candidate.lower()
            if key not in seen:
                result.append(candidate)
                seen.add(key)
        return result

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
        raw_object_clean = str(raw_object or "").strip()
        if not raw_object_clean:
            return None

        alias = self._apply_alias_rules(raw_object_clean, self.object_alias_rules)
        if alias:
            return {
                "raw": raw_object_clean,
                "candidate": alias,
                "canonical_text": alias,
                "source": "alias"
            }

        for candidate_text in self._object_search_terms(raw_object_clean):
            picked = self._pick_candidate(candidate_text, role="object")
            if picked:
                picked["raw"] = raw_object_clean
                return picked

        formula_phrase = self._extract_formula_phrase(raw_object_clean)
        if formula_phrase:
            return {
                "raw": raw_object_clean,
                "candidate": None,
                "canonical_text": formula_phrase,
                "source": "raw_formula"
            }

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
