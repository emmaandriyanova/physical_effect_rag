import re


class Verifier:
    def __init__(self):
        self.condition_markers = [
            "при ",
            "в случае ",
            "если ",
            "когда ",
            "после ",
            "до ",
            "во время ",
            "в условиях ",
        ]

        self.empty_values = {"", "-", "нет", "none", "null", "[]"}

    @staticmethod
    def _clean(value: str) -> str:
        if value is None:
            return ""
        value = str(value).strip()
        value = re.sub(r"\s+", " ", value)
        return value

    def _is_empty(self, value: str) -> bool:
        return self._clean(value).lower() in self.empty_values

    def _looks_like_condition(self, value: str) -> bool:
        text = self._clean(value).lower()
        return any(marker in text for marker in self.condition_markers)

    def _count_semantic_parts(self, value: str) -> int:
        text = self._clean(value)
        if not text:
            return 0

        parts = re.split(r"[.;]", text)
        parts = [p.strip() for p in parts if p.strip()]
        return len(parts)

    def verify(self, normalized_result: dict) -> dict:
        input_params = self._clean(normalized_result.get("input_params", ""))
        effect_object = self._clean(normalized_result.get("object", ""))
        output_params = self._clean(normalized_result.get("output_params", ""))

        issues = []
        warnings = []

        if self._is_empty(input_params):
            issues.append("пустое поле input_params")

        if self._is_empty(effect_object):
            issues.append("пустое поле object")

        if self._is_empty(output_params):
            issues.append("пустое поле output_params")

        if input_params and self._looks_like_condition(input_params):
            warnings.append("input_params содержит условие, а не только сущность")

        if output_params and self._looks_like_condition(output_params):
            warnings.append("output_params содержит условие, а не только сущность")

        if effect_object and self._looks_like_condition(effect_object):
            warnings.append("object содержит условие, а не только объект")

        if input_params and len(input_params) < 3:
            warnings.append("слишком короткое поле input_params")

        if effect_object and len(effect_object) < 2:
            warnings.append("слишком короткое поле object")

        if output_params and len(output_params) < 3:
            warnings.append("слишком короткое поле output_params")

        if input_params and input_params == output_params:
            warnings.append("input_params и output_params совпадают дословно")

        if effect_object and effect_object == input_params:
            warnings.append("object и input_params совпадают дословно")

        if effect_object and effect_object == output_params:
            warnings.append("object и output_params совпадают дословно")

        input_parts = self._count_semantic_parts(input_params)
        output_parts = self._count_semantic_parts(output_params)

        result = {
            "status": "ok" if not issues else "error",
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "result": {
                "input_params": input_params,
                "object": effect_object,
                "output_params": output_params
            },
            "stats": {
                "input_parts": input_parts,
                "output_parts": output_parts
            }
        }

        return result
