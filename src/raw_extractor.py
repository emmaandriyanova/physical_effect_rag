"""
Извлечение структуры физического эффекта (вход / объект / выход) из текста
патента с помощью дообученной языковой модели через LM Studio API.
Формирует промт с RAG-примером, разбирает JSON-ответ модели и выполняет
первичную постобработку извлечённых полей.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import json
import logging
import re

from lm_studio_client import LMStudioClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class RawExtractor:
    def __init__(
        self,
        lm_studio_url: str = "http://79.170.162.18:1234/v1/chat/completions",
        model_id: str = "qwen3-8b"
    ):
        self.lm_client = LMStudioClient(
            base_url=lm_studio_url,
            model_id=model_id
        )

    @staticmethod
    def build_examples_context(retrieved_examples: list[dict]) -> str:
        if not retrieved_examples:
            return "нет найденных похожих примеров."

        parts = []
        for i, ex in enumerate(retrieved_examples[:3], start=1):
            parts.append(
                f"Пример {i}:\n"
                f"Эффект: {ex.get('effect_name', '')}\n"
                f"Описание: {ex.get('description', '')}\n"
                f"Вход: {ex.get('input_params', '')}\n"
                f"Объект: {ex.get('object', '')}\n"
                f"Выход: {ex.get('output_params', '')}\n"
            )

        return "\n".join(parts)

    def build_prompts(self, query_text: str, examples_context: str):
        """
        Формирует системный и пользовательский промты для LLM-извлечения.

        :param query_text: текст патента для анализа
        :param examples_context: отформатированные примеры из RAG-ретривала
        :return: кортеж (system_prompt, user_prompt)
        """

        system_prompt = (
            "Ты — эксперт по физике и анализу патентов. "
            "Твоя задача — извлечь структуру физического эффекта из текста патента "
            "в формате базы данных FEText.\n\n"
            "Физический эффект: входное физическое воздействие → объект → выходная физическая величина.\n\n"

            "ФОРМАТ ПОЛЕЙ:\n\n"

            "input_params — входное физическое воздействие. Структура записи:\n"
            "  [Вид воздействия]. [Характеристика]. [Название физической величины] ([единицы]).\n"
            "  Если входов несколько — формат: Вход1:[запись1] Вход2:[запись2]\n\n"

            "object — физический объект или среда, на которую оказывается воздействие.\n"
            "  [Тип материала или среды], [конкретизация если принципиальна].\n\n"

            "output_params — выходная физическая величина или явление. Структура записи:\n"
            "  [Название величины или явления] ([единицы]). [Направление изменения].\n\n"

            "ПРАВИЛА ИЗВЛЕЧЕНИЯ:\n"
            "- Единицы измерения — обязательная часть записи, указывай в скобках\n"
            "- Характеристику воздействия включай если она принципиальна для эффекта "
            "(переменное/постоянное, линейно-поляризованное, монохроматическое и т.п.)\n"
            "- Направление изменения выхода (Увеличение/Уменьшение) включай если явно указано\n"
            "- Извлекай физическую суть эффекта, а не описание прибора или установки\n"
            "- Если в тексте упоминается известный физический эффект (эффект Фарадея, "
            "Керра, скин-эффект и т.д.) — извлекай именно его структуру\n"
            "- Вход и выход должны быть физически причинно-следственно связаны\n"
            "- Ориентируйся на разделы «технический результат» и «принцип действия», "
            "а не на описание конструкции\n"
            "- НЕ включай: названия приборов, алгоритмы обработки сигнала, "
            "описание экспериментальной установки\n"
            "- Пример из базы приведён только для понимания формата записи — "
            "НЕ копируй его содержание, извлекай из текста патента\n"
            "- Пиши слова раздельно, не сливай их: «Энергия фотона», а не «Энергияфотона»\n"
            "- Перед скобками с единицами всегда пробел: «Длина волны (нм)», а не «Длина волны(нм)»\n"
            "- После точки внутри строки — заглавная буква: «Излучение. Монохроматическое.»\n"
            "- Верни строго один JSON-объект, без пояснений, без markdown\n"
        )

        user_prompt = (
            "ПРИМЕР ФОРМАТА ЗАПИСИ из базы физических эффектов\n"
            "(только для понимания структуры — НЕ копируй, содержание извлекай из текста патента):\n"
            f"{examples_context}\n\n"
            "---\n\n"
            "ЗАДАЧА — извлеки физический эффект из текста патента ниже:\n\n"
            f"{query_text}\n\n"
            "Верни JSON только для данного текста патента:\n"
            "{\n"
            '  "input_params": "...",\n'
            '  "object": "...",\n'
            '  "output_params": "..."\n'
            "}\n"
        )
        return system_prompt, user_prompt

    @staticmethod
    def _extract_json_object(text: str) -> dict | None:
        if not text:
            return None

        text = text.strip()
        text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)
        text = text.strip()

        candidates = []
        start = 0
        while True:
            pos = text.find("{", start)
            if pos == -1:
                break
            brace_count = 0
            end = -1
            for i in range(pos, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break
            if end != -1:
                candidate = text[pos:end + 1].strip()
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        candidates.append(parsed)
                except json.JSONDecodeError:
                    try:
                        parsed = json.loads(candidate.replace("'", '"'))
                        if isinstance(parsed, dict):
                            candidates.append(parsed)
                    except json.JSONDecodeError:
                        pass
            start = pos + 1

        for c in candidates:
            if "raw_inputs" in c or "raw_outputs" in c or "input_params" in c or "output_params" in c:
                return c

        if candidates:
            return candidates[0]

        return None

    @staticmethod
    def _normalize_list(value) -> list[str]:
        if value is None:
            return []

        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]

        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []

        return [str(value).strip()]

    def _postprocess_raw_result(self, result: dict) -> dict:
        def clean_list(values):
            out = []
            seen = set()
            for v in values or []:
                text = str(v).strip()
                if not text:
                    continue
                key = text.lower()
                if key not in seen:
                    out.append(text)
                    seen.add(key)
            return out

        raw_inputs = clean_list(result.get("raw_inputs", []))
        raw_outputs = clean_list(result.get("raw_outputs", []))
        input_modifiers = clean_list(result.get("input_modifiers", []))
        output_modifiers = clean_list(result.get("output_modifiers", []))
        conditions = clean_list(result.get("conditions", []))
        raw_object = str(result.get("raw_object", "")).strip()

        bad_output_words = {"изменение", "образование", "происхождение", "процесс", "эффект"}
        raw_outputs = [x for x in raw_outputs if x.lower() not in bad_output_words]

        return {
            "raw_inputs": raw_inputs,
            "raw_object": raw_object,
            "raw_outputs": raw_outputs,
            "input_modifiers": input_modifiers,
            "output_modifiers": output_modifiers,
            "conditions": conditions
        }

    def extract(self, query_text: str, retrieved_examples: list[dict]) -> dict:
        """
        Извлекает структуру физического эффекта из текста патента через LLM.

        :param query_text: текст патента для анализа
        :param retrieved_examples: список примеров из базы физических эффектов для RAG-контекста
        :return: словарь со статусом и полями raw_inputs, raw_object, raw_outputs,
                 либо словарь с ошибкой и raw_response при неудаче
        """
        examples_context = self.build_examples_context(retrieved_examples)
        system_prompt, user_prompt = self.build_prompts(query_text, examples_context)

        lm_response = self.lm_client.chat_optional(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=2000
        )

        if not lm_response:
            return {
                "status": "error",
                "message": "LM Studio не вернул ответ"
            }

        parsed = self._extract_json_object(lm_response)

        if parsed is None:
            return {
                "status": "error",
                "message": "Ответ модели не удалось распарсить как JSON",
                "raw_response": lm_response
            }

        input_val  = str(parsed.get("input_params",  "") or "").strip()
        object_val = str(parsed.get("object",        "") or "").strip()
        output_val = str(parsed.get("output_params", "") or "").strip()

        if re.search(r'Вход\d+[:.]\s*', input_val):
            raw_inputs = [p.strip() for p in re.split(r'Вход\d+[:.]\s*', input_val) if p.strip()]
        else:
            raw_inputs = [input_val] if input_val else []

        raw_result = {
            "raw_inputs":       raw_inputs,
            "raw_object":       object_val,
            "raw_outputs":      [output_val] if output_val else [],
            "input_modifiers":  [],
            "output_modifiers": [],
            "conditions":       []
        }

        raw_result = self._postprocess_raw_result(raw_result)

        logging.info(f"raw_result перед возвратом: {raw_result}")
        return {
            "status": "ok",
            "result": raw_result
        }

