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

    def build_prompts(self, query_text: str, examples_context: str, candidate_bundle: dict | None):

        few_shot = (
            "Вот примеры правильного извлечения из текстов этого домена.\n"
            "Изучи их внимательно — они показывают нужный формат и логику.\n\n"

            "--- ПРИМЕР 1 ---\n"
            "ТЕКСТ: Полные потери энергии состоят из статических гистерезисных потерь, классических и потерь на вихревые токи. "
            "Величина потерь зависит от частоты перемагничивания, удельного электрического сопротивления, магнитной проницаемости сплава и напряженности магнитного поля. "
            "В аморфных материалах потери растут с увеличением частоты.\n"
            "ПРАВИЛЬНЫЙ ОТВЕТ:\n"
            '{"raw_inputs": ["Магнитное поле"],'
            ' "raw_object": "Аморфный ферромагнитный сплав",'
            ' "raw_outputs": ["Удельные потери энергии (объемная плотность теплового потока) (Вт/м**3)"],'
            ' "input_modifiers": ["переменное", "увеличение частоты перемагничивания (Гц)"],'
            ' "output_modifiers": ["увеличение"],'
            ' "conditions": ["зависит от удельного электрического сопротивления и магнитной проницаемости сплава"]}\n'
            "ПОЯСНЕНИЕ: хотя текст перечисляет много факторов ('зависит от A, B, C'), "
            "главное внешнее воздействие — Магнитное поле. Частота — его характеристика, идёт в input_modifiers. "
            "Выход — конкретная физическая величина с единицами.\n\n"

            "--- ПРИМЕР 2 ---\n"
            "ТЕКСТ: Потери мощности зависят от объемной доли, размера и характера распределения кристаллов альфа-Fe. "
            "При низких частотах кристаллическая фаза образуется на поверхности и потери увеличиваются.\n"
            "ПРАВИЛЬНЫЙ ОТВЕТ:\n"
            '{"raw_inputs": ["Объемная доля кристаллической фазы в сплаве (безразмерная величина)"],'
            ' "raw_object": "Сплав на основе железа типа Fe-Ni-Mo-B-Si",'
            ' "raw_outputs": ["Удельные потери энергии (объемная плотность теплового потока) (Вт/м**3)"],'
            ' "input_modifiers": ["увеличение от 0 до 0,01"],'
            ' "output_modifiers": ["уменьшение"],'
            ' "conditions": ["при низких частотах потери увеличиваются", "частичная кристаллизация не более 1%"]}\n'
            "ПОЯСНЕНИЕ: объемная доля кристаллической фазы — это главная независимая переменная. "
            "Выход — та же физическая величина что изменяется: удельные потери энергии с единицами.\n\n"

            "--- ПРИМЕР 3 ---\n"
            "ТЕКСТ: В сплавах на основе Fe при температурах ниже температуры Кюри проявляется аномалия теплового расширения. "
            "У сплава Fe(83)B(17) обнаружены инварные свойства: в интервале температур 273-573 К "
            "температурный коэффициент линейного расширения близок к 0.\n"
            "ПРАВИЛЬНЫЙ ОТВЕТ:\n"
            '{"raw_inputs": ["Температура (К)"],'
            ' "raw_object": "Аморфный сплав, например Fe(80)P(13)C(7), Fe(79)Si(10)B(11), Fe(87)P(13) и др.",'
            ' "raw_outputs": ["Температурный коэффициент линейного расширения (К**-1)"],'
            ' "input_modifiers": ["увеличение от 50 до 900 К"],'
            ' "output_modifiers": ["немонотонное изменение"],'
            ' "conditions": ["ниже температуры Кюри", "у Fe(83)B(17) ТКЛР близок к 0 в диапазоне 273-573 К"]}\n'
            "ПОЯСНЕНИЕ: температура — входная переменная. "
            "Выход — температурный коэффициент с единицами (К**-1), не 'длина образцов'.\n\n"

            "--- ПРИМЕР 4 ---\n"
            "ТЕКСТ: При закалке из расплава определяющими величинами для скорости охлаждения являются "
            "коэффициент теплопередачи и толщина слоя расплава. "
            "При увеличении коэффициента теплопередачи скорость охлаждения увеличивается.\n"
            "ПРАВИЛЬНЫЙ ОТВЕТ:\n"
            '{"raw_inputs": ["Коэффициент теплопередачи расплава (Вт/(м**2*К))"],'
            ' "raw_object": "Жидкий расплав металла на металлической подложке, например Fe(40)Ni(40)P(14)B(6) и др.",'
            ' "raw_outputs": ["Скорость охлаждения расплава (К/с)"],'
            ' "input_modifiers": ["увеличение"],'
            ' "output_modifiers": ["увеличение"],'
            ' "conditions": ["толщина слоя расплава также влияет на скорость охлаждения"]}\n'
            "ПОЯСНЕНИЕ: хотя упомянуты два фактора (коэффициент и толщина), "
            "главное воздействие — коэффициент теплопередачи (он в названии эффекта). "
            "Выход — скорость охлаждения с единицами (К/с).\n\n"
            
            "--- ПРИМЕР 5 (ДВА НЕЗАВИСИМЫХ ВХОДА) ---\n"
            "ТЕКСТ: Параэлектрический резонанс - резонансное поглощение радиоволн, связанное с перемещением "
            "электрических дипольных моментов частиц вещества во внешних электрических полях. "
            "Если кристалл находится в переменном электрическом поле, то возникают переходы на определенной "
            "частоте, соответствующей разности уровней энергии. Характерная область частот параэлектрического "
            "резонанса диапазон СВЧ (10^10 - 10^11 Гц). Например, наблюдается при Т < 10 K в KCl с примесью "
            "ионов OH(-), CN(-), Li(+).\n"
            "ПРАВИЛЬНЫЙ ОТВЕТ:\n"
            '{"raw_inputs": ["Электромагнитное излучение. Радиоволны (СВЧ). Спектр. Частота колебаний (Гц)",'
            ' "Электрическое поле. Постоянное. Напряженность электрического поля (В/м)"],'
            ' "raw_object": "Параэлектрик, например кристалл KCl, содержащий примесные ионы OH(-), CN(-), нецентральные ионы Li(+); Fe в кристалле Si",'
            ' "raw_outputs": ["Коэффициент поглощения (безразмерная величина)"],'
            ' "input_modifiers": ["увеличение от 10^10 до 10^11 Гц"],'
            ' "output_modifiers": ["увеличение на резонансной частоте"],'
            ' "conditions": ["при Т < 10 K"]}\n'
            "ПОЯСНЕНИЕ: ДВА разных независимых воздействия одновременно — радиоволны И электрическое поле. "
            "Поэтому raw_inputs содержит ДВА элемента списка. Это не одно поле с характеристиками, а две "
            "разные физические сущности.\n\n"

            "--- ПРИМЕР 6 (ОДИН ВХОД, ВЫХОД — ФИЗВЕЛИЧИНА С НАПРАВЛЕНИЕМ) ---\n"
            "ТЕКСТ: Возникновение разности потенциалов между концами капилляра или поверхностями пористой "
            "перегородки при продавливании через нее жидкости. Разность потенциалов прямо пропорциональна "
            "давлению.\n"
            "ПРАВИЛЬНЫЙ ОТВЕТ:\n"
            '{"raw_inputs": ["Давление (Па)"],'
            ' "raw_object": "Пористая перегородка (диафрагма), через которую под давлением протекает жидкость",'
            ' "raw_outputs": ["Разность потенциалов (В)"],'
            ' "input_modifiers": ["увеличение"],'
            ' "output_modifiers": ["увеличение"],'
            ' "conditions": []}\n'
            "ПОЯСНЕНИЕ: один простой случай — одно воздействие, одна физвеличина на выходе. "
            "Объект описан конкретно (пористая перегородка с протекающей жидкостью), а не общо ('пористое тело').\n\n"

            "--- ПРИМЕР 7 (ВХОД С УМЕНЬШЕНИЕМ) ---\n"
            "ТЕКСТ: В сегнетополупроводниках в определенном температурном интервале существует спонтанная "
            "поляризация. Она возникает при фазовом переходе из неполярной параэлектрической фазы в полярную "
            "сегнетоэлектрическую при температуре Кюри. С уменьшением температуры от температуры Кюри "
            "поляризованность сегнетополупроводника нелинейно растет и при некотором значении Т достигает "
            "насыщения.\n"
            "ПРАВИЛЬНЫЙ ОТВЕТ:\n"
            '{"raw_inputs": ["Температура сегнетополупроводника (К)"],'
            ' "raw_object": "Монодоменный сегнетополупроводник, например GeTe, SnTe, TbTe, твердый раствор на их основе, LiNbO(3), PbTiO(3)",'
            ' "raw_outputs": ["Спонтанная поляризованность (Кл/м**2)"],'
            ' "input_modifiers": ["уменьшение от температуры Кюри"],'
            ' "output_modifiers": ["увеличение от 0 до насыщения"],'
            ' "conditions": ["возникает при фазовом переходе при температуре Кюри"]}\n'
            "ПОЯСНЕНИЕ: input_modifiers МОЖЕТ быть 'уменьшение' — направление зависит от того, что описано "
            "в тексте, а не всегда 'увеличение'. Объект — конкретный материал со списком соединений.\n\n"
        
            "КЛЮЧЕВЫЕ ПРАВИЛА из примеров:\n"
            "1. Если текст говорит 'зависит от A, B, C' — выбери ОДНО главное воздействие, остальное в conditions.\n"
            "2. Выход — всегда конкретная физическая величина С единицами если они упомянуты в тексте: (Вт/м**3), (К/с), (К**-1), (м**2/с).\n"
            "3. Объект — конкретный материал с составом: не 'Металлы', не 'Ферромагнетик', а 'Аморфный сплав Fe-Ni-Mo-B-Si'.\n"
            "4. Направление изменения (увеличение/уменьшение/замедление/ускорение) → input_modifiers или output_modifiers, НЕ в raw_inputs и НЕ в raw_outputs.\n"
            "5. raw_outputs НИКОГДА не должен быть пустым. Если в тексте 'X замедляется/ускоряется' → выход = X (физвеличина с единицами), а слова 'замедление'/'ускорение' → output_modifiers.\n"
            "6. ВХОД vs СЛЕДСТВИЕ: 'электрический разряд', 'нагрев образца', 'плавление' — это процессы/следствия, НЕ входы. Ищи первопричину: электрическое поле, температура, давление, магнитное поле и т.п.\n"
            "7. ОДИН вход или НЕСКОЛЬКО ВХОДОВ. Физический эффект — это связь между воздействиями "
            "и наблюдаемым результатом. Воздействий может быть несколько, и важно их все выявить.\n"
            "ВАЖНО: разные ВХОДЫ — это разные физические сущности (поле, температура, давление, поток, "
            "излучение). Свойства одной сущности (тип, частота, амплитуда, направление изменения) — это "
            "не отдельные входы, а характеристики того же входа, и идут в input_modifiers.\n"
            "Например: 'Температура (К). Увеличение' = ОДИН вход (температура), 'увеличение' это "
            "характеристика. 'Магнитное поле. Переменное. Частота...' = ОДИН вход (поле), 'переменное' "
            "и 'частота' — характеристики того же поля.\n"
            "ОДИН вход — когда в тексте действует одно физическое явление с разными характеристиками.\n"
            "НЕСКОЛЬКО входов — когда:\n"
            "   - в системе одновременно действуют разные физические явления (поле и излучение, "
            "температура и давление, поле и поток вещества);\n"
            "   - помимо изменяемого параметра в системе присутствует управляющее или поддерживающее "
            "воздействие (например, поле создаёт условия, в которых параметр играет роль);\n"
            "   - объект находится в среде, и сама среда оказывает воздействие (поле, поток, "
            "температурные условия среды).\n"
            "Все независимые физические воздействия должны быть включены в raw_inputs как отдельные элементы.\n\n"
        )

        system_prompt = (
            few_shot +
            "Теперь извлеки структуру физического эффекта из нового текста.\n\n"
            "Физический эффект описывается тремя компонентами:\n"
            "A (вход) — главная независимая переменная или внешнее воздействие;\n"
            "B (объект) — конкретный материал, вещество или среда;\n"
            "C (выход) — измеримая физическая величина с единицами или чётко названный процесс.\n\n"
            "Верни строго один JSON-объект:\n"
            "- raw_inputs: ОДНО главное воздействие (не перечисляй все факторы)\n"
            "- raw_object: конкретный материал с составом\n"
            "- raw_outputs: физическая величина с единицами если есть в тексте\n"
            "- input_modifiers: направление изменения и качественные характеристики входа\n"
            "- output_modifiers: направление изменения выхода\n"
            "- conditions: диапазоны, вторичные факторы, побочные эффекты\n"
            "- отвечай строго одним JSON-объектом и больше ничем\n"
            "- не придумывай сущности которых нет в тексте\n"
        )

        candidate_text = "Нет кандидатов."
        if candidate_bundle:
            candidate_text = (
                f"Кандидаты на входы: {candidate_bundle.get('candidate_inputs', [])}\n"
                f"Кандидат на объект: {candidate_bundle.get('candidate_object', '')}\n"
                f"Кандидаты на выходы: {candidate_bundle.get('candidate_outputs', [])}\n"
                f"Кандидаты на условия: {candidate_bundle.get('candidate_conditions', [])}\n"
                f"Кандидаты на input_modifiers: {candidate_bundle.get('candidate_input_modifiers', [])}\n"
                f"Кандидаты на output_modifiers: {candidate_bundle.get('candidate_output_modifiers', [])}\n"
            )

        user_prompt = (
            f"Исходный текст:\n{query_text}\n\n"
            f"Похожие примеры из базы:\n{examples_context}\n\n"
            f"Предварительно найденные кандидаты:\n{candidate_text}\n\n"
            "Верни только JSON такого вида:\n"
            "{\n"
            '  "raw_inputs": ["..."],\n'
            '  "raw_object": "...",\n'
            '  "raw_outputs": ["..."],\n'
            '  "input_modifiers": ["...", "..."],\n'
            '  "output_modifiers": ["...", "..."],\n'
            '  "conditions": ["...", "..."]\n'
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
            if "raw_inputs" in c or "raw_outputs" in c:
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

    def _postprocess_raw_result(self, result: dict, source_text: str, candidate_bundle: dict | None = None) -> dict:
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

        if candidate_bundle:
            cand_object = str(candidate_bundle.get("candidate_object", "")).strip()
            cand_conditions = clean_list(candidate_bundle.get("candidate_conditions", []))

            if not raw_object and cand_object:
                raw_object = cand_object
            conditions = clean_list(conditions + cand_conditions)

        return {
            "raw_inputs": raw_inputs,
            "raw_object": raw_object,
            "raw_outputs": raw_outputs,
            "input_modifiers": input_modifiers,
            "output_modifiers": output_modifiers,
            "conditions": conditions
        }

    def extract(self, query_text: str, retrieved_examples: list[dict], candidate_bundle: dict | None = None) -> dict:
        examples_context = self.build_examples_context(retrieved_examples)
        system_prompt, user_prompt = self.build_prompts(query_text, examples_context, candidate_bundle)

        lm_response = self.lm_client.chat_optional(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=2000
        )

        if not lm_response:
            short_system = (
                "Извлеки структуру физического эффекта из текста. "
                "Верни JSON с полями: raw_inputs, raw_object, raw_outputs, "
                "input_modifiers, output_modifiers, conditions. "
                "raw_inputs — входное воздействие или параметр. "
                "raw_outputs — измеримый результат с единицами. "
                "Отвечай только JSON."
            )
            short_user = (
                f"Текст:\n{query_text}\n\n"
                "Верни только JSON:\n"
                '{"raw_inputs": ["..."], "raw_object": "...", "raw_outputs": ["..."], '
                '"input_modifiers": ["..."], "output_modifiers": ["..."], "conditions": []}'
            )
            lm_response = self.lm_client.chat_optional(
                system_prompt=short_system,
                user_prompt=short_user,
                temperature=0.0,
                max_tokens=1000
            )

        if not lm_response:
            minimal_system = "Отвечай только JSON. Никакого текста кроме JSON."
            minimal_user = (
                f"Текст: {query_text[:2000]}\n\n"
                "Найди: 1) что воздействует 2) на что 3) что меняется.\n"
                '{"raw_inputs":["воздействие"],"raw_object":"объект","raw_outputs":["результат"],'
                '"input_modifiers":[],"output_modifiers":[],"conditions":[]}'
            )
            lm_response = self.lm_client.chat_optional(
                system_prompt=minimal_system,
                user_prompt=minimal_user,
                temperature=0.0,
                max_tokens=500
            )

        if not lm_response:
            return {
                "status": "error",
                "message": "LM Studio не вернул ответ на этапе raw extraction"
            }

        parsed = self._extract_json_object(lm_response)

        if parsed is None:
            short_system = (
                "Извлеки структуру физического эффекта из текста. "
                "Верни JSON с полями: raw_inputs, raw_object, raw_outputs, "
                "input_modifiers, output_modifiers, conditions. "
                "Отвечай только JSON без пояснений."
            )
            short_user = (
                f"Текст:\n{query_text}\n\n"
                'Верни только JSON:\n{"raw_inputs": ["..."], "raw_object": "...", "raw_outputs": ["..."], '
                '"input_modifiers": [], "output_modifiers": [], "conditions": []}'
            )
            lm_response = self.lm_client.chat_optional(
                system_prompt=short_system,
                user_prompt=short_user,
                temperature=0.0,
                max_tokens=1000
            )
            if lm_response:
                parsed = self._extract_json_object(lm_response)

        if parsed is None:
            minimal_system = "Отвечай только JSON. Никакого текста кроме JSON."
            minimal_user = (
                f"Текст: {query_text[:500]}\n\n"
                "Найди: 1) что воздействует 2) на что 3) что меняется.\n"
                '{"raw_inputs":["воздействие"],"raw_object":"объект","raw_outputs":["результат"],'
                '"input_modifiers":[],"output_modifiers":[],"conditions":[]}'
            )
            lm_response = self.lm_client.chat_optional(
                system_prompt=minimal_system,
                user_prompt=minimal_user,
                temperature=0.0,
                max_tokens=500
            )
            if lm_response:
                parsed = self._extract_json_object(lm_response)

        if parsed is None:
            return {
                "status": "error",
                "message": "Ответ модели не удалось распарсить как JSON",
                "raw_response": lm_response
            }

        raw_result = {
            "raw_inputs": self._normalize_list(parsed.get("raw_inputs", [])),
            "raw_object": str(parsed.get("raw_object", "")).strip(),
            "raw_outputs": self._normalize_list(parsed.get("raw_outputs", [])),
            "input_modifiers": self._normalize_list(parsed.get("input_modifiers", [])),
            "output_modifiers": self._normalize_list(parsed.get("output_modifiers", [])),
            "conditions": self._normalize_list(parsed.get("conditions", []))
        }

        raw_result = self._postprocess_raw_result(raw_result, query_text, candidate_bundle)

        return {
            "status": "ok",
            "result": raw_result
        }

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        text = str(text or "").lower()
        text = text.replace("ё", "е")
        text = re.sub(r"[^а-яa-z0-9\s-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _filter_modifiers_by_source_text(self, items: list[str], source_text: str) -> list[str]:
        source_norm = self._normalize_for_match(source_text)
        result = []

        for item in items:
            item_norm = self._normalize_for_match(item)
            if not item_norm:
                continue

            if item_norm in source_norm:
                result.append(item)
                continue

            words = [w for w in item_norm.split() if len(w) >= 4]
            if words and all(w in source_norm for w in words):
                result.append(item)

        unique = []
        seen = set()
        for x in result:
            key = x.strip().lower()
            if key not in seen:
                unique.append(x)
                seen.add(key)

        return unique
