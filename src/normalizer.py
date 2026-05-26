import re


class Normalizer:
    def __init__(
        self,
        lm_studio_url: str = "",
        model_id: str = ""
    ):
        pass

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
    def _dedup(items: list[str]) -> list[str]:
        result = []
        seen = set()
        for item in items:
            item = str(item).strip()
            if not item:
                continue
            key = item.lower()
            if key not in seen:
                result.append(item)
                seen.add(key)
        return result

    @staticmethod
    def _canonicalize_input_for_dedup(text: str) -> str:
        t = str(text or "").strip().lower()
        t = re.sub(r"\s+", " ", t)
        if "магнит" in t and "пол" in t:
            return "магнитное поле"
        if "электр" in t and "пол" in t:
            return "электрическое поле"
        if "лазер" in t and "излуч" in t:
            return "лазерное излучение"
        if "ультразв" in t:
            return "ультразвук"
        if "давлен" in t:
            return "давление"
        if (
            "механическ" in t
            or "силов" in t
            or "нагруж" in t
            or "нагруз" in t
            or "растяж" in t
            or "сжат" in t
            or "сдвиг" in t
            or "изгиб" in t
        ):
            return "механическое воздействие"
        return t

    @staticmethod
    def _canonicalize_output_modifier(mod: str) -> str:
        mod_low = mod.lower().strip()
        if any(x in mod_low for x in ["повыш", "увелич", "возраст", "усилен", "ускорен"]):
            return "Увеличение"
        if any(x in mod_low for x in ["сниж", "уменьш", "пада", "ослаб", "замедлен"]):
            return "Уменьшение"
        generic_patterns = [
            r"^изменени\w*$",
            r"^образовани\w*$",
            r"^возникновени\w*$",
            r"^происхождени\w*$",
            r"^наличие$",
            r"^процесс$",
            r"^эффект$",
        ]
        if any(re.match(pattern, mod_low) for pattern in generic_patterns):
            return ""
        return mod.capitalize() if mod else ""

    @staticmethod
    def _is_parameter_like(text: str) -> bool:
        text_low = text.lower()

        parameter_patterns = [
            r"\bдлительност\w+\s+импульс\w*\b",
            r"\bинтенсивност\w+\s+лазер\w*\b",
            r"\bмощност\w+\s+лазер\w*\b",
            r"\bплотност\w+\s+энерги\w*\b",
            r"\bамплитуд\w+\s+колебан\w*\b",
        ]
        return any(re.search(p, text_low) for p in parameter_patterns)

    @staticmethod
    def _is_condition_like(text: str) -> bool:
        text_low = text.lower()
        condition_markers = [
            "при ",
            "в зависимости",
            "после ",
            "до ",
            "если ",
            "в условиях ",
            "в области ",
            "определяется ",
            "зависит ",
            "может ",
            "возможно ",
            "например",
        ]
        return any(marker in text_low for marker in condition_markers)

    @staticmethod
    def _is_generic_process_word(text: str) -> bool:
        text_low = text.lower().strip()
        generic_patterns = [
            r"^изменени\w*$",
            r"^образовани\w*$",
            r"^возникновени\w*$",
            r"^происхождени\w*$",
            r"^наличие$",
            r"^процесс$",
            r"^эффект$",
        ]
        return any(re.match(pattern, text_low) for pattern in generic_patterns)

    @staticmethod
    def _looks_like_magnetic(text: str) -> bool:
        text = text.lower()
        return "магнит" in text and "пол" in text

    @staticmethod
    def _looks_like_electric(text: str) -> bool:
        text = text.lower()
        return "электр" in text and "пол" in text

    @staticmethod
    def _looks_like_laser(text: str) -> bool:
        text = text.lower()
        return "лазер" in text or "излуч" in text or "свет" in text


    def _normalize_output_core(self, text: str) -> str:
        t = str(text or "").strip()
        t_low = t.lower()

        if re.search(r"\([^)]{1,40}\)", t) and re.search(
                r"(па|вт|дж|ом|гц|кг|м\*\*|к\*\*|м/с|а/м|нм|мм|%|кл)", t_low
        ):
            result = t.strip().capitalize() if t else t
            return self._fix_units_case(result)

        if re.search(r"\bостаточн\w*\s+деформ\w*\b", t_low):
            return "Остаточная деформация"
        if re.search(r"\bпластическ\w*\s+деформ\w*\b", t_low):
            return "Пластическая деформация"
        if re.search(r"\bдеформир\w*\b|\bдеформирован\w*\b", t_low):
            return "Пластическая деформация"
        if re.search(r"\bразупрочнен\w*\b", t_low):
            return "Разупрочнение"
        if re.search(r"\bэрози\w*\b", t_low):
            return "Эрозия"
        if re.search(r"\bкавитац\w*\b", t_low):
            return "Кавитация"
        if re.search(r"\bзакалк\w*\b", t_low):
            return "Закалка"
        if re.search(r"\bизменени\w+\s+структур\w*\b|\bмикроструктур\w*\b", t_low):
            return "Изменение структуры"
        if re.search(r"\bнагрев\w*\b|\bнагрет\w*\b", t_low):
            return "Нагрев"
        if re.search(r"\bплавлен\w*\b", t_low):
            return "Плавление"
        if re.search(r"\bиспарен\w*\b", t_low):
            return "Испарение"
        if re.search(r"\bфазов\w*\s+переход\w*\b", t_low):
            return "Фазовый переход"
        if re.search(r"\bплазм\w*\b", t_low):
            return "Образование плазмы"
        if re.search(r"\bэлектрическ\w*\s+ток\b|\bток\b", t_low):
            return "Электрический ток"
        if re.search(r"\bполяризац\w*\b", t_low):
            return "Поляризация"
        if re.search(r"\bмагнитн\w+\s+проницаемост\w*\b", t_low):
            return "Магнитная проницаемость"
        if re.search(r"\bэлектрическ\w+\s+сопротивлен\w*\b", t_low):
            return "Электрическое сопротивление"
        if re.search(r"\bпроводимост\w*\b", t_low):
            return "Проводимость"
        if re.search(r"\bтемператур\w*\b", t_low):
            return "Температура"
        if re.search(r"\bдавлен\w*\b", t_low):
            return "Давление"

        result = t.strip() if t[0].isupper() else t.strip().capitalize()
        return self._fix_units_case(result)

    def _fix_units_case(self, text: str) -> str:
        fixes = [
            (r'\bвт\b', 'Вт'), (r'\bдж\b', 'Дж'), (r'\bпа\b', 'Па'),
            (r'\bгц\b', 'Гц'), (r'\bом\b', 'Ом'), (r'\bкг\b', 'кг'),
            (r'\bк\*\*', 'К**'), (r'\bм\*\*', 'м**'), (r'\bа/м', 'А/м'),
        ]
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _looks_like_valid_output(text: str) -> bool:
        t = str(text or "").strip().lower()
        if not t:
            return False
        bad_patterns = [
            r"ющ\.*$",
            r"ащ\.*$",
            r"ивш\.*$",
            r"ующ\.*$",
            r"вш\.*$",
        ]
        if any(re.search(p, t) for p in bad_patterns):
            return False
        if len(t) < 4:
            return False
        return True

    def _dedup_inputs_semantically(self, inputs: list[str]) -> list[str]:
        result = []
        seen = set()
        for item in inputs:
            clean_item = self._clean(item)
            if not clean_item:
                continue
            canon = self._canonicalize_input_for_dedup(clean_item)
            if canon not in seen:
                result.append(clean_item)
                seen.add(canon)
        return result

    @staticmethod
    def _remove_output_stage_chains(outputs: list[str], conditions: list[str]) -> list[str]:
        lowered = {x.lower(): x for x in outputs}
        has_melting = "плавление" in lowered
        has_phase = "фазовый переход" in lowered
        has_plasma = "образование плазмы" in lowered
        result = outputs[:]
        if has_melting and has_phase:
            result = [x for x in result if x.lower() != "фазовый переход"]
        cond_text = " ".join(conditions).lower()
        if has_plasma and ("испарен" in cond_text or "при дальнейшем увеличении энергии" in cond_text):
            result = [x for x in result if x.lower() != "образование плазмы"]
        return result

    @staticmethod
    def _can_attach_growth_modifier(outputs: list[str]) -> bool:
        if not outputs:
            return False
        allowed = {
            "Пластическая деформация", "Электрический ток", "Нагрев",
            "Разупрочнение", "Поляризация", "Намагниченность",
            "Электрическое смещение", "Кавитация", "Давление", "Смещение", "Движение",
        }
        return any(x in allowed for x in outputs)

    def _filter_inputs(self, raw_inputs: list[str], normalized_inputs: list[dict]) -> list[str]:
        result = []
        for idx, raw in enumerate(raw_inputs):
            raw = self._clean(raw)
            if not raw:
                continue

            raw_low = raw.lower()
            if re.match(r"^изменение\s+параметр\w*", raw_low):
                continue
            if re.match(r"^параметрическ\w+", raw_low):
                continue
            if re.match(r"^(увеличение|уменьшение|снижение|повышение|рост)\b", raw_low):
                continue

            if self._is_parameter_like(raw):
                continue
            if self._is_condition_like(raw):
                continue


            candidate = self._cap(raw)
            candidate = self._fix_units_case(candidate)
            result.append(candidate)


        return self._dedup_inputs_semantically(result)

    def _filter_outputs(self, raw_outputs: list[str], normalized_outputs: list[dict], conditions: list[str]) -> list[str]:
        result = []
        for idx, raw in enumerate(raw_outputs):
            raw = self._clean(raw)
            if not raw:
                continue

            if self._is_parameter_like(raw):
                continue
            if self._is_condition_like(raw):
                continue
            if self._is_generic_process_word(raw):
                continue

            candidate = raw

            candidate = self._normalize_output_core(candidate)

            if not self._looks_like_valid_output(candidate):
                continue

            result.append(candidate)

        result = self._dedup(result)
        result = self._remove_output_stage_chains(result, conditions)
        result = self._dedup(result)

        if result:
            return result

        fallback = []
        for raw in raw_outputs:
            raw = self._clean(raw)
            if not raw:
                continue
            candidate = self._normalize_output_core(raw)
            if self._is_generic_process_word(candidate):
                continue
            if self._is_parameter_like(candidate):
                continue
            fallback.append(candidate)

        return self._dedup(fallback)

    def _filter_object(self, raw_object: str, normalized_object: dict | None) -> str:
        raw_object = self._clean(raw_object)

        if normalized_object:
            candidate = self._clean(normalized_object.get("canonical_text", raw_object))
        else:
            candidate = raw_object

        candidate = self._cap(candidate)

        if self._is_parameter_like(candidate):
            return raw_object

        return candidate

    def _attach_input_modifiers(self, inputs: list[str], modifiers: list[str]) -> list[str]:
        if not inputs:
            return []

        modifiers = [re.sub(r"\s*\(.*?\)", "", m).strip() for m in modifiers]
        modifiers = [self._clean(m).lower() for m in modifiers if self._clean(m)]
        modifiers = self._dedup(modifiers)
        if not modifiers:
            return inputs

        result = []
        magnetic_mods = [m for m in modifiers if any(x in m for x in ["переменн", "постоянн", "продольн", "поперечн"])]
        electric_mods = [m for m in modifiers if
                         any(x in m for x in ["высокочастот", "низкочастот", "переменн", "постоянн"])]
        laser_mods = [m for m in modifiers if any(x in m for x in ["импульс", "непрерывн", "монохромат"])]
        used = set()

        for item in inputs:
            line = item
            if self._looks_like_magnetic(item) and magnetic_mods:
                human = ", ".join(self._cap(x) for x in magnetic_mods)
                line = f"{item}. {human}"
                used.update(magnetic_mods)
            elif self._looks_like_electric(item) and electric_mods:
                human = ", ".join(self._cap(x) for x in electric_mods)
                line = f"{item}. {human}"
                used.update(electric_mods)
            elif self._looks_like_laser(item) and laser_mods:
                human = ", ".join(self._cap(x) for x in laser_mods)
                line = f"{item}. {human}"
                used.update(laser_mods)
            result.append(line)

        direction_mods = [
            m for m in modifiers
            if any(x in m for x in ["увеличение", "уменьшение", "немонотонное"])
               and m not in used
        ]
        if direction_mods and result:
            first = direction_mods[0]
            if "увеличение" in first:
                direction_word = "Увеличение"
            elif "уменьшение" in first:
                direction_word = "Уменьшение"
            else:
                direction_word = self._cap(first)
            if direction_word.lower() not in result[0].lower():
                result[0] = f"{result[0]}. {direction_word}"

        return self._dedup(result)

    def _attach_output_modifiers(self, outputs: list[str], modifiers: list[str]) -> list[str]:
        modifiers = [
            self._canonicalize_output_modifier(self._clean(m))
            for m in modifiers if self._clean(m)
        ]
        modifiers = [m for m in modifiers if m]
        modifiers = self._dedup(modifiers)

        if not outputs:
            return []

        result = outputs[:]

        if not modifiers:
            return self._dedup(result)

        attached_mods = set()
        for i, out in enumerate(result):
            for mod in modifiers:
                if mod in attached_mods:
                    continue
                if mod.lower() in out.lower():
                    attached_mods.add(mod)
                    continue
                result[i] = f"{out}. {mod}"
                attached_mods.add(mod)
                break

        remaining = [m for m in modifiers if m not in attached_mods]
        if remaining and result:
            result[0] = result[0] + ". " + ". ".join(remaining)

        return self._dedup(result)

    def _format_inputs_fetext(self, inputs: list[str]) -> str:
        if not inputs:
            return ""

        if len(inputs) == 1:
            return inputs[0].strip()

        parts = []
        for idx, item in enumerate(inputs, start=1):
            parts.append(f"Вход{idx}:{item}")
        return " ".join(parts).strip()

    def _format_outputs_fetext(self, outputs: list[str]) -> str:
        if not outputs:
            return ""
        return ". ".join(outputs).strip()

    def normalize(
            self,
            raw_result: dict,
            main_example: dict | None,
            aux_examples: list[dict],
            thesaurus_candidates: dict
    ) -> dict:
        raw_inputs = raw_result.get("raw_inputs", []) or []
        raw_outputs = raw_result.get("raw_outputs", []) or []
        raw_object = raw_result.get("raw_object", "") or ""
        conditions = raw_result.get("conditions", []) or []
        input_modifiers = raw_result.get("input_modifiers", []) or []
        output_modifiers = raw_result.get("output_modifiers", []) or []

        normalized_inputs = thesaurus_candidates.get("inputs", []) or []
        normalized_outputs = thesaurus_candidates.get("outputs", []) or []
        normalized_object = thesaurus_candidates.get("object")

        final_inputs = self._filter_inputs(raw_inputs, normalized_inputs)
        final_inputs = self._attach_input_modifiers(final_inputs, input_modifiers)
        final_inputs = self._dedup_inputs_semantically(final_inputs)

        final_outputs = self._filter_outputs(raw_outputs, normalized_outputs, conditions)
        final_outputs = self._attach_output_modifiers(final_outputs, output_modifiers)

        final_object = self._filter_object(raw_object, normalized_object)

        return {
            "status": "ok",
            "result": {
                "input_params": self._format_inputs_fetext(final_inputs),
                "object": final_object,
                "output_params": self._format_outputs_fetext(final_outputs),
            }
        }