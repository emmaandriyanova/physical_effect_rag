import re


class CandidateMiner:
    def __init__(self):
        self.split_regex = re.compile(r"(?<=[.!?;])\s+")

        self.input_markers = [
            "под действием",
            "при воздействии",
            "при действии",
            "под влиянием",
            "в условиях",
        ]

        self.output_markers = [
            "происходит",
            "наблюдается",
            "приводит к",
            "вызывает",
            "сопровождается",
            "обусловливает",
            "в результате",
        ]

        self.condition_markers = [
            "в зависимости",
            "при дальнейшем",
            "может",
            "возможно",
            "например",
            "максимален",
            "определяется",
            "зависит",
        ]

    @staticmethod
    def _clean(text: str) -> str:
        text = str(text or "").strip()
        text = text.replace("–", "-").replace("—", "-").replace("−", "-")
        text = re.sub(r"\s+", " ", text)
        return text.strip(" .,:;")

    @staticmethod
    def _dedup(items: list[str]) -> list[str]:
        out = []
        seen = set()
        for x in items:
            x = str(x).strip()
            if not x:
                continue
            key = x.lower()
            if key not in seen:
                out.append(x)
                seen.add(key)
        return out

    def _split_sentences(self, text: str) -> list[str]:
        text = self._clean(text)
        if not text:
            return []
        return [self._clean(x) for x in self.split_regex.split(text) if self._clean(x)]

    def _extract_input_candidates_from_sentence(self, text: str) -> list[str]:
        t = text.lower()
        found = []

        if re.search(r"\bмагнит\w*\s+пол\w*\b", t):
            found.append("магнитное поле")
        if re.search(r"\bэлектр\w*\s+пол\w*\b", t):
            found.append("электрическое поле")
        if re.search(r"\bлазер\w*\s+излуч\w*\b", t):
            found.append("лазерное излучение")
        if re.search(r"\bультразв\w*\b", t):
            found.append("ультразвук")
        if re.search(r"\bвысок\w*\s+давлен\w*\b|\bдавлен\w*\s+\w*\s*высок\w*\b", t):
            found.append("механическое воздействие")
        elif re.search(r"\bдавлен\w*\b", t):
            found.append("давление")
        if re.search(
            r"\bнагруж\w*\b|\bнагруз\w*\b|\bмеханическ\w*\b|\bсжат\w*\b|\bрастяж\w*\b|\bсдвиг\w*\b|\bизгиб\w*\b",
            t
        ):
            found.append("механическое воздействие")

        if re.search(r"\bтемператур\w*\b", t):
            found.append("температура")
        if re.search(r"\bконцентрац\w*\b|\bдол\w*\s+концентрац\w*\b", t):
            found.append("концентрация")
        if re.search(r"\bобъемн\w*\s+дол\w*\b", t):
            found.append("объемная доля")
        if re.search(r"\bкоэффициент\s+теплопередач\w*\b", t):
            found.append("коэффициент теплопередачи")
        if re.search(r"\bтолщин\w*\b", t):
            found.append("толщина")
        if re.search(r"\bмасс\w*\s+концентрац\w*\b|\bсодержани\w*\b", t):
            found.append("концентрация")

        return self._dedup(found)

    def _extract_output_candidates_from_sentence(self, text: str) -> list[str]:
        t = text.lower()
        found = []

        if re.search(r"\bпластическ\w*\s+деформ\w*\b", t):
            found.append("пластическая деформация")
        if re.search(r"\bразупрочнен\w*\b", t):
            found.append("разупрочнение")
        if re.search(r"\bнагрев\w*\b", t):
            found.append("нагрев")
        if re.search(r"\bплавлен\w*\b", t):
            found.append("плавление")
        if re.search(r"\bиспарен\w*\b", t):
            found.append("испарение")
        if re.search(r"\bфазов\w*\s+переход\w*\b", t):
            found.append("фазовый переход")
        if re.search(r"\bплазм\w*\b", t):
            found.append("образование плазмы")
        if re.search(r"\bэлектрическ\w*\s+ток\b|\bток течет\b|\bраспределени\w+\s+тока\b", t):
            found.append("электрический ток")

        if re.search(r"\bсопротивлен\w*\b|\bэлектросопротивлен\w*\b", t):
            found.append("электрическое сопротивление")
        if re.search(r"\bтеплоемкост\w*\b", t):
            found.append("теплоёмкость")
        if re.search(r"\bпотер\w*\s+энерги\w*\b|\bпотер\w*\s+мощност\w*\b", t):
            found.append("потери энергии")
        if re.search(r"\bмодул\w*\s+упруг\w*\b|\bмодул\w*\s+юнг\w*\b", t):
            found.append("модуль упругости")
        if re.search(r"\bударн\w*\s+вязкост\w*\b", t):
            found.append("ударная вязкость")
        if re.search(r"\bпрочност\w*\b", t):
            found.append("прочность")
        if re.search(r"\bкоэффициент\s+диффузи\w*\b", t):
            found.append("коэффициент диффузии")

        return self._dedup(found)

    def _extract_input_modifiers(self, text: str) -> list[str]:
        t = text.lower()
        mods = []
        if "переменн" in t:
            mods.append("переменное")
        if "постоянн" in t:
            mods.append("постоянное")
        if "высокочастот" in t:
            mods.append("высокочастотное")
        if "низкочастот" in t:
            mods.append("низкочастотное")
        if "импульс" in t:
            mods.append("импульсное")
        if re.search(r"\bувелич\w*\b|\bповыш\w*\b|\bрост\w*\b", t):
            mods.append("увеличение")
        if re.search(r"\bуменьш\w*\b|\bсниж\w*\b", t):
            mods.append("уменьшение")
        return self._dedup(mods)

    def _extract_output_modifiers(self, text: str) -> list[str]:
        t = text.lower()
        mods = []
        if re.search(r"\bувелич\w*\b|\bповыш\w*\b|\bвозраст\w*\b", t):
            mods.append("увеличение")
        if re.search(r"\bсниж\w*\b|\bуменьш\w*\b|\bослаб\w*\b", t):
            mods.append("уменьшение")
        if re.search(r"\bнемонотон\w*\b", t):
            mods.append("немонотонное изменение")
        return self._dedup(mods)

    def _extract_object_candidates(self, text: str) -> list[str]:
        t = text.lower()
        found = []

        if re.search(r"\bаморфн\w*\s+сплав\w*\b", t):
            found.append("аморфный сплав")
        if re.search(r"\bметаллическ\w*\s+стекл\w*\b", t):
            found.append("металлическое стекло")
        if re.search(r"\bаморфн\w*\s+матери\w*\b", t):
            found.append("аморфный материал")
        if re.search(r"\bпорошок\w*\s+аморфн\w*\b|\bаморфн\w*\s+порошок\w*\b", t):
            found.append("аморфный порошок")
        if re.search(r"\bжидк\w*\s+расплав\w*\b|\bрасплав\w*\b", t):
            found.append("расплав металла")
        if re.search(r"\bпленк\w*\s+аморфн\w*\b|\bаморфн\w*\s+пленк\w*\b", t):
            found.append("пленка аморфного сплава")
        if re.search(r"\bсплав\w*\s+на\s+основ\w*\b", t):
            found.append("сплав на основе металла")

        if not found:
            if re.search(r"\bпроводящ\w*\s+сред\w*\b", t):
                found.append("проводящая среда")
            if re.search(r"\bпроводник\w*\b", t):
                found.append("проводник")
            if re.search(r"\bметалл\w*\b", t):
                found.append("металлы")
            if re.search(r"\bматериал\w*\b", t):
                found.append("материал")

        return self._dedup(found)

    def mine(self, text: str) -> dict:
        sentences = self._split_sentences(text)

        candidate_inputs = []
        candidate_outputs = []
        candidate_objects = []
        candidate_conditions = []
        candidate_input_modifiers = []
        candidate_output_modifiers = []

        for sent in sentences:
            low = sent.lower()

            inputs_in_sentence = self._extract_input_candidates_from_sentence(sent)
            if inputs_in_sentence:
                candidate_inputs.extend(inputs_in_sentence)
                candidate_input_modifiers.extend(self._extract_input_modifiers(sent))

            outputs_in_sentence = self._extract_output_candidates_from_sentence(sent)
            if outputs_in_sentence:
                candidate_outputs.extend(outputs_in_sentence)
                candidate_output_modifiers.extend(self._extract_output_modifiers(sent))

            candidate_objects.extend(self._extract_object_candidates(sent))

            if any(marker in low for marker in self.condition_markers):
                candidate_conditions.append(sent)

        candidate_inputs = self._dedup(candidate_inputs)
        candidate_outputs = self._dedup(candidate_outputs)
        candidate_objects = self._dedup(candidate_objects)
        candidate_conditions = self._dedup(candidate_conditions)
        candidate_input_modifiers = self._dedup(candidate_input_modifiers)
        candidate_output_modifiers = self._dedup(candidate_output_modifiers)

        best_object = candidate_objects[0] if candidate_objects else ""

        return {
            "candidate_inputs": candidate_inputs,
            "candidate_object": best_object,
            "candidate_outputs": candidate_outputs,
            "candidate_conditions": candidate_conditions,
            "candidate_input_modifiers": candidate_input_modifiers,
            "candidate_output_modifiers": candidate_output_modifiers,
        }