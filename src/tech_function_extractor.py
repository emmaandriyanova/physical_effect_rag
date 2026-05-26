
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config import KEYT5_MODEL_PATH


class TechFunctionExtractor:
    def __init__(
        self,
        model_path: str = KEYT5_MODEL_PATH,
        max_input_length: int = 512,
        max_output_length: int = 128,
        num_beams: int = 5,
        repetition_penalty: float = 1.2,
    ):
        self.model_path = model_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.eval()

    def preprocess_for_keyt5(self, text: str) -> str:
        priority_markers = [
            "технический результат",
            "цель изобретения",
            "задача изобретения",
            "технической задачей",
            "техническим результатом",
            "изобретение позволяет",
            "достигается",
            "обеспечивает",
            "позволяет обеспечить",
            "направлено на",
        ]

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        priority = []
        other = []

        for sent in sentences:
            sent_low = sent.lower()
            if any(m in sent_low for m in priority_markers):
                priority.append(sent)
            else:
                other.append(sent)

        ordered = priority + other

        result = ""
        for sent in ordered:
            if len(result) + len(sent) > self.max_input_length * 4:
                break
            result += sent + " "

        return result.strip() if result.strip() else text[:self.max_input_length * 4]

    def extract(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        preprocessed = self.preprocess_for_keyt5(text)

        inputs = self.tokenizer(
            preprocessed,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                repetition_penalty=self.repetition_penalty,
            )

        raw = self.tokenizer.decode(output[0], skip_special_tokens=True)

        functions = [f.strip().rstrip(";").strip() for f in raw.split(";")]
        functions = [f for f in functions if f]

        return functions