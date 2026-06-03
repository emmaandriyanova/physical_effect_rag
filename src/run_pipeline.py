import logging

from config import LM_STUDIO_URL, LM_STUDIO_MODEL_ID
from retriever import FERetriever
from thesaurus_match import ThesaurusMatcher
from raw_extractor import RawExtractor
from normalizer import Normalizer
from verifier import Verifier
from thesaurus_normalizer import ThesaurusNormalizer
from fetext_formatter import FETextFormatter
from text_preprocessor import normalize_text

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


class RAGPipeline:
    def __init__(
        self,
        lm_studio_url: str = LM_STUDIO_URL,
        model_id: str = LM_STUDIO_MODEL_ID
    ):
        self.retriever = FERetriever()
        self.thesaurus_matcher = ThesaurusMatcher()
        self.thesaurus_normalizer = ThesaurusNormalizer(self.thesaurus_matcher)
        self.fetext_formatter = FETextFormatter(self.thesaurus_matcher)
        self.raw_extractor = RawExtractor(
            lm_studio_url=lm_studio_url,
            model_id=model_id
        )
        self.normalizer = Normalizer()
        self.verifier = Verifier()

    def run(self, query_text: str) -> dict:
        query_text = normalize_text(query_text.strip())

        if not query_text:
            return {
                "status": "error",
                "message": "Пустой входной текст"
            }

        example_bundle = self.retriever.get_example_bundle(query_text, main_k=1, aux_k=0)
        retrieved_examples = []
        if example_bundle["main_example"]:
            retrieved_examples.append(example_bundle["main_example"])
        retrieved_examples.extend(example_bundle["aux_examples"])

        raw_extraction = self.raw_extractor.extract(
            query_text=query_text,
            retrieved_examples=retrieved_examples
        )

        if raw_extraction.get("status") != "ok":
            return {
                "status": "error",
                "stage": "raw_extraction",
                "message": raw_extraction.get("message", "ошибка"),
                "debug": {
                    "raw_response": raw_extraction.get("raw_response", ""),
                    "retrieved_examples": retrieved_examples
                }
            }

        raw_result = raw_extraction["result"]

        thesaurus_candidates = self.thesaurus_normalizer.normalize_raw_result(raw_result)

        normalization = self.normalizer.normalize(
            raw_result=raw_result,
            thesaurus_candidates=thesaurus_candidates
        )

        if normalization.get("status") != "ok":
            return {
                "status": "error",
                "stage": "normalization",
                "message": normalization.get("message", "ошибка нормализации"),
                "debug": {
                    "raw_result": raw_result,
                    "thesaurus_candidates": thesaurus_candidates
                }
            }

        normalized_result = normalization["result"]

        formatted_result = self.fetext_formatter.format_fields(
            input_params=normalized_result.get("input_params", ""),
            object_text=normalized_result.get("object", ""),
            output_params=normalized_result.get("output_params", "")
        )

        verified = self.verifier.verify(formatted_result)

        pipeline_status = "ok" if verified.get("is_valid") else "error"
        message = None
        if pipeline_status == "error":
            issues = verified.get("issues", [])
            message = "; ".join(issues) if issues else "ошибка верификации"

        return {
            "status": pipeline_status,
            "source": "llm",
            "stage": "verification" if pipeline_status == "error" else None,
            "message": message,
            "result": verified["result"],
            "verification": {
                "is_valid": verified["is_valid"],
                "issues": verified["issues"],
                "warnings": verified["warnings"],
                "stats": verified["stats"]
            },
            "debug": {
                "raw_result": raw_result,
                "retrieved_examples": retrieved_examples,
                "thesaurus_candidates": thesaurus_candidates,
                "normalized_result": normalized_result,
                "formatted_result": formatted_result
            }
        }
