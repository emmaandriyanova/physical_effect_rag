import logging

from config import LM_STUDIO_URL, LM_STUDIO_MODEL_ID
from exact_match import ExactMatchFinder
from retriever import FERetriever
from thesaurus_match import ThesaurusMatcher
from raw_extractor import RawExtractor
from normalizer import Normalizer
from verifier import Verifier
from thesaurus_normalizer import ThesaurusNormalizer
from candidate_miner import CandidateMiner
from fetext_formatter import FETextFormatter
from text_preprocessor import normalize_text

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


class RAGPipeline:
    def __init__(
        self,
        lm_studio_url: str = LM_STUDIO_URL,
        model_id: str = LM_STUDIO_MODEL_ID
    ):

        self.candidate_miner = CandidateMiner()
        self.exact_matcher = ExactMatchFinder()
        self.retriever = FERetriever()
        self.thesaurus_matcher = ThesaurusMatcher()
        self.thesaurus_normalizer = ThesaurusNormalizer(self.thesaurus_matcher)
        self.fetext_formatter = FETextFormatter(self.thesaurus_matcher)

        self.raw_extractor = RawExtractor(
            lm_studio_url=lm_studio_url,
            model_id=model_id
        )
        self.normalizer = Normalizer(
            lm_studio_url=lm_studio_url,
            model_id=model_id
        )
        self.verifier = Verifier()

    def run(self, query_text: str) -> dict:
        query_text = normalize_text(query_text.strip())

        if not query_text:
            return {
                "status": "error",
                "message": "Пустой входной текст"
            }

        exact_result = self.exact_matcher.find(query_text)
        if exact_result:
            return {
                "status": "ok",
                "source": "exact_match",
                "result": {
                    "input_params": exact_result["input_params"],
                    "object": exact_result["object"],
                    "output_params": exact_result["output_params"]
                },
                "verification": None,
                "debug": {
                    "effect_id": exact_result["effect_id"],
                    "effect_name": exact_result["effect_name"],
                    "match_type": exact_result["match_type"]
                }
            }

        near = self.exact_matcher.find_near_duplicate(query_text)



        main_example = None
        aux_examples = []

        if near:
            main_example = near["example"]

        if main_example is None:
            example_bundle = self.retriever.get_example_bundle(query_text, main_k=1, aux_k=2)
            main_example = example_bundle["main_example"]
            aux_examples = example_bundle["aux_examples"]
        else:
            example_bundle = self.retriever.get_example_bundle(query_text, main_k=0, aux_k=2)
            aux_examples = example_bundle["aux_examples"]

        retrieved_examples = []
        if main_example:
            retrieved_examples.append(main_example)
        retrieved_examples.extend(aux_examples)

        candidate_bundle = self.candidate_miner.mine(query_text)
        raw_extraction = self.raw_extractor.extract(
            query_text=query_text,
            retrieved_examples=retrieved_examples,
            candidate_bundle=candidate_bundle
        )

        if raw_extraction.get("status") != "ok":
            return {
                "status": "error",
                "stage": "raw_extraction",
                "message": raw_extraction.get("message", "ошибка"),
                "debug": {
                    "raw_response": raw_extraction.get("raw_response", ""),
                    "main_example": main_example,
                    "aux_examples": aux_examples
                }
            }

        raw_result = raw_extraction["result"]

        thesaurus_candidates = self.thesaurus_normalizer.normalize_raw_result(raw_result)

        normalization = self.normalizer.normalize(
            raw_result=raw_result,
            main_example=main_example,
            aux_examples=aux_examples,
            thesaurus_candidates=thesaurus_candidates
        )

        if normalization.get("status") != "ok":
            return {
                "status": "error",
                "stage": "normalization",
                "message": normalization.get("message", "ошибка нормализации"),
                "debug": {
                    "raw_result": raw_result,
                    "main_example": main_example,
                    "aux_examples": aux_examples,
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

        return {
            "status": "ok",
            "source": "llm_rag",
            "result": verified["result"],
            "verification": {
                "is_valid": verified["is_valid"],
                "issues": verified["issues"],
                "warnings": verified["warnings"],
                "stats": verified["stats"]
            },
            "debug": {
                "raw_result": raw_result,
                "main_example": main_example,
                "aux_examples": aux_examples,
                "thesaurus_candidates": thesaurus_candidates,
                "normalized_result": normalized_result,
                "formatted_result": formatted_result
            }
        }
