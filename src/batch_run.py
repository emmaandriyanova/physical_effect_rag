
import re
import logging
import warnings
from text_preprocessor import normalize_text
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as tl; tl.set_verbosity_error()
except Exception: pass
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from run_pipeline import RAGPipeline

texts_path = Path(__file__).parent.parent / "data" / "patents_40_texts.txt"
content = texts_path.read_text(encoding="utf-8")
ALL_TEXTS = [p.strip() for p in content.split("\n\n") if p.strip()]

CHECK_IDS = [16, 17, 20, 22]
TEXTS = [(ALL_TEXTS[i-1], i) for i in CHECK_IDS]

def prettify_input(text: str) -> str:
    parts = re.split(r"Вход\d+:\s*", str(text or "").strip())
    return "; ".join(p.strip(" .") for p in parts if p.strip())


pipeline = RAGPipeline()

for raw_text, orig_id in TEXTS:
    text = raw_text.strip()
    text = normalize_text(text)

    print(f"\n{'='*60}")
    print(f"[id={orig_id}] {text.split(chr(10))[0][:60]}")
    print('='*60)

    if not text:
        continue

    try:
        result = pipeline.run(text)
    except Exception as e:
        import traceback

        print(f"ошибка исключение: {e}")
        traceback.print_exc()
        continue

    if result is None:
        print("ошибка: pipeline вернул None")
        continue

    if result.get("status") == "ok":
        r = result["result"]
        print(f"вход:   {prettify_input(r.get('input_params', ''))}")
        print(f"объект: {r.get('object', '')}")
        print(f"выход:  {r.get('output_params', '')}")
    else:
        print(f"ошибка: {result.get('message', '???')}")
        print(f"стадия: {result.get('stage', '???')}")