import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging
import re
import hashlib
import html

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_text(text: str) -> str:

    if not text:
        return ""

    text = html.unescape(str(text))

    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = text.replace("–", "-").replace("—", "-").replace("−", "-")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def canonicalize_text(text: str) -> str:

    text = clean_text(text).lower()

    text = re.sub(r"\s*([,.;:!?()])\s*", r"\1", text)

    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def text_hash(text: str) -> str:

    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_and_prepare_xml(xml_path: str) -> ET.Element:

    with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
        raw_text = f.read()

    raw_text = raw_text.replace("\ufeff", "")
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    raw_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", raw_text)

    raw_text = raw_text.strip()

    raw_text = re.sub(r"^\s*<\?xml[^>]*\?>", "", raw_text).strip()

    raw_text = raw_text.replace("<FEText\n", "<FEText>\n")

    raw_text = f"<root>\n{raw_text}\n</root>"

    return ET.fromstring(raw_text)

def parse_fetext_xml(xml_path: str) -> pd.DataFrame:
    logging.info(f"начинаем парсинг файла: {xml_path}")

    root = load_and_prepare_xml(xml_path)

    effects = []

    for effect in root.findall('.//FEText'):
        effect_id = clean_text(effect.findtext('IDFE', ''))

        if not effect_id:
            continue

        description = clean_text(effect.findtext('text', ''))
        canonical_description = canonicalize_text(description)

        effect_data = {
            'id': effect_id,
            'name': clean_text(effect.findtext('name', '')),
            'description': description,
            'canonical_description': canonical_description,
            'description_hash': text_hash(canonical_description) if canonical_description else "",
            'input_params': clean_text(effect.findtext('textInp', '')),
            'output_params': clean_text(effect.findtext('textOut', '')),
            'object': clean_text(effect.findtext('textObj', '')),
            'application': clean_text(effect.findtext('textApp', '')),
            'literature': clean_text(effect.findtext('textLit', ''))
        }

        effects.append(effect_data)

    df = pd.DataFrame(effects)

    logging.info(f"найдено {len(df)} эффектов")

    df = df[df['description'] != ""]

    output_path = Path(__file__).parent.parent / 'data' / 'parsed_effects.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    logging.info(f"сохранено в {output_path}")

    return df


if __name__ == "__main__":
    xml_file = Path(__file__).parent.parent / 'data' / 'FEText.xml'

    df = parse_fetext_xml(xml_file)

    print("\nпервые 3 эффекта:")
    print(df[['id', 'name', 'description_hash']].head(3))