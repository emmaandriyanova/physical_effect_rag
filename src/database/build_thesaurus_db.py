"""
Сборка векторной базы данных LanceDB для тезауруса физических эффектов:
парсинг XML-файла тезауруса, генерация эмбеддингов терминов и запись
в таблицу LanceDB для последующего семантического поиска.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import xml.etree.ElementTree as ET
import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import pyarrow as pa
import re
import html
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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

    raw_text = f"<root>\n{raw_text}\n</root>"

    return ET.fromstring(raw_text)


def build_thesaurus_table():
    db_path = "./lancedb_data"
    xml_file = Path(__file__).parent.parent.parent / "data" / "Thes.xml"
    table_name = "thesaurus"
    embedding_model = "intfloat/multilingual-e5-small"

    logging.info("создание таблицы тезауруса")

    db = lancedb.connect(str(Path(__file__).parent.parent.parent / db_path))
    embedder = SentenceTransformer(embedding_model)

    root = load_and_prepare_xml(xml_file)

    nodes = {}

    for node in root.findall(".//Thes"):
        node_id = clean_text(node.findtext("id", ""))
        name = clean_text(node.findtext("name", ""))
        parent = clean_text(node.findtext("parent", ""))

        if node_id:
            nodes[node_id] = {
                "id": node_id,
                "name": name,
                "canonical_name": canonicalize_text(name),
                "name_hash": text_hash(canonicalize_text(name)) if name else "",
                "parent": parent,
                "path_list": [],
                "full_path": "",
                "path_depth": 0,
                "top_level": ""
            }

    def get_path(node_id: str):
        path = []
        current = node_id
        visited = set()

        while current and current in nodes and current not in visited:
            visited.add(current)
            node = nodes[current]

            if node["name"]:
                path.insert(0, node["name"])

            current = node["parent"]
            if current == "ROOT" or not current:
                break

        return path

    for node_id in nodes:
        path = get_path(node_id)
        nodes[node_id]["path_list"] = path
        nodes[node_id]["full_path"] = " > ".join(path)
        nodes[node_id]["path_depth"] = len(path)
        nodes[node_id]["top_level"] = path[0] if path else ""

    data = []
    for node_id, node in nodes.items():
        search_text = (
            f"Термин: {node['name']}. "
            f"Путь: {node['full_path']}. "
            f"ID: {node['id']}. "
            f"Родитель: {node['parent']}."
        )

        vector = embedder.encode(
            f"passage: {search_text}",
            normalize_embeddings=True
        ).tolist()

        data.append({
            "vector": vector,
            "node_id": node["id"],
            "name": node["name"],
            "canonical_name": node["canonical_name"],
            "name_hash": node["name_hash"],
            "full_path": node["full_path"],
            "path": " > ".join(node["path_list"]),
            "path_depth": node["path_depth"],
            "top_level": node["top_level"],
            "parent": node["parent"],
            "search_text": search_text
        })

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), embedder.get_sentence_embedding_dimension())),
        pa.field("node_id", pa.string()),
        pa.field("name", pa.string()),
        pa.field("canonical_name", pa.string()),
        pa.field("name_hash", pa.string()),
        pa.field("full_path", pa.string()),
        pa.field("path", pa.string()),
        pa.field("path_depth", pa.int32()),
        pa.field("top_level", pa.string()),
        pa.field("parent", pa.string()),
        pa.field("search_text", pa.string()),
    ])

    table = db.create_table(table_name, schema=schema, mode="overwrite")

    batch_size = 100
    for i in range(0, len(data), batch_size):
        table.add(data[i:i + batch_size])
        logging.info(f"загружено {min(i + batch_size, len(data))} из {len(data)}")

    table.create_index(metric="cosine")
    logging.info(f"таблица тезауруса готова. всего записей: {table.count_rows()}")

    return table


if __name__ == "__main__":
    build_thesaurus_table()