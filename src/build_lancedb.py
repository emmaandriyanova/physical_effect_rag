"""
Сборка векторной базы данных LanceDB из предварительно вычисленных
эмбеддингов физических эффектов: загрузка чанков и векторов, создание
и заполнение таблицы с индексом для семантического поиска.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import lancedb
import numpy as np
import pickle
from pathlib import Path
import logging
import pyarrow as pa
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LanceDBBuilder:
    def __init__(self, db_path: str = "./lancedb_data"):
        self.db_path = Path(__file__).parent.parent / db_path

        if self.db_path.exists():
            shutil.rmtree(self.db_path)
            logging.info(f"удалена старая база: {self.db_path}")

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        logging.info(f"создана новая база: {self.db_path}")

    def create_table(self, table_name: str = "physical_effects", embedding_dim: int = 384):
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), embedding_dim)),

            pa.field("chunk_id", pa.string()),
            pa.field("effect_id", pa.string()),
            pa.field("effect_name", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("text", pa.string()),


            pa.field("description", pa.string()),
            pa.field("canonical_description", pa.string()),
            pa.field("description_hash", pa.string()),
            pa.field("input_params", pa.string()),
            pa.field("object", pa.string()),
            pa.field("output_params", pa.string()),
        ])

        self.table = self.db.create_table(table_name, schema=schema, mode="overwrite")
        logging.info(f"создана таблица: {table_name}")
        return self.table

    def load_data(self, chunks_path: Path, embeddings_path: Path):
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        embeddings = np.load(embeddings_path)

        if len(chunks) != len(embeddings):
            raise ValueError("кол-во чанков и эмбеддингов не совпадает")

        data = []
        for chunk, emb in zip(chunks, embeddings):
            data.append({
                "vector": emb.tolist(),

                "chunk_id": chunk.get("chunk_id", ""),
                "effect_id": chunk.get("effect_id", ""),
                "effect_name": chunk.get("effect_name", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "text": chunk.get("text", ""),


                "description": chunk.get("description", ""),
                "canonical_description": chunk.get("canonical_description", ""),
                "description_hash": chunk.get("description_hash", ""),
                "input_params": chunk.get("input_params", ""),
                "object": chunk.get("object", ""),
                "output_params": chunk.get("output_params", ""),
            })

        batch_size = 100
        for i in range(0, len(data), batch_size):
            self.table.add(data[i:i + batch_size])
            logging.info(f"загружено {min(i + batch_size, len(data))} из {len(data)}")

        try:
            self.table.create_index(
                metric="cosine",
                index_type="IVF_PQ",
                num_partitions=32,
                num_sub_vectors=32
            )
            logging.info("индекс IVF_PQ создан")
        except Exception as e:
            logging.warning(f"не удалось создать индекс с параметрами: {e}")
            self.table.create_index(metric="cosine")
            logging.info("создан обычный индекс")

        return self.table


def main():
    cache_dir = Path(__file__).parent.parent / "embeddings_cache"
    chunks_path = cache_dir / "chunks_with_emb.pkl"
    embeddings_path = cache_dir / "embeddings.npy"

    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("не найдены файлы embeddings_cache")

    embeddings = np.load(embeddings_path)
    embedding_dim = embeddings.shape[1]

    builder = LanceDBBuilder()
    builder.create_table(embedding_dim=embedding_dim)
    builder.load_data(chunks_path, embeddings_path)

    print(f"\nбаза готова: {builder.db_path}")
    print(f"всего записей: {builder.table.count_rows()}")


if __name__ == "__main__":
    main()