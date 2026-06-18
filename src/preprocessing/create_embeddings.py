"""
Генерация векторных эмбеддингов для чанков физических эффектов с помощью
модели SentenceTransformer (multilingual-e5-small) и сохранение результатов
в формате pickle для последующей загрузки в LanceDB.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EmbeddingGenerator:

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        logging.info(f"загружаем модель эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logging.info("модель загружена")

    def generate_passage_embeddings(self, texts, batch_size=64):
        prepared = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(
            prepared,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings

    def save_embeddings(self, embeddings, chunks, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        emb_path = output_dir / "embeddings.npy"
        np.save(emb_path, embeddings)

        chunks_path = output_dir / "chunks_with_emb.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        metadata_path = output_dir / "metadata.json"
        metadata = {
            "embedding_model": self.model_name,
            "num_chunks": len(chunks),
            "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logging.info(f"эмбеддинги сохранены в {emb_path}")
        logging.info(f"чанки сохранены в {chunks_path}")
        logging.info(f"метаданные сохранены в {metadata_path}")

        return emb_path, chunks_path, metadata_path


def main():
    chunks_path = Path(__file__).parent.parent.parent / "data" / "chunks.json"

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    logging.info(f"загружено {len(chunks)} чанков")

    generator = EmbeddingGenerator()
    embeddings = generator.generate_passage_embeddings(texts)

    logging.info(f"форма эмбеддингов: {embeddings.shape}")

    cache_dir = Path(__file__).parent.parent.parent / "embeddings_cache"
    generator.save_embeddings(embeddings, chunks, cache_dir)

    print(f"\nсоздано {len(embeddings)} эмбеддингов размерностью {embeddings.shape[1]}")


if __name__ == "__main__":
    main()