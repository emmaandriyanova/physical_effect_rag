# Программный модуль для выявления физических эффектов и технических функций из текстов патентов

**Автор:** Андриянова Анастасия Владиславовна, ИВТ-463  
**Контакт:** flomaster0909@mail.ru | github.com/emmaandriyanova

---

## Описание

RAG-система (Retrieval-Augmented Generation) для автоматического извлечения структурированных описаний физических эффектов и технических функций из текстов патентов на русском языке.

Физический эффект представляется в формате базы данных FEText как тройка:
- **Вход** — входное физическое воздействие (вид, характеристика, величина, единицы)
- **Объект** — физический объект или среда, на которую оказывается воздействие
- **Выход** — выходная физическая величина или явление (название, единицы, направление изменения)

Технические функции извлекаются отдельно с помощью дообученной модели KeyT5.

---

## Архитектура пайплайна

```
Текст патента
      │
      ▼
[text_preprocessor]   — нормализация текста (смешанные символы, пробелы)
      │
      ▼
[FERetriever]         — векторный поиск похожего эффекта в LanceDB (RAG-контекст)
      │
      ▼
[RawExtractor]        — LLM-извлечение (дообученная Qwen3-8b через LM Studio)
      │
      ▼
[ThesaurusNormalizer] — алиас-правила + семантический поиск в тезаурусе
      │
      ▼
[Normalizer]          — фильтрация, дедупликация, применение тезаурусных кандидатов
      │
      ▼
[FETextFormatter]     — приведение к формату FEText, нормализация регистра
      │
      ▼
[Verifier]            — проверка корректности результата
      │
      ▼
Структурированная запись физического эффекта
```

---

## Файловая структура

```
physical_effects_rag/
├── src/                          # Исходный код
│   ├── main.py                   # Графический интерфейс (Tkinter)
│   ├── config.py                 # Централизованная конфигурация
│   ├── run_pipeline.py           # Основной класс RAGPipeline
│   │
│   ├── raw_extractor.py          # LLM-извлечение структуры эффекта
│   ├── normalizer.py             # Нормализация сырых результатов
│   ├── fetext_formatter.py       # Форматирование по стандарту FEText
│   ├── thesaurus_normalizer.py   # Нормализация через тезаурус
│   ├── thesaurus_match.py        # Поиск в тезаурусе (LanceDB + эмбеддинги)
│   ├── retriever.py              # Векторный поиск (RAG-ретривал)
│   ├── verifier.py               # Верификация результата
│   ├── lm_studio_client.py       # HTTP-клиент для LM Studio API
│   ├── text_preprocessor.py      # Предобработка текста
│   ├── tech_function_extractor.py# Извлечение техфункций (KeyT5)
│   │
│   ├── parse_xml.py              # Парсинг FEText.xml → CSV/JSON
│   ├── create_chunks.py          # Разбиение записей на чанки
│   ├── create_embeddings.py      # Генерация эмбеддингов
│   ├── build_lancedb.py          # Сборка базы данных LanceDB
│   ├── build_thesaurus_db.py     # Сборка тезауруса в LanceDB
│   │
│   ├── evaluate_rag.py           # Прогон пайплайна на тестовой выборке
│   ├── compare_ranked.py         # Сравнение RAG vs. базовая модель
│   ├── batch_run.py              # Пакетный запуск для отладки
│   └── debug_pipeline.py         # Отладочный скрипт
│
├── data/
│   ├── FEText.xml                # База физических эффектов (исходник)
│   ├── Thes.xml                  # Тезаурус физических эффектов
│   ├── parsed_effects.csv        # Распарсенные эффекты
│   ├── chunks.json               # Чанки для индексации
│   └── patents_40_texts.txt      # Тестовая выборка (40 патентов)
│
├── lancedb_data/                 # Векторная база данных LanceDB
├── embeddings_cache/             # Кэш эмбеддингов
├── requirements.txt              # Зависимости Python
└── README.md                     # Данный файл
```

---

## Требования

- **Python:** 3.10 или выше
- **LM Studio:** запущенный локально или на удалённом сервере с загруженной моделью `qwen3-fetext-q4km.gguf`
- **ОЗУ:** не менее 8 ГБ (для загрузки эмбеддинговой модели и тезауруса)

---

## Установка зависимостей

```bash
pip install -r requirements.txt
pip install transformers torch
```

Полный список зависимостей:

| Пакет | Назначение |
|-------|-----------|
| `pandas`, `numpy` | Работа с табличными данными |
| `lancedb`, `pyarrow` | Векторная база данных |
| `sentence-transformers` | Эмбеддинги (`intfloat/multilingual-e5-small`) |
| `transformers`, `torch` | Модель KeyT5 для техфункций |
| `requests` | HTTP-клиент для LM Studio |
| `lxml` | Парсинг XML |
| `tqdm` | Прогресс-бары |

---

## Модели

Модели доступны на Google Drive: [https://drive.google.com/drive/folders/1mrdEDhgPoQfrmmuSROgiV4lgPuqiy19n?usp=share_link]

| Модель | Описание |
|--------|---------|
| `qwen3-fetext-q4km.gguf` | Qwen3-8B, дообученная на базе FEText (Q4_K_M квантизация) |
| `keyt5-finetuned/` | KeyT5-large, дообученная на извлечение технических функций |

После скачивания укажите пути в `src/config.py`:

```python
LM_STUDIO_URL    = "http://your-lm-studio-host:1234/v1/completions"
KEYT5_MODEL_PATH = "/path/to/keyt5-finetuned"
```

---

## Запуск

### Графический интерфейс

```bash
cd src
python main.py
```

Откроется окно приложения. Вставьте текст патента в поле ввода и нажмите **Отправить**. Результаты отображаются на вкладках «Физический эффект» и «Технические функции».

### Сборка базы данных (однократно)

Если база данных ещё не собрана:

```bash
cd src
python parse_xml.py          # парсинг FEText.xml
python create_chunks.py      # создание чанков
python create_embeddings.py  # генерация эмбеддингов
python build_lancedb.py      # сборка векторной БД
python build_thesaurus_db.py # сборка тезауруса
```

### Оценка качества

```bash
cd src
python evaluate_rag.py    # прогон на 40 патентах → data/results_rag.csv
python compare_ranked.py  # сравнение RAG vs. без RAG → data/comparison_*.csv
```

---

## Формат результата

```json
{
  "status": "ok",
  "result": {
    "input_params": "Электрическое поле. Высокочастотное. Напряжённость (В/м).",
    "object": "Проводящая среда.",
    "output_params": "Электрический ток. Уменьшение."
  },
  "verification": {
    "is_valid": true,
    "issues": [],
    "warnings": []
  }
}
```
