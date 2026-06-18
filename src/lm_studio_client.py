"""
HTTP-клиент для взаимодействия с дообученной языковой моделью через LM Studio
API. Формирует запросы в формате completion с промтом-префиксом, обрабатывает
теги <think>, повторяет запрос до трёх раз при сетевых ошибках.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import requests
import logging
import re
from typing import Optional

from config import LM_STUDIO_URL, LM_STUDIO_MODEL_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LMStudioClient:
    def __init__(
        self,
        base_url: str = LM_STUDIO_URL,
        model_id: str = LM_STUDIO_MODEL_ID,
        timeout: int = 300
    ):
        self.base_url = base_url
        self.model_id = model_id
        self.timeout = timeout

    @staticmethod
    def clean_response_text(text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = text.strip()

        json_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if json_fence:
            return json_fence.group(1).strip()

        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        start = text.find("{")
        if start > 0:
            text = text[start:]

        return text.strip()

    def chat(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.0,
            max_tokens: int = 2000
    ) -> str:
        """
        Отправляет запрос к языковой модели в LM Studio и возвращает очищенный ответ.

        :param system_prompt: инструкции для модели, добавляются как префикс перед текстом
        :param user_prompt: основной текст запроса (патент + пример)
        :param temperature: температура генерации (0.0 — детерминированный режим)
        :param max_tokens: максимальное число токенов в ответе
        :return: строка с ответом модели, очищенная от тегов <think> и markdown
        """
        prefix = f"{system_prompt}\n\n" if system_prompt and system_prompt.strip() else ""
        payload = {
            "model": self.model_id,
            "prompt": f"{prefix}### Текст:\n{user_prompt}\n\n### JSON:\n<think>\n\n</think>\n",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["###", "\n\n\n"],
        }

        logging.info("отправка запроса в лм студио")

        response = requests.post(
            self.base_url,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["text"]
        logging.info(f"RAW ответ модели ПОЛНЫЙ: '{content}'")
        return self.clean_response_text(content)

    def chat_optional(self, system_prompt, user_prompt, temperature=0.0, max_tokens=2000):
        """
        Вызывает chat() с повторными попытками при сетевых ошибках.

        :param system_prompt: инструкции для модели
        :param user_prompt: основной текст запроса
        :param temperature: температура генерации
        :param max_tokens: максимальное число токенов в ответе
        :return: строка с ответом модели или None при трёх неудачных попытках
        """
        for attempt in range(3):
            try:
                return self.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except requests.RequestException as e:
                err_body = ""
                if hasattr(e, "response") and e.response is not None:
                    try:
                        err_body = e.response.text[:500]
                    except Exception:
                        pass
                logging.error(
                    f"попытка {attempt + 1}/3 — ошибка запроса: {e}\n"
                    f"тело ответа: {err_body}"
                )
                if attempt == 2:
                    return None
            except KeyError as e:
                logging.error(f"неожиданная структура ответа: {e}")
                return None
            except Exception as e:
                logging.error(f"непредвиденная ошибка: {e}")
                return None
        return None


if __name__ == "__main__":
    client = LMStudioClient()

    system_prompt = (
        "ты извлекаешь структуру физического эффекта. "
        "отвечай кратко и строго по запросу."
    )

    user_prompt = (
        "текст: затухание электромагнитных волн по мере их проникновения "
        "в глубь проводящей среды. "
        "назови только физический эффект."
    )

    result = client.chat_optional(system_prompt, user_prompt)

    print("\nрезультат:")
    print(result if result else "нет ответа")
