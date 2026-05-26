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
            max_tokens: int = 1000
    ) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\n/no_think"}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_thinking": False
        }

        logging.info("отправка запроса в лм студио")

        response = requests.post(
            self.base_url,
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        return self.clean_response_text(content)

    def chat_optional(self, system_prompt, user_prompt, temperature=0.0, max_tokens=800):
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
