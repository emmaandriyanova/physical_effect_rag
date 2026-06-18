"""
Предобработка входного текста патента: исправление смешанной латиницы
и кириллицы (визуально схожих символов), нормализация пробелов, переносов
строк и спецсимволов перед подачей в пайплайн.

© 2025–2026 Андриянова Анастасия Владиславовна
Создан: 2025
Последнее изменение: 02.06.2026
Контакт: flomaster0909@mail.ru | github.com/emmaandriyanova
"""
import re
import unicodedata


LATIN_TO_CYRILLIC = {
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н',
    'K': 'К', 'M': 'М', 'O': 'О', 'P': 'Р', 'T': 'Т',
    'X': 'Х', 'Y': 'У',
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р',
    'x': 'х', 'y': 'у',
}


def fix_mixed_script(text: str) -> str:
    def fix_word(word: str) -> str:
        has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in word)
        has_latin_lookalike = any(c in LATIN_TO_CYRILLIC for c in word)

        if has_cyrillic and has_latin_lookalike:
            return ''.join(LATIN_TO_CYRILLIC.get(c, c) for c in word)
        return word

    tokens = re.split(r'(\s+|[^\w])', text)
    return ''.join(fix_word(t) for t in tokens)


def normalize_text(text: str) -> str:
    if not text:
        return text

    text = re.sub(r'^[─\-\s]*\[\d+\][^\n]*\n', '', text).strip()

    text = fix_mixed_script(text)

    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


if __name__ == "__main__":
    test_cases = [
        "Koнцeнтpaция блaгopoднoгo мeтaллa",
        "Teмпepaтypa (C). Увeличeниe oт -50 дo +50 C.",
        "Cилoвoe (мexaничecкoe) вoздeйcтвиe",
        "Удeльнoe элeктpocoпpoтивлeниe (Oм*м)",
    ]
    for t in test_cases:
        print(f"до:   {t}")
        print(f"после: {normalize_text(t)}")
        print()