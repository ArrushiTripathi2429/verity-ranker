import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]

