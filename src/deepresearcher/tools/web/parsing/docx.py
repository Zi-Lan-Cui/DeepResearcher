import re
from io import BytesIO

from docx import Document


def parse_docx(data: bytes) -> tuple[str, str]:
    document = Document(BytesIO(data))
    text = re.sub(
        r"\s+", " ", "\n".join(paragraph.text for paragraph in document.paragraphs)
    ).strip()
    return "", text
