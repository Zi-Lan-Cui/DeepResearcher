import re
from io import BytesIO

from pypdf import PdfReader


def parse_pdf(data: bytes) -> tuple[str, str]:
    reader = PdfReader(BytesIO(data))
    text = re.sub(
        r"\s+", " ", "\n".join(page.extract_text() or "" for page in reader.pages)
    ).strip()
    return "", text
