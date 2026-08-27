def parse_text(data: bytes) -> tuple[str, str]:
    return "", data.decode("utf-8", errors="replace").strip()
