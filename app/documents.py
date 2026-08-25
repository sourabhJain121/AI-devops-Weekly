"""Document loading and text extraction, page by page where the format has pages."""
import re
from pathlib import Path

from app import config


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((number, text))
    return pages


def _read_text(path: Path) -> list[tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def clean(text: str) -> str:
    """Normalise whitespace without destroying paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, cleaned_text), ...] for a supported file."""
    suffix = path.suffix.lower()
    if suffix not in config.SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    pages = _read_pdf(path) if suffix == ".pdf" else _read_text(path)
    return [(number, clean(text)) for number, text in pages if clean(text)]


def discover(directory: Path) -> list[Path]:
    """All supported documents under a directory, deterministically ordered."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_SUFFIXES
        and not p.name.startswith("README")
    )
