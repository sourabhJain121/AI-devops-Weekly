"""Chunking: section-aware, paragraph-aware, size-bounded, with overlap.

Strategy, and why:
  * A chunk never spans a heading. Policy documents are written so that each
    numbered section is a self-contained rule, so a section boundary is a real
    semantic boundary - and it means the `section` recorded on a chunk is always
    the section the text actually came from, which is what makes citations honest.
  * Within a section, paragraphs are packed up to CHUNK_SIZE so a rule and its
    conditions stay together instead of being split across chunks.
  * Only a paragraph longer than CHUNK_SIZE is hard-split, with CHUNK_OVERLAP
    characters repeated so a sentence cut in half is still retrievable.
"""
import re

HEADING = re.compile(
    r"^\s*(?:#{1,6}\s+.+"          # markdown heading
    r"|\d+(?:\.\d+)*[.)]\s+\S.{0,80}"  # "1." / "2.3)" numbered heading
    r"|[A-Z][A-Z \-/&]{4,60})\s*$"     # SHOUTED HEADING
)

MIN_CHUNK_CHARS = 30


def _is_heading(paragraph: str) -> bool:
    return "\n" not in paragraph and bool(HEADING.match(paragraph))


def _heading_text(paragraph: str) -> str:
    return paragraph.lstrip("#").strip()[:120]


def chunk(text: str, size: int, overlap: int) -> list[dict]:
    """Split text into [{'text': ..., 'section': ...}] chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[dict] = []
    buffer = ""
    section = ""

    def flush() -> None:
        nonlocal buffer
        if len(buffer.strip()) >= MIN_CHUNK_CHARS:
            chunks.append({"text": buffer.strip(), "section": section})
        buffer = ""

    for paragraph in paragraphs:
        if _is_heading(paragraph):
            flush()                       # a chunk never crosses a section boundary
            section = _heading_text(paragraph)
            buffer = paragraph            # keep the heading as the chunk's first line
            continue

        # Hard-split a paragraph that cannot fit in one chunk on its own.
        while len(paragraph) > size:
            split_at = paragraph.rfind(" ", 0, size)
            if split_at < size * 0.6:
                split_at = size
            head = paragraph[:split_at].strip()
            if buffer:
                flush()
            buffer = head
            flush()
            paragraph = paragraph[max(split_at - overlap, 0):].strip()

        if not buffer:
            buffer = paragraph
        elif len(buffer) + len(paragraph) + 2 <= size:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            tail = buffer[-overlap:] if overlap else ""
            flush()
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    flush()
    return chunks
