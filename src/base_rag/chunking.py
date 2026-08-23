from __future__ import annotations

import hashlib
import re

from base_rag.models import Chunk, SourceDocument


def chunk_documents(documents: list[SourceDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        ordinal = 0
        for section, text in _sections(document):
            for piece in _split_text(text, chunk_size, overlap):
                digest = hashlib.sha256(f"{document.source_id}|{document.page}|{section}|{ordinal}|{piece}".encode("utf-8")).hexdigest()[:20]
                chunks.append(Chunk(digest, piece, document.source_id, document.source_path, document.media_type, ordinal, document.title, document.page, section, document.metadata))
                ordinal += 1
    return chunks


def _sections(document: SourceDocument) -> list[tuple[str | None, str]]:
    if document.media_type != "markdown":
        return [(document.section, document.text)]
    sections: list[tuple[str | None, str]] = []
    heading = document.section
    buffer: list[str] = []
    for line in document.text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            if "\n".join(buffer).strip():
                sections.append((heading, "\n".join(buffer).strip()))
            heading = match.group(1)
            buffer = []
        else:
            buffer.append(line)
    if "\n".join(buffer).strip():
        sections.append((heading, "\n".join(buffer).strip()))
    return sections or [(document.section, document.text)]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n\n", start + chunk_size // 2, end), text.rfind("。", start + chunk_size // 2, end), text.rfind("\n", start + chunk_size // 2, end), text.rfind(" ", start + chunk_size // 2, end))
            if boundary > start:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces
