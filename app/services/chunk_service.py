from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.services.embedding_service import (
    create_embedding,
    get_embedding_model,
)


CHUNK_MAX_TOKENS = 100
CHUNK_OVERLAP_TOKENS = 20


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    if overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be smaller than max_tokens"
        )

    if not text.strip():
        return []

    tokenizer = get_embedding_model().tokenizer

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = encoding["offset_mapping"]

    if not offsets:
        return []

    chunks = []

    step = max_tokens - overlap_tokens
    start_token = 0
    token_count = len(offsets)

    while start_token < token_count:
        end_token = min(
            start_token + max_tokens,
            token_count,
        )

        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]

        chunk = text[
            char_start:char_end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end_token >= token_count:
            break

        start_token += step

    return chunks


def create_chunks_for_document(
    db: Session,
    document_id: int,
    pages: list[tuple[int, str]],
) -> list[Chunk]:
    chunks = []
    chunk_index = 0

    for page_number, page_text in pages:
        page_chunks = chunk_text(page_text)

        for content in page_chunks:
            embedding = create_embedding(content)

            chunk = Chunk(
                document_id=document_id,
                page_number=page_number,
                chunk_index=chunk_index,
                content=content,
                embedding=embedding.tolist(),
            )

            chunks.append(chunk)
            chunk_index += 1

    db.add_all(chunks)
    db.commit()

    return chunks


def get_chunks_by_document(
    db: Session,
    document_id: int,
) -> list[Chunk]:
    statement = (
        select(Chunk)
        .where(
            Chunk.document_id == document_id
        )
        .order_by(Chunk.chunk_index)
    )

    return list(
        db.scalars(statement).all()
    )