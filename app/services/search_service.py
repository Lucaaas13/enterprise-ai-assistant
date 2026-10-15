import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chunk import Chunk
from app.models.document import Document
from app.services.embedding_service import create_embedding


def semantic_search(
    db: Session,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    query_embedding = create_embedding(query).tolist()

    distance = Chunk.embedding.cosine_distance(query_embedding)

    statement = (
        select(
            Chunk,
            Document.title,
            distance.label("distance"),
        )
        .join(
            Document,
            Document.id == Chunk.document_id,
        )
        .where(Chunk.embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )

    rows = db.execute(statement).all()
    results = []

    for chunk, document_title, cosine_distance in rows:
        score = 1 - float(cosine_distance)

        results.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "document_title": document_title,
                "page_number": chunk.page_number,
                "content": chunk.content,
                "score": score,
            }
        )

    return results


STOP_WORDS = {
    "quel",
    "quelle",
    "quels",
    "quelles",
    "est",
    "sont",
    "dans",
    "pour",
    "avec",
    "utilise",
    "utilisee",
    "utilisees",
    "utilises",
    "utilisé",
    "utilisée",
    "utilisées",
    "utilisés",
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "du",
    "de",
    "au",
    "aux",
    "projet",
}


def build_lexical_query(query: str) -> str:
    words = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ0-9+#.-]+",
        query.lower(),
    )

    terms = [word for word in words if word not in STOP_WORDS and len(word) >= 3]

    return " OR ".join(terms)


def lexical_search(
    db: Session,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    lexical_query = build_lexical_query(query)

    if not lexical_query:
        return []

    ts_query = func.websearch_to_tsquery(
        "simple",
        lexical_query,
    )

    ts_vector = func.to_tsvector(
        "simple",
        Chunk.content,
    )

    lexical_score = func.ts_rank_cd(
        ts_vector,
        ts_query,
    )

    statement = (
        select(
            Chunk,
            Document.title,
            lexical_score.label("lexical_score"),
        )
        .join(
            Document,
            Document.id == Chunk.document_id,
        )
        .where(ts_vector.op("@@")(ts_query))
        .order_by(lexical_score.desc())
        .limit(top_k)
    )

    rows = db.execute(statement).all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "document_title": document_title,
            "page_number": chunk.page_number,
            "content": chunk.content,
            "score": float(score),
        }
        for chunk, document_title, score in rows
    ]


def hybrid_search(
    db: Session,
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[dict]:
    vector_results = semantic_search(
        db=db,
        query=query,
        top_k=candidate_k,
    )

    lexical_results = lexical_search(
        db=db,
        query=query,
        top_k=candidate_k,
    )

    rrf_k = 60

    combined = {}

    for rank, result in enumerate(
        vector_results,
        start=1,
    ):
        chunk_id = result["chunk_id"]

        combined[chunk_id] = {
            **result,
            "rrf_score": 0.0,
        }

        combined[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    for rank, result in enumerate(
        lexical_results,
        start=1,
    ):
        chunk_id = result["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = {
                **result,
                "rrf_score": 0.0,
            }

        combined[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

    ranked_results = sorted(
        combined.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )
    for result in ranked_results:
        result["score"] = result["rrf_score"]

    return ranked_results[:top_k]
