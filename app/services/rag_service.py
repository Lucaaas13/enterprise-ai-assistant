from sqlalchemy.orm import Session

from app.services.llm_service import generate_answer
from app.services.search_service import hybrid_search


def answer_question(
    db: Session,
    question: str,
    top_k: int = 5,
) -> dict:
    search_results = hybrid_search(
        db=db,
        query=question,
        top_k=top_k,
        candidate_k=10,
    )

    if not search_results:
        return {
            "answer": (
                "Je ne trouve pas cette information "
                "dans les documents fournis."
            ),
            "sources": [],
        }

    context_parts = []

    for index, result in enumerate(
        search_results,
        start=1,
    ):
        context_parts.append(
            f"""
        ===== SOURCE {index} =====
        Document : {result["document_title"]}
        Page : {result["page_number"]}

        PASSAGE :
        {result["content"]}

        ===== FIN SOURCE {index} =====
        """
        )

    context = "\n\n".join(context_parts)

    answer = generate_answer(
        question=question,
        context=context,
    )

    NO_ANSWER = (
        "Je ne trouve pas cette information "
        "dans les documents fournis."
    )

    if NO_ANSWER.lower() in answer.lower():
        return {
            "answer": NO_ANSWER,
            "sources": [],
        }

    sources = [
        {
            "chunk_id": result["chunk_id"],
            "document_id": result["document_id"],
            "document_title": result["document_title"],
            "page_number": result["page_number"],
            "score": result["score"],
        }
        for result in search_results
    ]

    return {
        "answer": answer,
        "sources": sources,
    }