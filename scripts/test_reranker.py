from app.database.database import SessionLocal
from app.services.search_service import hybrid_search
from app.services.reranker_service import rerank_results


QUESTIONS = [
    "Quel outil est utilisé pour exécuter le LLM localement ?",
    "Quelle technologie est utilisée pour conteneuriser le projet ?",
    "Que faut-il faire avant d'utiliser LangGraph ou Azure ?",
]


def main():
    db = SessionLocal()

    try:
        for question in QUESTIONS:
            print()
            print("=" * 70)
            print("QUESTION :", question)

            candidates = hybrid_search(
                db=db,
                query=question,
                top_k=10,
                candidate_k=10,
            )

            results = rerank_results(
                query=question,
                results=candidates,
                top_k=3,
            )

            for index, result in enumerate(
                results,
                start=1,
            ):
                print()
                print(
                    f"{index}. "
                    f"chunk={result['chunk_id']} | "
                    f"page={result['page_number']} | "
                    f"reranker="
                    f"{result['reranker_score']:.4f}"
                )

                print(
                    result["content"][:300]
                    .replace("\n", " ")
                )

    finally:
        db.close()


if __name__ == "__main__":
    main()