from app.database.database import SessionLocal
from app.services.search_service import (
    lexical_search,
    semantic_search,
    hybrid_search,
)


QUESTION = (
    "Quel outil est utilisé pour exécuter "
    "le LLM localement ?"
)


def show_results(name, results):
    print()
    print("=" * 70)
    print(name)

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{index}. "
            f"chunk={result['chunk_id']} | "
            f"page={result['page_number']} | "
            f"score={result['score']:.6f}"
        )

        print(
            result["content"][:250]
            .replace("\n", " ")
        )

        print()


def main():
    db = SessionLocal()

    try:
        semantic = semantic_search(
            db=db,
            query=QUESTION,
            top_k=5,
        )

        lexical = lexical_search(
            db=db,
            query=QUESTION,
            top_k=5,
        )

        hybrid = hybrid_search(
            db=db,
            query=QUESTION,
            top_k=5,
            candidate_k=10,
        )

        show_results(
            "SEMANTIC",
            semantic,
        )

        show_results(
            "LEXICAL",
            lexical,
        )

        show_results(
            "HYBRID",
            hybrid,
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()