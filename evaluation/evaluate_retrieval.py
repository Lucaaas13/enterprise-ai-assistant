import json
import time
from pathlib import Path

from app.database.database import SessionLocal
from app.services.search_service import semantic_search


QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def find_expected_rank(
    results: list[dict],
    expected_terms: list[str],
) -> int | None:
    """
    Retourne le rang du premier chunk contenant tous les termes attendus.

    Exemple :
    - rang 1 = le bon passage est le premier résultat
    - rang 4 = le bon passage est le quatrième résultat
    - None = aucun passage correspondant dans le Top 10
    """

    if not expected_terms:
        return None

    for index, result in enumerate(results, start=1):
        content = result["content"].lower()

        if all(
            term.lower() in content
            for term in expected_terms
        ):
            return index

    return None


def main():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    db = SessionLocal()

    total_answerable = 0

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    latencies = []

    try:
        # Le premier appel charge le modèle Sentence Transformers.
        # On ne veut pas inclure ce temps de chargement
        # dans les mesures de latence.
        print("Warm-up du modèle d'embeddings...")

        semantic_search(
            db=db,
            query="warmup",
            top_k=1,
        )

        print("Warm-up terminé.")

        for item in questions:
            question = item["question"]

            start = time.perf_counter()

            results = semantic_search(
                db=db,
                query=question,
                top_k=10,
            )

            latency = time.perf_counter() - start

            latencies.append(latency)

            print()
            print("=" * 70)
            print("QUESTION :", question)

            if results:
                print(
                    "TOP SCORE :",
                    round(results[0]["score"], 4),
                )

            print(
                "LATENCE :",
                round(latency, 3),
                "s",
            )

            # Question dont la réponse existe dans le document
            if item["should_answer"]:
                total_answerable += 1

                rank = find_expected_rank(
                    results,
                    item["expected_terms"],
                )

                if rank is None:
                    print(
                        "PASSAGE ATTENDU : "
                        "NON TROUVE DANS LE TOP 10"
                    )

                    print()
                    print("TOP 3 RETOURNES :")

                    for index, result in enumerate(
                        results[:3],
                        start=1,
                    ):
                        preview = (
                            result["content"][:300]
                            .replace("\n", " ")
                        )

                        print(
                            f"{index}. "
                            f"score={result['score']:.4f} | "
                            f"page={result['page_number']}"
                        )

                        print(preview)
                        print()

                else:
                    print(
                        "RANG DU PASSAGE ATTENDU :",
                        rank,
                    )

                    if rank <= 1:
                        recall_at_1 += 1

                    if rank <= 3:
                        recall_at_3 += 1

                    if rank <= 5:
                        recall_at_5 += 1

                    if rank <= 10:
                        recall_at_10 += 1

            # Question dont la réponse n'existe pas dans le document
            else:
                print("QUESTION HORS DOCUMENT")

    finally:
        db.close()

    print()
    print("=" * 70)
    print("RESULTATS")

    if total_answerable:
        print(
            "Recall@1 :",
            f"{recall_at_1 / total_answerable:.2%}",
        )

        print(
            "Recall@3 :",
            f"{recall_at_3 / total_answerable:.2%}",
        )

        print(
            "Recall@5 :",
            f"{recall_at_5 / total_answerable:.2%}",
        )

        print(
            "Recall@10 :",
            f"{recall_at_10 / total_answerable:.2%}",
        )

    if latencies:
        average_latency = sum(latencies) / len(latencies)

        print(
            "Latence moyenne :",
            f"{average_latency:.3f}s",
        )


if __name__ == "__main__":
    main()