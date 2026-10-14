import json
import time
from datetime import datetime
from pathlib import Path

from app.database.database import SessionLocal
from app.services.search_service import semantic_search


QUESTIONS_FILE = Path(__file__).parent / "questions.json"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def find_expected_rank(
    results: list[dict],
    expected_pages: list[int],
) -> int | None:
    if not expected_pages:
        return None

    for index, result in enumerate(
        results,
        start=1,
    ):
        if result["page_number"] in expected_pages:
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
    evaluation_results = []

    try:
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

            latency = (
                time.perf_counter()
                - start
            )

            latencies.append(latency)

            print()
            print("=" * 70)
            print("QUESTION :", question)

            top_score = None

            if results:
                top_score = results[0]["score"]

                print(
                    "TOP SCORE :",
                    round(top_score, 4),
                )

            print(
                "LATENCE :",
                round(latency, 3),
                "s",
            )

            if item["should_answer"]:
                total_answerable += 1

                rank = find_expected_rank(
                    results,
                    item["expected_pages"],
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

                evaluation_results.append(
                    {
                        "id": item["id"],
                        "question": question,
                        "should_answer": True,
                        "expected_pages": item[
                            "expected_pages"
                        ],
                        "rank": rank,
                        "top_score": top_score,
                        "latency_seconds": latency,
                    }
                )

            else:
                print("QUESTION HORS DOCUMENT")

                evaluation_results.append(
                    {
                        "id": item["id"],
                        "question": question,
                        "should_answer": False,
                        "expected_pages": [],
                        "rank": None,
                        "top_score": top_score,
                        "latency_seconds": latency,
                    }
                )

    finally:
        db.close()

    print()
    print("=" * 70)
    print("RESULTATS")

    metrics = {}

    if total_answerable:
        recall_1 = (
            recall_at_1
            / total_answerable
        )

        recall_3 = (
            recall_at_3
            / total_answerable
        )

        recall_5 = (
            recall_at_5
            / total_answerable
        )

        recall_10 = (
            recall_at_10
            / total_answerable
        )

        metrics["recall_at_1"] = recall_1
        metrics["recall_at_3"] = recall_3
        metrics["recall_at_5"] = recall_5
        metrics["recall_at_10"] = recall_10

        print(
            "Recall@1 :",
            f"{recall_1:.2%}",
        )

        print(
            "Recall@3 :",
            f"{recall_3:.2%}",
        )

        print(
            "Recall@5 :",
            f"{recall_5:.2%}",
        )

        print(
            "Recall@10 :",
            f"{recall_10:.2%}",
        )

    average_latency = 0.0

    if latencies:
        average_latency = (
            sum(latencies)
            / len(latencies)
        )

        metrics[
            "average_latency_seconds"
        ] = average_latency

        print(
            "Latence moyenne :",
            f"{average_latency:.3f}s",
        )

    benchmark = {
        "name": "vector_retrieval_baseline",
        "created_at": datetime.now().isoformat(),
        "configuration": {
            "retrieval": (
                "pgvector_semantic_search"
            ),
            "embedding_model": (
                "sentence-transformers/"
                "paraphrase-multilingual-"
                "MiniLM-L12-v2"
            ),
            "embedding_dimension": 384,
            "chunk_size_tokens": 100,
            "chunk_overlap_tokens": 20,
            "max_results": 10,
        },
        "dataset": {
            "total_questions": len(
                questions
            ),
            "answerable_questions": (
                total_answerable
            ),
            "unanswerable_questions": (
                len(questions)
                - total_answerable
            ),
        },
        "metrics": metrics,
        "results": evaluation_results,
    }

    output_file = (
        RESULTS_DIR
        / "retrieval_vector_baseline.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            benchmark,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Benchmark sauvegarde :",
        output_file,
    )


if __name__ == "__main__":
    main()