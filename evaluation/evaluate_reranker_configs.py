import json
import time
from pathlib import Path

from app.database.database import SessionLocal
from app.services.search_service import hybrid_search
from app.services.reranker_service import rerank_results


QUESTIONS_FILE = Path(__file__).parent / "questions.json"


CONFIGURATIONS = [
    {
        "candidate_k": 5,
        "rerank_top_k": 5,
    },
    {
        "candidate_k": 10,
        "rerank_top_k": 5,
    },
    {
        "candidate_k": 20,
        "rerank_top_k": 5,
    },
    {
        "candidate_k": 10,
        "rerank_top_k": 3,
    },
    {
        "candidate_k": 10,
        "rerank_top_k": 10,
    },
]


def find_expected_rank(
    results: list[dict],
    expected_pages: list[int],
) -> int | None:
    for index, result in enumerate(
        results,
        start=1,
    ):
        if result["page_number"] in expected_pages:
            return index

    return None


def evaluate_configuration(
    questions,
    candidate_k,
    rerank_top_k,
):
    db = SessionLocal()

    ranks = []
    latencies = []

    try:
        for item in questions:
            if not item["should_answer"]:
                continue

            start = time.perf_counter()

            candidates = hybrid_search(
                db=db,
                query=item["question"],
                top_k=candidate_k,
                candidate_k=candidate_k,
            )

            results = rerank_results(
                query=item["question"],
                results=candidates,
                top_k=rerank_top_k,
            )

            latency = (
                time.perf_counter()
                - start
            )

            rank = find_expected_rank(
                results,
                item["expected_pages"],
            )

            ranks.append(rank)
            latencies.append(latency)

    finally:
        db.close()

    total = len(ranks)

    def recall_at(k):
        return (
            sum(
                1
                for rank in ranks
                if rank is not None
                and rank <= k
            )
            / total
        )

    return {
        "candidate_k": candidate_k,
        "rerank_top_k": rerank_top_k,
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "average_latency": (
            sum(latencies)
            / len(latencies)
        ),
    }


def main():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    print("Warm-up reranker...")

    db = SessionLocal()

    try:
        candidates = hybrid_search(
            db=db,
            query="warmup",
            top_k=10,
            candidate_k=10,
        )

        rerank_results(
            query="warmup",
            results=candidates,
            top_k=5,
        )

    finally:
        db.close()

    print("Warm-up terminé.")

    results = []

    for config in CONFIGURATIONS:
        result = evaluate_configuration(
            questions=questions,
            candidate_k=config["candidate_k"],
            rerank_top_k=config["rerank_top_k"],
        )

        results.append(result)

    print()
    print("=" * 78)
    print("COMPARAISON RERANKER")
    print("=" * 78)

    print(
        f"{'CANDIDATS':>10}"
        f"{'TOP K':>10}"
        f"{'R@1':>10}"
        f"{'R@3':>10}"
        f"{'R@5':>10}"
        f"{'LATENCE':>14}"
    )

    print("-" * 78)

    for result in results:
        print(
            f"{result['candidate_k']:>10}"
            f"{result['rerank_top_k']:>10}"
            f"{result['recall_at_1']:>10.2%}"
            f"{result['recall_at_3']:>10.2%}"
            f"{result['recall_at_5']:>10.2%}"
            f"{result['average_latency']:>13.3f}s"
        )


if __name__ == "__main__":
    main()