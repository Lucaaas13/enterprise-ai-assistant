import json
import time
from datetime import datetime
from pathlib import Path

from app.database.database import SessionLocal
from app.services.search_service import (
    semantic_search,
    lexical_search,
    hybrid_search,
)
from app.services.reranker_service import rerank_results


QUESTIONS_FILE = Path(__file__).parent / "questions.json"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    RESULTS_DIR
    / "retrieval_strategies_comparison.json"
)


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


def compute_metrics(
    ranks: list[int | None],
    latencies: list[float],
) -> dict:
    total = len(ranks)

    if total == 0:
        return {
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "average_latency_seconds": 0.0,
        }

    def recall_at(k: int) -> float:
        successes = sum(
            1
            for rank in ranks
            if rank is not None and rank <= k
        )

        return successes / total

    average_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    return {
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "average_latency_seconds": (
            average_latency
        ),
    }


def run_strategy(
    name: str,
    questions: list[dict],
    search_function,
) -> dict:
    print()
    print("=" * 80)
    print("STRATEGIE :", name)
    print("=" * 80)

    db = SessionLocal()

    ranks = []
    latencies = []
    detailed_results = []

    try:
        for item in questions:
            if not item["should_answer"]:
                continue

            question = item["question"]

            start = time.perf_counter()

            results = search_function(
                db,
                question,
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

            print()
            print(
                f"[{item['id']}]",
                question,
            )

            if rank is None:
                print(
                    "Rang attendu : NON TROUVE"
                )
            else:
                print(
                    "Rang attendu :",
                    rank,
                )

            print(
                "Latence :",
                f"{latency:.3f}s",
            )

            detailed_results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "expected_pages": (
                        item["expected_pages"]
                    ),
                    "rank": rank,
                    "latency_seconds": latency,
                    "top_results": [
                        {
                            "chunk_id": result[
                                "chunk_id"
                            ],
                            "page_number": result[
                                "page_number"
                            ],
                            "score": result.get(
                                "reranker_score",
                                result.get("score"),
                            ),
                        }
                        for result in results[:10]
                    ],
                }
            )

    finally:
        db.close()

    metrics = compute_metrics(
        ranks,
        latencies,
    )

    print()
    print("-" * 80)

    print(
        "Recall@1 :",
        f"{metrics['recall_at_1']:.2%}",
    )

    print(
        "Recall@3 :",
        f"{metrics['recall_at_3']:.2%}",
    )

    print(
        "Recall@5 :",
        f"{metrics['recall_at_5']:.2%}",
    )

    print(
        "Recall@10 :",
        f"{metrics['recall_at_10']:.2%}",
    )

    print(
        "Latence moyenne :",
        (
            f"{metrics['average_latency_seconds']:.3f}s"
        ),
    )

    return {
        "name": name,
        "metrics": metrics,
        "results": detailed_results,
    }


def main():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    answerable_questions = [
        item
        for item in questions
        if item["should_answer"]
    ]

    print(
        "Questions répondables :",
        len(answerable_questions),
    )

    print()
    print(
        "Warm-up embeddings et reranker..."
    )

    warmup_db = SessionLocal()

    try:
        semantic_search(
            db=warmup_db,
            query="warmup",
            top_k=10,
        )

        warmup_hybrid = hybrid_search(
            db=warmup_db,
            query="warmup",
            top_k=10,
            candidate_k=10,
        )

        if warmup_hybrid:
            rerank_results(
                query="warmup",
                results=warmup_hybrid,
                top_k=10,
            )

    finally:
        warmup_db.close()

    print("Warm-up terminé.")

    strategies = []

    strategies.append(
        run_strategy(
            name="vector_semantic",
            questions=questions,
            search_function=(
                lambda db, query:
                semantic_search(
                    db=db,
                    query=query,
                    top_k=10,
                )
            ),
        )
    )

    strategies.append(
        run_strategy(
            name="lexical_postgresql_fts",
            questions=questions,
            search_function=(
                lambda db, query:
                lexical_search(
                    db=db,
                    query=query,
                    top_k=10,
                )
            ),
        )
    )

    strategies.append(
        run_strategy(
            name="hybrid_rrf",
            questions=questions,
            search_function=(
                lambda db, query:
                hybrid_search(
                    db=db,
                    query=query,
                    top_k=10,
                    candidate_k=10,
                )
            ),
        )
    )

    def hybrid_with_reranker(
        db,
        query,
    ):
        candidates = hybrid_search(
            db=db,
            query=query,
            top_k=10,
            candidate_k=10,
        )

        return rerank_results(
            query=query,
            results=candidates,
            top_k=10,
        )

    strategies.append(
        run_strategy(
            name="hybrid_rrf_plus_reranker",
            questions=questions,
            search_function=(
                hybrid_with_reranker
            ),
        )
    )

    benchmark = {
        "name": (
            "retrieval_strategies_comparison"
        ),
        "created_at": (
            datetime.now().isoformat()
        ),
        "configuration": {
            "dataset_questions": len(
                questions
            ),
            "answerable_questions": len(
                answerable_questions
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
            "hybrid_candidate_k": 10,
            "rrf_k": 60,
            "reranker_model": (
                "cross-encoder/"
                "mmarco-mMiniLMv2-"
                "L12-H384-v1"
            ),
        },
        "strategies": strategies,
    }

    with open(
        OUTPUT_FILE,
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
    print("=" * 80)
    print("COMPARAISON FINALE")
    print("=" * 80)

    print(
        f"{'STRATEGIE':<30}"
        f"{'R@1':>8}"
        f"{'R@3':>8}"
        f"{'R@5':>8}"
        f"{'R@10':>8}"
        f"{'LATENCE':>12}"
    )

    print("-" * 74)

    for strategy in strategies:
        metrics = strategy["metrics"]

        print(
            f"{strategy['name']:<30}"
            f"{metrics['recall_at_1']:>8.2%}"
            f"{metrics['recall_at_3']:>8.2%}"
            f"{metrics['recall_at_5']:>8.2%}"
            f"{metrics['recall_at_10']:>8.2%}"
            f"{metrics['average_latency_seconds']:>11.3f}s"
        )

    print()
    print(
        "Benchmark sauvegardé :",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()