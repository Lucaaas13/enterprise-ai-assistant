import json
import time
from datetime import datetime
from pathlib import Path

from app.database.database import SessionLocal
from app.services.rag_service import answer_question


QUESTIONS_FILE = Path(__file__).parent / "questions.json"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

NO_ANSWER = (
    "Je ne trouve pas cette information "
    "dans les documents fournis."
)


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
    )


def answer_contains_expected_terms(
    answer: str,
    expected_terms: list[str],
) -> bool:
    if not expected_terms:
        return True

    normalized_answer = normalize_text(answer)

    return all(
        normalize_text(term) in normalized_answer
        for term in expected_terms
    )


def main():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    db = SessionLocal()

    total = len(questions)

    correct = 0

    answerable_total = 0
    answerable_correct = 0

    unanswerable_total = 0
    unanswerable_correct = 0

    hallucinations = 0

    latencies = []

    evaluation_results = []

    try:
        print("Warm-up du pipeline RAG...")

        answer_question(
            db=db,
            question=(
                "Quelle technologie est utilisée "
                "dans le projet ?"
            ),
            top_k=5,
        )

        print("Warm-up terminé.")

        for item in questions:
            question = item["question"]

            print()
            print("=" * 70)
            print("QUESTION :", question)

            start = time.perf_counter()

            result = answer_question(
                db=db,
                question=question,
                top_k=5,
            )

            latency = (
                time.perf_counter()
                - start
            )

            latencies.append(latency)

            answer = result["answer"]
            sources = result["sources"]

            print("REPONSE :", answer)

            print(
                "LATENCE :",
                round(latency, 3),
                "s",
            )

            success = False
            evaluation_type = ""

            if item["should_answer"]:
                answerable_total += 1

                is_abstention = (
                    normalize_text(NO_ANSWER)
                    in normalize_text(answer)
                )

                if is_abstention:
                    success = False
                else:
                    success = (
                        answer_contains_expected_terms(
                            answer,
                            item["expected_terms"],
                        )
                    )

                evaluation_type = "answer"

                if success:
                    correct += 1
                    answerable_correct += 1

                    print("RESULTAT : OK")

                else:
                    print("RESULTAT : ECHEC")

                    print(
                        "REPONSE ATTENDUE :",
                        item["reference_answer"],
                    )

            else:
                unanswerable_total += 1

                refused = (
                    normalize_text(NO_ANSWER)
                    in normalize_text(answer)
                )

                success = refused
                evaluation_type = "abstention"

                if refused:
                    correct += 1
                    unanswerable_correct += 1

                    print("ABSTENTION : OK")

                else:
                    hallucinations += 1

                    print(
                        "ABSTENTION : ECHEC "
                        "- hallucination potentielle"
                    )

            evaluation_results.append(
                {
                    "id": item["id"],
                    "question": question,
                    "should_answer": (
                        item["should_answer"]
                    ),
                    "reference_answer": (
                        item["reference_answer"]
                    ),
                    "expected_terms": (
                        item["expected_terms"]
                    ),
                    "expected_pages": (
                        item["expected_pages"]
                    ),
                    "answer": answer,
                    "sources": sources,
                    "latency_seconds": latency,
                    "evaluation_type": (
                        evaluation_type
                    ),
                    "success": success,
                }
            )

    finally:
        db.close()

    print()
    print("=" * 70)
    print("RESULTATS RAG")

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    answerable_accuracy = (
        answerable_correct
        / answerable_total
        if answerable_total
        else 0.0
    )

    abstention_accuracy = (
        unanswerable_correct
        / unanswerable_total
        if unanswerable_total
        else 0.0
    )

    average_latency = (
        sum(latencies)
        / len(latencies)
        if latencies
        else 0.0
    )

    print(
        "Accuracy globale :",
        f"{accuracy:.2%}",
    )

    print(
        "Questions répondables :",
        f"{answerable_accuracy:.2%}",
    )

    print(
        "Abstentions correctes :",
        f"{abstention_accuracy:.2%}",
    )

    print(
        "Hallucinations potentielles :",
        hallucinations,
    )

    print(
        "Latence moyenne :",
        f"{average_latency:.3f}s",
    )

    benchmark = {
        "name": "rag_baseline",
        "created_at": datetime.now().isoformat(),
        "configuration": {
            "embedding_model": (
                "sentence-transformers/"
                "paraphrase-multilingual-"
                "MiniLM-L12-v2"
            ),
            "embedding_dimension": 384,
            "chunk_size_tokens": 100,
            "chunk_overlap_tokens": 20,
            "retrieval": "hybrid_rrf",
            "candidate_k": 10,
            "top_k": 5,
            "reranker": None,
            "llm": "qwen2.5:3b",
            "llm_runtime": "ollama",
        },
        "dataset": {
            "total_questions": total,
            "answerable_questions": (
                answerable_total
            ),
            "unanswerable_questions": (
                unanswerable_total
            ),
        },
        "metrics": {
            "accuracy": accuracy,
            "answerable_accuracy": (
                answerable_accuracy
            ),
            "abstention_accuracy": (
                abstention_accuracy
            ),
            "hallucinations": hallucinations,
            "average_latency_seconds": (
                average_latency
            ),
        },
        "results": evaluation_results,
    }

    output_file = (
        RESULTS_DIR
        / "rag_baseline.json"
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