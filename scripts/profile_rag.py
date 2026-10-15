import time

from app.database.database import SessionLocal
from app.services.llm_service import call_ollama
from app.services.search_service import hybrid_search

QUESTION = "Quelle base de données vectorielle est utilisée dans le projet ?"


def main():
    db = SessionLocal()

    try:
        print("Warm-up...")

        hybrid_search(
            db=db,
            query="warmup",
            top_k=1,
            candidate_k=10,
        )

        print("Warm-up terminé.")
        print()
        print("Warm-up Ollama...")

        call_ollama(
            question="Test",
            context="Source : test.",
        )

        print("Warm-up Ollama terminé.")
        print()

        # 1. Mesure du retrieval
        start_retrieval = time.perf_counter()

        search_results = hybrid_search(
            db=db,
            query=QUESTION,
            top_k=5,
            candidate_k=10,
        )

        retrieval_time = time.perf_counter() - start_retrieval

        # 2. Construction du contexte
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

        # 3. Mesure du LLM
        start_llm = time.perf_counter()

        ollama_result = call_ollama(
            question=QUESTION,
            context=context,
        )

        answer = ollama_result["response"]

        llm_time = time.perf_counter() - start_llm

        total_time = retrieval_time + llm_time

        # 4. Affichage de la réponse
        print("QUESTION :")
        print(QUESTION)

        print()
        print("REPONSE :")
        print(answer)

        print()
        print("=" * 70)
        print("PROFIL")

        print(
            "Retrieval :",
            f"{retrieval_time:.3f}s",
        )

        print(
            "LLM :",
            f"{llm_time:.3f}s",
        )

        print(
            "Total :",
            f"{total_time:.3f}s",
        )

        if total_time > 0:
            print(
                "Part LLM :",
                f"{llm_time / total_time:.2%}",
            )

        # 5. Metriques internes Ollama
        print()
        print("=" * 70)
        print("OLLAMA")

        total_duration = ollama_result.get("total_duration", 0) / 1_000_000_000

        load_duration = ollama_result.get("load_duration", 0) / 1_000_000_000

        prompt_eval_duration = (
            ollama_result.get(
                "prompt_eval_duration",
                0,
            )
            / 1_000_000_000
        )

        eval_duration = (
            ollama_result.get(
                "eval_duration",
                0,
            )
            / 1_000_000_000
        )

        prompt_tokens = ollama_result.get(
            "prompt_eval_count",
            0,
        )

        generated_tokens = ollama_result.get(
            "eval_count",
            0,
        )

        print(
            "Chargement modele :",
            f"{load_duration:.3f}s",
        )

        print(
            "Tokens prompt :",
            prompt_tokens,
        )

        print(
            "Evaluation prompt :",
            f"{prompt_eval_duration:.3f}s",
        )

        print(
            "Tokens generes :",
            generated_tokens,
        )

        print(
            "Generation :",
            f"{eval_duration:.3f}s",
        )

        print(
            "Duree Ollama totale :",
            f"{total_duration:.3f}s",
        )

        if eval_duration > 0:
            print(
                "Vitesse generation :",
                (f"{generated_tokens / eval_duration:.2f} tokens/s"),
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()
