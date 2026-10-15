from app.database.database import SessionLocal
from app.models.chunk import Chunk
from app.services.llm_service import generate_answer

TESTS = [
    {
        "question": ("Quel outil est utilisé pour exécuter le LLM localement ?"),
        "chunk_id": 37,
    },
    {
        "question": ("Que faut-il faire avant d'utiliser LangGraph ou Azure ?"),
        "chunk_id": 17,
    },
]


def main():
    db = SessionLocal()

    try:
        for test in TESTS:
            chunk = db.get(
                Chunk,
                test["chunk_id"],
            )

            if chunk is None:
                print(f"Chunk {test['chunk_id']} introuvable")
                continue

            context = f"""
===== SOURCE 1 =====
Page : {chunk.page_number}

PASSAGE :
{chunk.content}

===== FIN SOURCE 1 =====
"""

            print()
            print("=" * 70)
            print("QUESTION :")
            print(test["question"])

            print()
            print("CHUNK UTILISE :")
            print(chunk.content)

            answer = generate_answer(
                question=test["question"],
                context=context,
            )

            print()
            print("REPONSE QWEN :")
            print(answer)

    finally:
        db.close()


if __name__ == "__main__":
    main()
