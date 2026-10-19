from app.agents.graph import run_agent

QUESTIONS = [
    "Quel outil est utilisé pour exécuter le LLM localement ?",
    "Comment fonctionne le RAG dans le projet ?",
    "À quoi sert MLflow ?",
    "Combien de documents sont enregistrés dans la base ?",
    "Combien de documents y a-t-il dans chaque catégorie ?",
    "Quels sont les documents les plus récents ?",
    "Explique le rôle de pgvector.",
    "Quelle stratégie de recherche hybride est prévue ?",
    "Génère-moi un rapport sur les technologies du projet.",
    "Fais une synthèse structurée des différentes phases du projet.",
]


def main():
    for question in QUESTIONS:
        print()
        print("=" * 70)
        print("QUESTION :", question)

        result = run_agent(question)

        print(
            "ROUTE :",
            result["route"],
        )

        print(
            "DATABASE ACTION :",
            result.get("database_action"),
        )

        print(
            "TRACE :",
            " -> ".join(result["trace"]),
        )

        print(
            "STEPS :",
            result["steps"],
        )

        print(
            "RESULTAT :",
            result["tool_result"],
        )

        print(
            "REPONSE :",
            result["answer"],
        )


if __name__ == "__main__":
    main()
