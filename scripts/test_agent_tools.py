from app.agents.schemas import (
    DatabaseQueryInput,
    GenerateReportInput,
    SearchDocumentsInput,
)
from app.agents.tools import (
    generate_report,
    query_database,
    search_documents,
)


def main():
    print("=" * 60)
    print("SEARCH DOCUMENTS")
    print("=" * 60)

    search_result = search_documents(
        SearchDocumentsInput(
            query=("Quel outil est utilisé pour exécuter le LLM localement ?"),
            top_k=3,
        )
    )

    for result in search_result.results:
        print()
        print(
            "Document :",
            result.document_title,
        )
        print(
            "Page :",
            result.page_number,
        )
        print(
            "Score :",
            round(
                result.score,
                4,
            ),
        )

    print()
    print("=" * 60)
    print("QUERY DATABASE")
    print("=" * 60)

    count_result = query_database(DatabaseQueryInput(action="document_count"))

    print(count_result.model_dump())

    categories_result = query_database(DatabaseQueryInput(action="category_counts"))

    print(categories_result.model_dump())

    print()
    print("=" * 60)
    print("GENERATE REPORT")
    print("=" * 60)

    report = generate_report(
        GenerateReportInput(
            title="Documents retrouvés",
            items=[
                (f"{result.document_title} - page {result.page_number}")
                for result in (search_result.results)
            ],
        )
    )

    print(report.report)


if __name__ == "__main__":
    main()
