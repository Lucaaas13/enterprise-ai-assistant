from mcp.server.fastmcp import FastMCP

from app.agents.schemas import (
    DatabaseQueryInput,
    SearchDocumentsInput,
)
from app.agents.tools import (
    query_database,
    search_documents,
)

mcp = FastMCP("Enterprise AI Knowledge Assistant")


@mcp.tool()
def search_knowledge(
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Search the enterprise knowledge base.

    Returns relevant document chunks with
    document title, page and score.
    """
    result = search_documents(
        SearchDocumentsInput(
            query=query,
            top_k=top_k,
        )
    )

    return result.model_dump()


@mcp.tool()
def get_document_count() -> dict:
    """
    Return the number of documents
    stored in PostgreSQL.
    """
    result = query_database(DatabaseQueryInput(action="document_count"))

    return result.model_dump()


@mcp.tool()
def get_category_counts() -> dict:
    """
    Return the number of documents
    grouped by category.
    """
    result = query_database(DatabaseQueryInput(action="category_counts"))

    return result.model_dump()


@mcp.tool()
def get_recent_documents(
    limit: int = 5,
) -> dict:
    """
    Return the most recently added
    documents.
    """
    result = query_database(
        DatabaseQueryInput(
            action="recent_documents",
            limit=limit,
        )
    )

    return result.model_dump()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
