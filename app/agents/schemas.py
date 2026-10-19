from typing import Literal

from pydantic import BaseModel, Field


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class SearchDocumentResult(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    page_number: int
    score: float
    content: str


class SearchDocumentsOutput(BaseModel):
    results: list[SearchDocumentResult]


class DatabaseQueryInput(BaseModel):
    action: Literal[
        "document_count",
        "category_counts",
        "recent_documents",
    ]

    limit: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class DatabaseQueryOutput(BaseModel):
    action: str
    data: dict | list


class GenerateReportInput(BaseModel):
    title: str = Field(min_length=1)
    items: list[str]


class GenerateReportOutput(BaseModel):
    report: str


class AgentDecision(BaseModel):
    tool: Literal[
        "search_documents",
        "query_database",
        "generate_report",
    ]

    database_action: (
        Literal[
            "document_count",
            "category_counts",
            "recent_documents",
        ]
        | None
    ) = None
