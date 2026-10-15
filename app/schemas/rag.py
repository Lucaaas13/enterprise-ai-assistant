from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


class RagSource(BaseModel):
    chunk_id: int
    document_id: int
    document_title: str
    page_number: int
    score: float


class RagResponse(BaseModel):
    answer: str
    sources: list[RagSource]
