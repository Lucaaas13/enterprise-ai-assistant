from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentCreate(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=200,
    )
    content: str = Field(
        min_length=1,
    )

    @field_validator("title")
    @classmethod
    def validate_title(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Title cannot be empty")

        return value


class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    category: str | None = None
    category_confidence: float | None = None

    model_config = ConfigDict(from_attributes=True)
