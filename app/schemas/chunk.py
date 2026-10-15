from pydantic import BaseModel, ConfigDict


class ChunkResponse(BaseModel):
    id: int
    document_id: int
    page_number: int
    chunk_index: int
    content: str

    model_config = ConfigDict(from_attributes=True)
