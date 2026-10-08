from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.database.database import Base, engine
from app.models.document import Document
from app.models.chunk import Chunk
from app.api.search import router as search_router
from app.api.rag import router as rag_router

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(documents_router)
app.include_router(search_router)
app.include_router(rag_router)