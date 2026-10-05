from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.database.database import Base, engine
from app.models.document import Document


Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(documents_router)