from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.rag import (
    RagRequest,
    RagResponse,
)
from app.services.rag_service import answer_question


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


@router.post(
    "/ask",
    response_model=RagResponse,
)
def ask_question(
    request: RagRequest,
    db: Session = Depends(get_db),
):
    return answer_question(
        db=db,
        question=request.question,
        top_k=request.top_k,
    )