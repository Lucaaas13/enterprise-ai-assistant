from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.document import DocumentCreate, DocumentResponse
from app.services.document_service import create_document, get_documents


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=201,
)
def add_document(
    document: DocumentCreate,
    db: Session = Depends(get_db),
):
    return create_document(db, document)


@router.get(
    "",
    response_model=list[DocumentResponse],
)
def list_documents(
    db: Session = Depends(get_db),
):
    return get_documents(db)