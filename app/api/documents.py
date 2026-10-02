from fastapi import APIRouter

from app.schemas.document import DocumentCreate, DocumentResponse
from app.services.document_service import create_document, get_documents


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("", response_model=DocumentResponse, status_code=201)
def add_document(document: DocumentCreate):
    return create_document(document)


@router.get("", response_model=list[DocumentResponse])
def list_documents():
    return get_documents()