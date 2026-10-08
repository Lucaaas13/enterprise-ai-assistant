from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.document import DocumentCreate, DocumentResponse
from app.services.document_service import create_document, get_documents
from app.services.pdf_service import extract_pages_from_pdf
from app.services.chunk_service import (
    create_chunks_for_document,
    get_chunks_by_document,
)
from app.schemas.chunk import ChunkResponse

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

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=201,
)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    file_content = await file.read()

    try:
        pages = extract_pages_from_pdf(file_content)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read PDF file",
        )

    if not pages:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from PDF",
        )

    full_text = "\n\n".join(
        text for _, text in pages
    )

    document = DocumentCreate(
        title=file.filename or "Untitled PDF",
        content=full_text,
    )

    saved_document = create_document(
        db,
        document,
    )

    create_chunks_for_document(
        db,
        saved_document.id,
        pages,
    )

    return saved_document
    
@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
)
def list_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
):
    return get_chunks_by_document(
        db,
        document_id,
    )