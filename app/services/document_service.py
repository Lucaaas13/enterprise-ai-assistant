from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.schemas.document import DocumentCreate


def create_document(
    db: Session,
    document: DocumentCreate,
    category: str | None = None,
    category_confidence: float | None = None,
) -> Document:
    new_document = Document(
        title=document.title,
        content=document.content,
        category=category,
        category_confidence=category_confidence,
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    return new_document


def get_documents(
    db: Session,
) -> list[Document]:
    statement = select(Document).order_by(Document.id)

    return list(db.scalars(statement).all())
