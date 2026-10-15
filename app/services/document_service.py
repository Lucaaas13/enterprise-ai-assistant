from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.schemas.document import DocumentCreate


def create_document(
    db: Session,
    document: DocumentCreate,
) -> Document:
    new_document = Document(
        title=document.title,
        content=document.content,
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    return new_document


def get_documents(db: Session) -> list[Document]:
    statement = select(Document).order_by(Document.id)

    return list(db.scalars(statement).all())
