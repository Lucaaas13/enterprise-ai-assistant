from app.schemas.document import DocumentCreate, DocumentResponse


documents: list[DocumentResponse] = []
next_id = 1


def create_document(document: DocumentCreate) -> DocumentResponse:
    global next_id

    new_document = DocumentResponse(
        id=next_id,
        title=document.title,
        content=document.content,
    )

    documents.append(new_document)
    next_id += 1

    return new_document


def get_documents() -> list[DocumentResponse]:
    return documents