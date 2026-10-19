from sqlalchemy import func, select

from app.agents.schemas import (
    DatabaseQueryInput,
    DatabaseQueryOutput,
    GenerateReportInput,
    GenerateReportOutput,
    SearchDocumentResult,
    SearchDocumentsInput,
    SearchDocumentsOutput,
)
from app.database.database import SessionLocal
from app.models.document import Document
from app.services.search_service import hybrid_search


def search_documents(
    payload: SearchDocumentsInput,
) -> SearchDocumentsOutput:
    db = SessionLocal()

    try:
        results = hybrid_search(
            db=db,
            query=payload.query,
            top_k=payload.top_k,
            candidate_k=10,
        )

        return SearchDocumentsOutput(
            results=[
                SearchDocumentResult(
                    chunk_id=result["chunk_id"],
                    document_id=result["document_id"],
                    document_title=result["document_title"],
                    page_number=result["page_number"],
                    score=result["score"],
                    content=result["content"],
                )
                for result in results
            ]
        )

    finally:
        db.close()


def query_database(
    payload: DatabaseQueryInput,
) -> DatabaseQueryOutput:
    db = SessionLocal()

    try:
        if payload.action == "document_count":
            count = db.scalar(select(func.count(Document.id)))

            return DatabaseQueryOutput(
                action=payload.action,
                data={"count": int(count or 0)},
            )

        if payload.action == "category_counts":
            statement = (
                select(
                    Document.category,
                    func.count(Document.id),
                )
                .group_by(Document.category)
                .order_by(func.count(Document.id).desc())
            )

            rows = db.execute(statement).all()

            data = [
                {
                    "category": (category or "non_classee"),
                    "count": count,
                }
                for category, count in rows
            ]

            return DatabaseQueryOutput(
                action=payload.action,
                data=data,
            )

        if payload.action == "recent_documents":
            statement = select(Document).order_by(Document.id.desc()).limit(payload.limit)

            documents = list(db.scalars(statement).all())

            data = [
                {
                    "id": document.id,
                    "title": document.title,
                    "category": (document.category),
                    "category_confidence": (document.category_confidence),
                }
                for document in documents
            ]

            return DatabaseQueryOutput(
                action=payload.action,
                data=data,
            )

        raise ValueError("Unsupported database action")

    finally:
        db.close()


def generate_report(
    payload: GenerateReportInput,
) -> GenerateReportOutput:
    lines = [
        f"# {payload.title}",
        "",
    ]

    for index, item in enumerate(
        payload.items,
        start=1,
    ):
        lines.append(f"{index}. {item}")

    return GenerateReportOutput(report="\n".join(lines))
