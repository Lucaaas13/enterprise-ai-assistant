from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.search import SearchRequest, SearchResult
from app.services.search_service import semantic_search


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


@router.post(
    "",
    response_model=list[SearchResult],
)
def search_documents(
    request: SearchRequest,
    db: Session = Depends(get_db),
):
    return semantic_search(
        db=db,
        query=request.query,
        top_k=request.top_k,
    )