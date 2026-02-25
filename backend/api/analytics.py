from fastapi import APIRouter
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.db.session import SessionLocal
from backend.db.models import Document, Query

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview")
def analytics_overview():
    db: Session = SessionLocal()

    try:
        # ✅ ONLY ACTIVE DOCUMENTS
        total_documents = (
            db.query(func.count(Document.document_id))
            .filter(Document.is_deleted == False)
            .scalar()
        )

        # Queries = historical (do NOT filter)
        total_queries = db.query(func.count(Query.id)).scalar()

        avg_similarity = db.query(func.avg(Query.best_similarity)).scalar()

        # ✅ Most queried ACTIVE document only
        most_queried = (
            db.query(
                Document.filename,
                func.count(Query.id).label("q_count"),
            )
            .join(Query, Query.document_id == Document.document_id)
            .filter(Document.is_deleted == False)
            .group_by(Document.filename)
            .order_by(func.count(Query.id).desc())
            .first()
        )

        return {
            "total_documents": total_documents or 0,
            "total_queries": total_queries or 0,
            "average_similarity": round(avg_similarity, 3) if avg_similarity else None,
            "most_queried_document": (
                {
                    "filename": most_queried.filename,
                    "query_count": most_queried.q_count,
                }
                if most_queried
                else None
            ),
        }

    finally:
        db.close()