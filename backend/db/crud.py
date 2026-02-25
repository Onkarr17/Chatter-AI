from sqlalchemy.orm import Session

from backend.db.models import Document, Query, TokenUsage


# -------------------------------------------------
# Document CRUD
# -------------------------------------------------

from sqlalchemy.orm import Session
from backend.db.models import Document


def create_document(
    db: Session,
    document_id: str,
    filename: str,
    total_pages: int,
    total_chunks: int,
):
    """
    Insert document metadata.

    Correct long-term behavior:
    - If document exists AND is_deleted = true → restore it
    - If document exists AND active → do nothing
    - Else → create new document
    """

    existing = (
        db.query(Document)
        .filter(Document.document_id == document_id)
        .first()
    )

    # 🟢 Case 1: Document existed but was deleted → restore it
    if existing and existing.is_deleted:
        existing.is_deleted = False
        existing.filename = filename
        existing.total_pages = total_pages
        existing.total_chunks = total_chunks

        db.commit()
        db.refresh(existing)
        return existing

    # 🟢 Case 2: Document exists and already active
    if existing:
        return existing

    # 🟢 Case 3: Brand-new document
    doc = Document(
        document_id=document_id,
        filename=filename,
        total_pages=total_pages,
        total_chunks=total_chunks,
        is_deleted=False,
    )

    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


# -------------------------------------------------
# Query analytics CRUD
# -------------------------------------------------

def create_query(
    db: Session,
    document_id: str,
    question: str,
    answer: str,
    best_similarity: float,
):
    """
    Insert analytics for a single user query.
    One row per question.
    """

    query = Query(
        document_id=document_id,
        question=question,
        answer=answer,
        best_similarity=best_similarity,
    )

    db.add(query)
    db.commit()
    db.refresh(query)
    return query


# -------------------------------------------------
# Token usage analytics CRUD
# -------------------------------------------------

def create_token_usage(
    db: Session,
    query_id: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
):
    """
    Store token usage for one query.
    Linked via query_id.
    """

    usage = TokenUsage(
        query_id=query_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    db.add(usage)
    db.commit()
    db.refresh(usage)
    return usage

# What this enables(down the line)
# Fetch all uploaded PDFs (excluding deleted ones)
# Used later by UI PDF-list screen

from sqlalchemy.orm import Session
from backend.db.models import Document


def list_active_documents(db: Session):
    """
    Returns all PDFs that are NOT deleted.
    Used later by UI to show uploaded PDFs.
    """
    return (
        db.query(Document)
        .filter(Document.is_deleted == False)
        .order_by(Document.created_at.desc())
        .all()
    )

# soft delete function for documents
def soft_delete_document(db: Session, document_id: str):
    """
    Marks a document as deleted instead of removing it.
    This is what ❌ button will call.
    """
    doc = (
        db.query(Document)
        .filter(Document.document_id == document_id)
        .first()
    )

    if not doc:
        return None

    doc.is_deleted = True
    db.commit()
    return doc


# DB check for active document before allowing RAG operations
def is_document_active(db, document_id: str) -> bool:
    """
    Returns True if document exists and is not deleted.
    Used by RAG to block deleted PDFs.
    """
    doc = (
        db.query(Document)
        .filter(
            Document.document_id == document_id,
            Document.is_deleted == False
        )
        .first()
    )
    return doc is not None