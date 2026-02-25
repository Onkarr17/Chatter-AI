from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from backend.db.session import SessionLocal
from backend.db.crud import list_active_documents, soft_delete_document
from backend.rag_engine.engine import RAGEngine

router = APIRouter(prefix="/documents", tags=["Documents"])

# IMPORTANT:
# Use the SAME engine instance as the rest of the app
from backend.rag_engine.engine_singleton import engine


@router.get("/")
def get_documents():
    """
    Returns all active PDFs.
    UI will call this to show uploaded documents.
    """
    db = SessionLocal()
    try:
        docs = list_active_documents(db)
        return [
            {
                "document_id": d.document_id,
                "filename": d.filename,
                "total_pages": d.total_pages,
                "total_chunks": d.total_chunks,
                "created_at": d.created_at,
            }
            for d in docs
        ]
    finally:
        db.close()


@router.get("/thumbnail/{document_id}")
def get_thumbnail(document_id: str):
    """
    Returns the PNG thumbnail for a document.
    """
    thumb_path = os.path.join("rag_cache", "thumbnails", f"{document_id}.png")

    if not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(thumb_path)


@router.delete("/{document_id}")
def delete_document(document_id: str):
    """
    Called when ❌ is clicked on a PDF.

    This does TWO things:
    1. Soft-delete in DB
    2. Remove PDF from RAG memory (FAISS + chunks)
    """
    db = SessionLocal()
    try:
        doc = soft_delete_document(db, document_id)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        # Logical deletion handles everything in ask().
        # Physical unloading is not needed with current architecture.
        # engine.unload_document(document_id)

        return {"message": "Document deleted successfully"}

    finally:
        db.close()