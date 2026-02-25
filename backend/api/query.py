from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import shutil
import os
import uuid

from backend.db.session import SessionLocal
from backend.db.crud import (
    create_document,
    create_query,
    create_token_usage,
)
from backend.rag_engine.engine import RAGEngine

router = APIRouter()
# engine = RAGEngine()
from backend.rag_engine.engine_singleton import engine

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/query")
def query_pdf(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """
    Uploads a PDF (if new) and asks a question
    across ALL ACTIVE PDFs.
    """

    db = SessionLocal()

    try:
        document_id = None
        
        # -----------------------------
        # 1. Save uploaded PDF (if provided)
        # -----------------------------
        if file and file.filename:
            temp_filename = f"{uuid.uuid4()}_{file.filename}"
            pdf_path = os.path.join(UPLOAD_DIR, temp_filename)

            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # -----------------------------
            # 2. Prepare document
            # -----------------------------
            doc_info = engine.prepare_document(pdf_path)
            document_id = doc_info["document_id"]

            create_document(
                db=db,
                document_id=document_id,
                filename=file.filename,
                total_pages=doc_info["total_pages"],
                total_chunks=doc_info["total_chunks"],
            )

        # -----------------------------
        # 3. MULTI-PDF QUESTION 🔥
        # -----------------------------
        result = engine.ask(question=question)

        raw_answer = result["answer"]

        # Normalize answer
        if isinstance(raw_answer, dict):
            answer_text = raw_answer.get("answer", "")
            usage = raw_answer.get("usage")
        else:
            answer_text = raw_answer
            usage = None

        best_similarity = result["best_similarity"]

        # -----------------------------
        # 4. Save query analytics
        # -----------------------------
        query_row = create_query(
            db=db,
            document_id=document_id,
            question=question,
            answer=answer_text,
            best_similarity=best_similarity,
        )

        if usage:
            create_token_usage(
                db=db,
                query_id=query_row.id,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )

        return {
            "answer": answer_text,
            "best_similarity": best_similarity,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()