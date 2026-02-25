from fastapi import APIRouter, Form, HTTPException
from backend.rag_engine.engine_singleton import engine

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/ask")
def ask_question(
    question: str = Form(...),
):
    """
    Ask a question across all ACTIVE documents.
    Works after refresh. No file required.
    """

    try:
        result = engine.ask(question=question)
        return {
            "answer": result["answer"],
            "best_similarity": result["best_similarity"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))