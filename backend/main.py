from fastapi import FastAPI
# uvicorn backend.main:app --reload

# Routers
from backend.api.query import router as query_router
from backend.api.documents import router as documents_router
from backend.api.analytics import router as analytics_router
from backend.api.chat import router as chat_router

# Shared engine
from backend.rag_engine.engine_singleton import engine

app = FastAPI(title="Chatter AI Backend")

# 🔥 Load cached documents ONCE at startup
engine.load_existing_documents()

# Register routers
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(analytics_router)
app.include_router(chat_router)