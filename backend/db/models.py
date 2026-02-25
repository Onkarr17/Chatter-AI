from sqlalchemy import Boolean, Column, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Document(Base):
    """
    Stores metadata about each uploaded PDF.
    One row per document.
    """
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    total_pages = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    is_deleted = Column(Boolean, default=False)      # Soft delete flag
    created_at = Column(DateTime, default=datetime.utcnow)


class Query(Base):
    """
    Stores analytics for each user question.
    One row per query.
    """
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    best_similarity = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class TokenUsage(Base):
    """
    Stores token usage per query (analytics & cost tracking).
    """
    __tablename__ = "token_usage"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)    