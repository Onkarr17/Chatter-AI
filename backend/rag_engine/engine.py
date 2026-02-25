import os

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("[WARNING] PyMuPDF not found. PDF thumbnails will not be generated.")

from backend.db.session import SessionLocal
from backend.db.crud import is_document_active

from backend.rag_engine.ingestion import read_pdf, split_into_chunks
from backend.rag_engine.embeddings import load_embedding_model, create_embeddings
from backend.rag_engine.retrieval import build_faiss_index, retrieve_top_chunks
from backend.rag_engine.cache import (
    get_document_hash,
    get_cache_paths,
    save_cache,
    load_cache,
)
from backend.rag_engine.llm import answer_from_pdf, answer_general


class RAGEngine:
    """
    Central RAG Engine.

    DEFAULT BEHAVIOR:
    - Multi-PDF intelligence
    - Searches across all ACTIVE PDFs
    - Survives page refresh & server restart
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        # Load embedding model ONCE
        self.embedder = load_embedding_model(embedding_model_name)

        # document_id -> { index, chunks }
        self._loaded_documents = {}

    # -------------------------------------------------
    # PDF INGESTION (UPLOAD TIME)
    # -------------------------------------------------
    def prepare_document(self, pdf_path: str):
        """
        Ingests a PDF:
        - Reads
        - Chunks
        - Embeds
        - Builds FAISS index
        - Saves cache to disk
        """

        pages = read_pdf(pdf_path)
        chunks = split_into_chunks(pages)
        embeddings = create_embeddings(chunks, self.embedder)
        index = build_faiss_index(embeddings)

        document_id = get_document_hash(pdf_path)

        # Attach document_id to each chunk (future-safe)
        for chunk in chunks:
            chunk["document_id"] = document_id

        # Persist cache to disk
        index_path, meta_path = get_cache_paths(document_id)
        save_cache(index, chunks, index_path, meta_path)

        # Keep in memory for fast access
        self._loaded_documents[document_id] = {
            "index": index,
            "chunks": chunks,
        }

        # Generate thumbnail for UI preview
        self._generate_thumbnail(pdf_path, document_id)

        return {
            "document_id": document_id,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
        }

    def _generate_thumbnail(self, pdf_path: str, document_id: str):
        """
        Generates a PNG thumbnail of the first page of the PDF.
        Stored in rag_cache/thumbnails/[document_id].png
        """
        try:
            thumb_dir = os.path.join("rag_cache", "thumbnails")
            os.makedirs(thumb_dir, exist_ok=True)
            thumb_path = os.path.join(thumb_dir, f"{document_id}.png")

            # Skip if already exists
            if fitz is None:
                print("[WARNING] Skipping thumbnail generation: PyMuPDF not installed.")
                return None

            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                page = doc[0]
                # High resolution (2x zoom)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                pix.save(thumb_path)
                print(f"[OK] Thumbnail generated: {thumb_path}")
            doc.close()
            return thumb_path
        except Exception as e:
            print(f"[ERROR] Thumbnail generation failed: {e}")
            return None

    # -------------------------------------------------
    # LOAD AFTER REFRESH / SERVER RESTART  ✅
    # -------------------------------------------------
    def load_existing_documents(self):
        """
        Loads all ACTIVE documents from disk cache into memory.
        Call ONCE at server startup.
        """

        from backend.db.models import Document

        db = SessionLocal()
        try:
            docs = (
                db.query(Document)
                .filter(Document.is_deleted == False)
                .all()
            )

            for doc in docs:
                index_path, meta_path = get_cache_paths(doc.document_id)

                if not os.path.exists(index_path):
                    continue  # Cache missing → skip safely

                index, chunks = load_cache(index_path, meta_path)

                # Backward compatibility
                for chunk in chunks:
                    chunk["document_id"] = doc.document_id

                self._loaded_documents[doc.document_id] = {
                    "index": index,
                    "chunks": chunks,
                }

                # Ensure thumbnail exists (lazy migration)
                if hasattr(doc, "file_path") and os.path.exists(doc.file_path):
                    self._generate_thumbnail(doc.file_path, doc.document_id)
                else:
                    # Search in uploads dir if absolute path not stored
                    # Usually files are in uploaded_pdfs/
                    potential_path = os.path.join("uploaded_pdfs", doc.filename)
                    if os.path.exists(potential_path):
                        self._generate_thumbnail(potential_path, doc.document_id)

            print(f"[OK] Loaded {len(self._loaded_documents)} documents into memory")

        finally:
            db.close()

    # -------------------------------------------------
    # MULTI-PDF QUESTION (CORE LOGIC)
    # -------------------------------------------------
    def ask(
        self,
        question: str,
        top_k: int = 8,
        conversation_history=None,
        allow_general_fallback: bool = False,
    ):
        """
        Ask a question across ALL ACTIVE documents.
        """

        if not self._loaded_documents:
            raise ValueError("No documents uploaded.")

        db = SessionLocal()
        all_retrieved = []

        try:
            for doc_id, doc in self._loaded_documents.items():
                # Skip soft-deleted PDFs
                if not is_document_active(db, doc_id):
                    continue

                retrieved, _ = retrieve_top_chunks(
                    query=question,
                    embedder=self.embedder,
                    index=doc["index"],
                    chunks=doc["chunks"],
                    top_k=top_k,
                )

                all_retrieved.extend(retrieved)

        finally:
            db.close()

        if not all_retrieved:
            return {
                "answer": "Not found in any uploaded document.",
                "best_similarity": 0.0,
            }

        # Global re-ranking
        all_retrieved.sort(key=lambda x: x[1], reverse=True)
        top_retrieved = all_retrieved[:top_k]

        best_similarity = top_retrieved[0][1]

        answer = answer_from_pdf(
            question=question,
            retrieved=top_retrieved,
            conversation_history=conversation_history,
        )

        if allow_general_fallback and "Not found in the PDF" in answer:
            answer = answer_general(
                question=question,
                conversation_history=conversation_history,
            )

        return {
            "answer": answer,
            "best_similarity": best_similarity,
        }