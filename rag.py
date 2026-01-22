from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

import os
import sys
import hashlib
from dotenv import load_dotenv
from groq import Groq


def safe_print(text, max_length=None):
    """
    Safely print text that may contain Unicode characters.
    Handles Windows console encoding issues.
    """
    try:
        if max_length:
            text = text[:max_length]
        # Try to print normally first
        print(text)
    except UnicodeEncodeError:
        # If that fails, replace problematic characters
        try:
            safe_text = text.encode('ascii', errors='replace').decode('ascii')
            print(safe_text)
        except Exception:
            print("[Text contains non-printable characters]")

# Default PDF path (only used when running rag.py standalone for testing)
# When imported by Streamlit, PDF path is passed dynamically via uploads
# Set to None for deployment - users will upload their own PDFs
PDF_PATH = None

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

CACHE_DIR = "rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache versioning - includes key parameters that affect cache validity
# Change this when you modify: embedding model, chunk size, FAISS index type, etc.
CACHE_VERSION = "v1_cosine_minilm"

# Cosine similarity threshold (range: -1 to 1, higher = more similar)
# With cosine similarity, we want scores above this threshold
# Lower threshold = more lenient (allows more matches)
# Note: Cosine similarity range is [-1, 1] where:
#   - 1 = identical/similar direction
#   - 0 = orthogonal (unrelated)
#   - -1 = opposite direction (dissimilar)
SIMILARITY_THRESHOLD = 0.3  # Cosine similarity: 0.3+ is reasonable match


# ---------------- STEP 1 ----------------
def step1_read_pdf(pdf_path=None):
    """
    Read PDF file. 
    
    Args:
        pdf_path: Path to PDF file. If None, uses default PDF_PATH (only for standalone use).
                  When called from Streamlit or other apps, always provide pdf_path.
    
    Returns:
        List of page documents
    """
    if pdf_path is None:
        pdf_path = PDF_PATH
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print("[OK] Total pages:", len(pages))
    if pages:
        print("\n--- Page 1 Preview (first 400 chars) ---\n")
        safe_print(pages[0].page_content, max_length=400)

    return pages


# ---------------- STEP 2 ----------------
def step2_split_into_chunks(pages, chunk_size=800, chunk_overlap=150):
    """
    Converts pages into smaller overlapping chunks for RAG.
    Output format of each chunk:
      { "id": "...", "page": page_number, "text": chunk_text }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for page_index, page_doc in enumerate(pages):
        page_no = page_index + 1
        page_text = page_doc.page_content or ""

        # split into chunks
        split_texts = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(split_texts):
            chunk_id = f"p{page_no}_c{chunk_index}"
            chunks.append({
                "id": chunk_id,
                "page": page_no,
                "text": chunk_text
            })

    print("\n[OK] Total chunks created:", len(chunks))

    # show sample chunks (safe for Unicode/Windows console)
    if chunks:
        print("\n--- Sample Chunk 1 ---\n")
        safe_print(chunks[0]["text"], max_length=250)

        if len(chunks) > 1:
            print("\n--- Sample Chunk 2 ---\n")
            safe_print(chunks[1]["text"], max_length=250)

    return chunks


# ---------------- STEP 3 ----------------
def step3_create_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """
    Create embeddings for each chunk text.
    Normalizes embeddings for cosine similarity (L2 normalization).
    embeddings shape = (num_chunks, embedding_dimension)
    """
    print("\n[LOADING] Loading embedding model:", model_name)
    embedder = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]

    print("[LOADING] Creating embeddings for chunks...")
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Normalize embeddings for cosine similarity
    # After normalization: cosine_similarity = dot_product
    # normalize_L2 modifies in-place and expects 2D array (n_samples, n_features)
    faiss.normalize_L2(embeddings)
    
    # Verify normalization (optional check - vectors should have L2 norm ≈ 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    avg_norm = np.mean(norms)
    if abs(avg_norm - 1.0) > 0.01:
        print(f"[WARNING] Warning: Average L2 norm is {avg_norm:.4f} (expected ~1.0)")
    
    print("[OK] Embeddings created and normalized!")
    print("Embeddings shape:", embeddings.shape)
    print(f"Average L2 norm: {avg_norm:.4f} (should be ~1.0)")

    return embedder, embeddings


# ---------------- STEP 4 ----------------
def step4_build_faiss_index(embeddings: np.ndarray):
    """
    Build FAISS index using Inner Product (IP) for cosine similarity.
    Embeddings must be L2-normalized first (done in step3).
    """
    dim = embeddings.shape[1]
    # IndexFlatIP uses inner product - with normalized vectors, this = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("\n[OK] FAISS index built (cosine similarity)!")
    print("Total vectors stored:", index.ntotal)
    return index


def step4_retrieve_top_chunks(question: str, embedder, index, chunks, top_k=8, force_rebuild_callback=None):
    """
    1) Embed user question and normalize
    2) Search in FAISS using cosine similarity (Inner Product)
    3) Return top-k relevant chunks + best similarity score
    
    Note: Cosine similarity range is [-1, 1] where:
    - 1 = identical/similar direction
    - 0 = orthogonal (unrelated)
    - -1 = opposite direction (dissimilar)
    """
    q_emb = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    
    # Normalize query embedding for cosine similarity
    # normalize_L2 expects 2D array, ensure it's reshaped correctly
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, top_k)
    raw_scores = D[0].tolist()
    indices = I[0].tolist()

    # Use raw scores - don't clamp! Cosine similarity can be negative.
    # If scores > 1.01, that indicates a normalization/cache bug - fix it, don't hide it
    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score > 1.01:
        print(f"\n[ERROR] ERROR: Similarity scores exceeded 1.0 (max: {max_score:.4f})")
        print("   This indicates the cache was built with unnormalized embeddings.")
        print("   This is a data integrity issue - forcing cache rebuild...")
        if force_rebuild_callback:
            force_rebuild_callback()
        # Return empty to trigger rebuild
        return [], 0.0

    similarities = raw_scores  # Use raw scores - preserve negative values and full range

    retrieved = []
    for idx, sim in zip(indices, similarities):
        if 0 <= idx < len(chunks):
            # Store raw similarity score (can be negative for dissimilar content)
            retrieved.append((chunks[idx], sim))

    best_similarity = similarities[0] if similarities else 0.0
    return retrieved, best_similarity


# ---------------- STEP 5 ----------------
def step5_groq_chat(prompt: str, max_tokens=350, conversation_history=None) -> str:
    """
    Calls Groq API to generate final answer.
    Needs GROQ_API_KEY in .env file.
    
    Args:
        prompt: The current question/prompt
        max_tokens: Maximum tokens for response
        conversation_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}, ...]
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "[ERROR] GROQ_API_KEY missing. Add it in .env file."

    client = Groq(api_key=api_key)
    
    # Build messages list with conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context and conversation history. Remember information shared in previous messages. If the context contains relevant information that can help answer the question (even if not exact), use it to provide a helpful answer. Only say 'Not found in the PDF' if the context truly has no relevant information at all. Always format answers in a structured, point-wise manner with clear formatting, bold headers for new points, numbered or bulleted lists, and proper spacing between points."}
    ]
    
    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current prompt
    messages.append({"role": "user", "content": prompt})
    
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content


def step5_answer_from_retrieved(question: str, retrieved, top_k=8, conversation_history=None, max_tokens=400) -> str:
    """
    Builds context from retrieved chunks and asks Groq LLM.
    Also attaches sources (page numbers).
    Uses more chunks (top_k=8) for better context understanding.
    
    Args:
        question: Current question
        retrieved: Retrieved chunks from FAISS
        top_k: Number of chunks to use
        conversation_history: Previous conversation messages for context
        max_tokens: Maximum tokens for response (controls answer length)
    """
    if not retrieved:
        return "Not found in the PDF."

    context_parts = []
    pages_used = set()

    # Use all retrieved chunks (up to top_k) for better context
    for c, dist in retrieved[:top_k]:
        pages_used.add(c["page"])
        context_parts.append(f"[Page {c['page']} | {c['id']}]\n{c['text']}")

    context = "\n\n".join(context_parts)
    pages_used = sorted(pages_used)

    # Adjust prompt based on max_tokens to control answer length and structure
    if max_tokens <= 200:
        length_instruction = "Provide a BRIEF, CONCISE answer in 2-3 bullet points maximum. Be direct and to the point."
        format_instruction = "Format: Use bullet points (• or -) with relevant emojis. Each point should be clear and concise. Use emojis to make it engaging (e.g., ✅, 📌, 💡)."
    elif max_tokens <= 400:
        length_instruction = "Provide a balanced answer with 3-5 key points. Include brief explanations for each point."
        format_instruction = "Format: Use numbered points (1., 2., 3.) or bullet points with emojis. Start each new point on a new line with clear spacing. Use **bold** for point headers and relevant emojis (✅, 📌, 💡, 🔑, ⚠️) to enhance readability."
    else:
        length_instruction = "Provide a DETAILED, COMPREHENSIVE answer with multiple well-structured points. Include examples, explanations, and all relevant details."
        format_instruction = "Format: Use numbered points (1., 2., 3.) with clear headers and emojis. Start each major point with a **bold header** and relevant emoji. Use sub-points with dashes (-) or bullets (•) for details. Add spacing between points and use emojis strategically (✅, 📌, 💡, 🔑, ⚠️, 📝, 🎯, etc.) to make it engaging and easy to scan."
    
    prompt = f"""
Answer the question based on the context provided below. 
The context contains relevant information from the PDF document.

If the context contains information that relates to the question (even if not an exact match), 
use that information to provide a helpful and accurate answer. 
Only say "Not found in the PDF" if the context truly has no relevant information whatsoever.

{length_instruction}

{format_instruction}

IMPORTANT FORMATTING RULES:
- Structure your answer in clear, numbered or bulleted points
- Each new point should start on a new line
- Use **bold** to highlight point headers or key terms
- Add blank lines between major points for clarity
- Use consistent formatting throughout
- Use relevant emojis strategically to make the answer more engaging and easier to read:
  • ✅ for benefits/advantages/positive points
  • ⚠️ for warnings/important notes
  • 📌 for key points or highlights
  • 💡 for tips/insights
  • 🔑 for key concepts
  • 📝 for examples
  • 🎯 for main points
  • ❌ for disadvantages/negatives
  • ⚡ for performance/quick facts

Context:
{context}

Question:
{question}

Provide a clear, well-structured, point-wise answer based on the context.
"""

    answer = step5_groq_chat(prompt, max_tokens=max_tokens, conversation_history=conversation_history)

    if "Not found in the PDF" not in answer and pages_used:
        answer += "\n\nSources: " + ", ".join([f"Page {p}" for p in pages_used])

    return answer


def step5_general_knowledge_answer(question: str, conversation_history=None, max_tokens=250) -> str:
    """
    Uses Groq WITHOUT PDF context (general knowledge).
    
    Args:
        question: Current question
        conversation_history: Previous conversation messages for context
        max_tokens: Maximum tokens for response (controls answer length)
    """
    # Adjust prompt based on max_tokens to control answer length and structure
    if max_tokens <= 200:
        length_instruction = "Answer BRIEFLY in 2-3 bullet points maximum. Be concise and direct."
        format_instruction = "Format: Use bullet points (• or -) with relevant emojis. Each point should be clear and concise. Use emojis to make it engaging (e.g., ✅, 📌, 💡)."
    elif max_tokens <= 400:
        length_instruction = "Answer with 3-5 key points in a structured format."
        format_instruction = "Format: Use numbered points (1., 2., 3.) or bullet points with emojis. Start each new point on a new line with clear spacing. Use **bold** for point headers and relevant emojis (✅, 📌, 💡, 🔑, ⚠️) to enhance readability."
    else:
        length_instruction = "Provide a DETAILED, COMPREHENSIVE answer with multiple well-structured points, examples, and explanations."
        format_instruction = "Format: Use numbered points (1., 2., 3.) with clear headers and emojis. Start each major point with a **bold header** and relevant emoji. Use sub-points with dashes (-) or bullets (•) for details. Add spacing between points and use emojis strategically (✅, 📌, 💡, 🔑, ⚠️, 📝, 🎯, etc.) to make it engaging and easy to scan."
    
    prompt = f"""
{length_instruction}

{format_instruction}

IMPORTANT FORMATTING RULES:
- Structure your answer in clear, numbered or bulleted points
- Each new point should start on a new line
- Use **bold** to highlight point headers or key terms
- Add blank lines between major points for clarity
- Use consistent formatting throughout
- Use relevant emojis strategically to make the answer more engaging and easier to read:
  • ✅ for benefits/advantages/positive points
  • ⚠️ for warnings/important notes
  • 📌 for key points or highlights
  • 💡 for tips/insights
  • 🔑 for key concepts
  • 📝 for examples
  • 🎯 for main points
  • ❌ for disadvantages/negatives
  • ⚡ for performance/quick facts

Remember information from previous conversation if relevant.

Question:
{question}
"""
    return step5_groq_chat(prompt, max_tokens=max_tokens, conversation_history=conversation_history)


# ---------------- STEP 6 ----------------
def step6_get_doc_hash_from_path(pdf_path: str) -> str:
    """
    Create a unique hash for the PDF file (so cache is separate per PDF).
    """
    with open(pdf_path, "rb") as f:
        data = f.read()
    return hashlib.sha256(data).hexdigest()[:16]


def step6_cache_paths(doc_hash: str):
    """
    Creates file paths for cache with versioning.
    Versioning prevents silent bugs when changing embedding model, chunk size, etc.
    """
    idx_path = os.path.join(CACHE_DIR, f"{doc_hash}_{CACHE_VERSION}.faiss")
    meta_path = os.path.join(CACHE_DIR, f"{doc_hash}_{CACHE_VERSION}.npz")
    return idx_path, meta_path


def step6_save_cache(index, chunks, idx_path: str, meta_path: str):
    """
    Save FAISS index + chunks metadata so next run is instant.
    """
    faiss.write_index(index, idx_path)
    np.savez_compressed(
        meta_path,
        ids=np.array([c["id"] for c in chunks], dtype=object),
        pages=np.array([c["page"] for c in chunks], dtype=np.int32),
        texts=np.array([c["text"] for c in chunks], dtype=object)
    )
    print("\n[OK] Cache saved!")
    print("Saved index:", idx_path)
    print("Saved meta :", meta_path)


def step6_load_cache(idx_path: str, meta_path: str):
    """
    Load FAISS index + chunks metadata.
    Checks if index type is correct (IndexFlatIP for cosine similarity).
    """
    index = faiss.read_index(idx_path)
    meta = np.load(meta_path, allow_pickle=True)

    # Check if this is the old L2 index type (should be IP for cosine similarity)
    index_type = type(index).__name__
    if "L2" in index_type or "FlatL2" in index_type:
        print(f"\n[WARNING] WARNING: Cache contains old index type ({index_type})")
        print("   Old cache uses L2 distance, not cosine similarity.")
        print("   Please delete the cache folder to rebuild with cosine similarity.")
        print("   For now, continuing but results may be incorrect...")

    # Safely extract metadata with error handling
    try:
        ids = meta["ids"].tolist() if "ids" in meta else []
        pages = meta["pages"].tolist() if "pages" in meta else []
        texts = meta["texts"].tolist() if "texts" in meta else []
    except (KeyError, IndexError) as e:
        print(f"\n[ERROR] Error loading cache metadata: {e}")
        print("   Cache file may be corrupted. Rebuilding cache...")
        raise Exception(f"Cache metadata corrupted: {e}")

    # Check if arrays have matching lengths
    if not (len(ids) == len(pages) == len(texts)):
        print(f"\n[ERROR] Cache metadata mismatch: ids={len(ids)}, pages={len(pages)}, texts={len(texts)}")
        raise Exception("Cache metadata arrays have mismatched lengths. Please delete cache and rebuild.")

    chunks = []
    for cid, p, t in zip(ids, pages, texts):
        chunks.append({"id": cid, "page": int(p), "text": t})

    print("\n[OK] Cache loaded!")
    print("Vectors:", index.ntotal)
    print("Chunks :", len(chunks))
    print("Index type:", index_type)
    return index, chunks


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Example usage (only for local testing)
    # For deployment, users upload PDFs via Streamlit UI
    if PDF_PATH is None or not os.path.exists(PDF_PATH):
        print("[INFO] No default PDF path set or file not found.")
        print("[INFO] Run via Streamlit app (streamlit run streamlit_app.py) to upload PDFs.")
        sys.exit(0)
    
    # STEP 6 (Caching): check if cache exists
    doc_hash = step6_get_doc_hash_from_path(PDF_PATH)
    idx_path, meta_path = step6_cache_paths(doc_hash)

    # always need embedder for question embedding
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize variables (will be set in if/else blocks)
    index = None
    chunks = None

    if os.path.exists(idx_path) and os.path.exists(meta_path):
        # [OK] Load from cache
        print("[OK] Cache HIT. Loading from disk...")
        index, chunks = step6_load_cache(idx_path, meta_path)
        
        # Check if we need to rebuild (old cache with L2 index)
        index_type = type(index).__name__
        if "L2" in index_type or "FlatL2" in index_type:
            print("\n[REBUILD] Old cache detected. Rebuilding with cosine similarity...")
            # Delete old cache
            os.remove(idx_path)
            os.remove(meta_path)
            # Rebuildj
            pages = step1_read_pdf()
            chunks = step2_split_into_chunks(pages)
            embedder, embeddings = step3_create_embeddings(chunks)
            index = step4_build_faiss_index(embeddings)
            step6_save_cache(index, chunks, idx_path, meta_path)
    else:
        # [ERROR] Cache not found -> build everything
        print("[WARNING] Cache MISS. Building index first time...")

        # Step 1
        pages = step1_read_pdf()

        # Step 2
        chunks = step2_split_into_chunks(pages)

        # Step 3
        embedder, embeddings = step3_create_embeddings(chunks)

        # Step 4
        index = step4_build_faiss_index(embeddings)

        # [OK] save cache after building
        step6_save_cache(index, chunks, idx_path, meta_path)

    # Step 5 + Step 7 (permission-based general knowledge)
    # Store variables in a dict to allow modification from nested function
    # This avoids nonlocal issues - we can modify dict values from nested function
    state = {'index': index, 'chunks': chunks, 'embedder': embedder}
    
    def rebuild_cache():
        """Callback function to rebuild cache when normalization issues detected"""
        print("\n[REBUILD] Rebuilding cache with proper normalization...")
        # Delete old cache
        if os.path.exists(idx_path):
            os.remove(idx_path)
        if os.path.exists(meta_path):
            os.remove(meta_path)
        # Rebuild
        pages = step1_read_pdf()
        new_chunks = step2_split_into_chunks(pages)
        new_embedder, embeddings = step3_create_embeddings(new_chunks)
        new_index = step4_build_faiss_index(embeddings)
        step6_save_cache(new_index, new_chunks, idx_path, meta_path)
        # Update state
        state['index'] = new_index
        state['chunks'] = new_chunks
        state['embedder'] = new_embedder
        print("[OK] Cache rebuilt successfully!")
    
    while True:
        question = input("\nAsk question (or exit): ").strip()
        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        retrieved, best_similarity = step4_retrieve_top_chunks(
            question, state['embedder'], state['index'], state['chunks'], top_k=8, force_rebuild_callback=rebuild_cache
        )
        
        # If rebuild was triggered, reload state and skip this iteration
        if not retrieved and best_similarity == 0.0:
            # Reload from state after rebuild
            index = state['index']
            chunks = state['chunks']
            embedder = state['embedder']
            print("Please ask your question again after cache rebuild.")
            continue

        print("\n--- Top Retrieved Chunks (Cosine Similarity) ---")
        for rank, (c, sim) in enumerate(retrieved, start=1):
            print(f"\n#{rank} — Page {c['page']} — similarity: {sim:.4f}")
            safe_print(c["text"], max_length=220)

        # Always try to answer from PDF first - cosine similarity gives better matches
        # The LLM will decide if context is relevant
        print("\n--- Final Answer (Groq from PDF) ---")
        final_answer = step5_answer_from_retrieved(question, retrieved, top_k=8)
        print(final_answer)
        
        # Only offer general knowledge if LLM explicitly says not found
        if "Not found in the PDF" in final_answer or "not found" in final_answer.lower():
            print(f"\n[WARNING] Note: Best match similarity was {best_similarity:.4f} (threshold: {SIMILARITY_THRESHOLD})")
            choice = input("\nDo you want a general answer instead? (y/n): ").strip().lower()
            if choice in ("y", "yes"):
                print("\n--- General Answer (Groq) ---")
                print(step5_general_knowledge_answer(question))
