import os
import hashlib
import numpy as np
import faiss


# -------------------------------------------------
# Cache configuration (kept minimal & explicit)
# -------------------------------------------------

CACHE_DIR = "rag_cache"
CACHE_VERSION = "v1_cosine_minilm"


# -------------------------------------------------
# 1. Document hash (your step6_get_doc_hash_from_path)
# -------------------------------------------------

def get_document_hash(pdf_path: str) -> str:
    """
    Create a stable hash for a PDF file.

    Why:
    - Same PDF → same cache
    - Different PDF → different cache
    """
    with open(pdf_path, "rb") as f:
        data = f.read()

    return hashlib.sha256(data).hexdigest()[:16]


# -------------------------------------------------
# 2. Cache path builder (your step6_cache_paths)
# -------------------------------------------------

def get_cache_paths(document_hash: str):
    """
    Build cache file paths for a document.

    Versioning prevents silent bugs when:
    - embedding model changes
    - chunk size changes
    - similarity strategy changes
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    index_path = os.path.join(
        CACHE_DIR, f"{document_hash}_{CACHE_VERSION}.faiss"
    )
    meta_path = os.path.join(
        CACHE_DIR, f"{document_hash}_{CACHE_VERSION}.npz"
    )

    return index_path, meta_path


# -------------------------------------------------
# 3. Save cache (your step6_save_cache)
# -------------------------------------------------

def save_cache(index, chunks, index_path: str, meta_path: str):
    """
    Save FAISS index and chunk metadata to disk.

    This makes subsequent runs instant.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, index_path)

    # Save chunk metadata
    np.savez_compressed(
        meta_path,
        ids=np.array([c["id"] for c in chunks], dtype=object),
        pages=np.array([c["page"] for c in chunks], dtype=np.int32),
        texts=np.array([c["text"] for c in chunks], dtype=object),
    )

    print("[OK] Cache saved")
    print("Index path:", index_path)
    print("Meta path :", meta_path)


# -------------------------------------------------
# 4. Load cache (your step6_load_cache)
# -------------------------------------------------

def load_cache(index_path: str, meta_path: str):
    """
    Load FAISS index and chunk metadata from disk.

    Performs safety checks to detect:
    - wrong index type
    - corrupted metadata
    """

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load metadata
    meta = np.load(meta_path, allow_pickle=True)

    index_type = type(index).__name__

    # Safety check: cosine similarity requires IP index
    if "L2" in index_type:
        print(
            "[WARNING] Cache was built using L2 distance. "
            "Cosine similarity requires Inner Product."
        )

    try:
        ids = meta["ids"].tolist()
        pages = meta["pages"].tolist()
        texts = meta["texts"].tolist()
    except Exception as e:
        raise RuntimeError(
            f"Cache metadata corrupted or invalid: {e}"
        )

    if not (len(ids) == len(pages) == len(texts)):
        raise RuntimeError(
            "Cache metadata length mismatch. "
            "Cache rebuild required."
        )

    chunks = []
    for cid, page, text in zip(ids, pages, texts):
        chunks.append(
            {
                "id": cid,
                "page": int(page),
                "text": text,
            }
        )

    print("[OK] Cache loaded")
    print("Vectors:", index.ntotal)
    print("Chunks :", len(chunks))
    print("Index type:", index_type)

    return index, chunks



def delete_cache(document_hash: str):
    index_path, meta_path = get_cache_paths(document_hash)

    if os.path.exists(index_path):
        os.remove(index_path)

    if os.path.exists(meta_path):
        os.remove(meta_path)