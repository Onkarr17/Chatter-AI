import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and return the embedding model.

    Why this is a separate function:
    - Model should be loaded ONCE (expensive operation)
    - Reused across multiple requests
    - Allows easy model swapping later
    """
    print(f"[LOADING] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def create_embeddings(chunks, embedder):
    """
    Create L2-normalized embeddings for document chunks.

    Args:
        chunks (list): List of chunk dictionaries.
                       Each chunk must have a 'text' key.
        embedder: Pre-loaded SentenceTransformer model

    Returns:
        np.ndarray: L2-normalized embeddings
                    Shape -> (num_chunks, embedding_dim)
    """

    # Extract raw text from chunks
    texts = [c["text"] for c in chunks]

    print("[LOADING] Creating embeddings for chunks...")

    # Convert text → vectors
    embeddings = embedder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # IMPORTANT:
    # L2 normalization ensures:
    # cosine_similarity = dot_product
    faiss.normalize_L2(embeddings)

    # Safety check: each vector should have norm ≈ 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    avg_norm = np.mean(norms)

    if abs(avg_norm - 1.0) > 0.01:
        print(
            f"[WARNING] Embeddings may not be properly normalized "
            f"(avg L2 norm = {avg_norm:.4f})"
        )

    print("[OK] Embeddings created and normalized")
    print("Embeddings shape:", embeddings.shape)
    print(f"Average L2 norm: {avg_norm:.4f}")

    return embeddings