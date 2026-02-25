import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index using Inner Product (IP).

    IMPORTANT:
    - Embeddings MUST already be L2-normalized.
    - With normalized vectors:
        inner_product == cosine_similarity

    Args:
        embeddings (np.ndarray): Shape (num_chunks, embedding_dim)

    Returns:
        faiss.IndexFlatIP: FAISS index ready for search
    """

    dim = embeddings.shape[1]

    # IndexFlatIP performs inner-product search
    # With normalized embeddings, this equals cosine similarity
    index = faiss.IndexFlatIP(dim)

    # Add all vectors to index
    index.add(embeddings)

    print("[OK] FAISS index built (cosine similarity)")
    print("Total vectors indexed:", index.ntotal)

    return index


def retrieve_top_chunks(
    query: str,
    embedder,
    index,
    chunks,
    top_k: int = 8,
):
    """
    Retrieve top-k relevant chunks using cosine similarity.

    Steps:
    1. Embed query
    2. Normalize query embedding
    3. Search FAISS index
    4. Attach similarity scores to chunks

    Args:
        query (str): User question
        embedder: Loaded SentenceTransformer model
        index: FAISS index
        chunks (list): List of chunk dictionaries
        top_k (int): Number of chunks to retrieve

    Returns:
        retrieved (list): [(chunk_dict, similarity_score), ...]
        best_similarity (float): Highest cosine similarity score
    """

    # Convert query text → embedding vector
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    # Ensure correct shape (FAISS expects 2D array)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Normalize query embedding for cosine similarity
    faiss.normalize_L2(query_embedding)

    # Perform FAISS search
    scores, indices = index.search(query_embedding, top_k)

    scores = scores[0]
    indices = indices[0]

    # Safety check:
    # Cosine similarity must always be in [-1, 1]
    max_score = scores.max() if len(scores) else 0.0
    if max_score > 1.01:
        raise ValueError(
            "Similarity score > 1 detected. "
            "Embeddings are likely not normalized correctly."
        )

    retrieved = []

    for idx, score in zip(indices, scores):
        if 0 <= idx < len(chunks):
            # Store raw similarity score (can be negative)
            retrieved.append((chunks[idx], float(score)))

    best_similarity = float(scores[0]) if len(scores) else 0.0

    return retrieved, best_similarity