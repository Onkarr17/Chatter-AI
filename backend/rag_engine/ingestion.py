from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def safe_print(text, max_length=None):
    """
    Safely print text that may contain Unicode characters.
    Handles Windows console encoding issues.
    """
    try:
        if max_length:
            text = text[:max_length]
        print(text)
    except UnicodeEncodeError:
        try:
            safe_text = text.encode("ascii", errors="replace").decode("ascii")
            print(safe_text)
        except Exception:
            print("[Text contains non-printable characters]")


def read_pdf(pdf_path: str):
    """
    Read a PDF file and return page-wise documents.

    Args:
        pdf_path: Absolute or relative path to the PDF file

    Returns:
        List of page documents
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print("[OK] Total pages:", len(pages))
    if pages:
        print("\n--- Page 1 Preview (first 400 chars) ---\n")
        safe_print(pages[0].page_content, max_length=400)

    return pages


def split_into_chunks(pages, chunk_size=800, chunk_overlap=150):
    """
    Split PDF pages into overlapping text chunks for RAG.

    Each chunk format:
    {
        "id": "p<page>_c<chunk_index>",
        "page": <page_number>,
        "text": <chunk_text>
    }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for page_index, page_doc in enumerate(pages):
        page_no = page_index + 1
        page_text = page_doc.page_content or ""

        split_texts = splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(split_texts):
            chunks.append({
                "id": f"p{page_no}_c{chunk_index}",
                "page": page_no,
                "text": chunk_text
            })

    print("[OK] Total chunks created:", len(chunks))

    # SAFETY CHECK: no text extracted from PDF
# This prevents downstream crashes in embeddings / FAISS
    if not chunks:
        raise ValueError(
          "No text chunks could be extracted from the PDF. "
          "The document may be scanned, image-based, or unsupported."
    )

# Show sample chunks (safe for Unicode/Windows console)
    if chunks:
        print("\n--- Sample Chunk 1 ---\n")
        safe_print(chunks[0]["text"], max_length=250)

        if len(chunks) > 1:
            print("\n--- Sample Chunk 2 ---\n")
            safe_print(chunks[1]["text"], max_length=250)

    return chunks