import os
from groq import Groq


# -------------------------------------------------
# 1. Low-level Groq chat call
# -------------------------------------------------

def groq_chat(
    prompt: str,
    max_tokens: int = 350,
    conversation_history=None,
):
    """
    Low-level Groq call.
    Returns BOTH answer + token usage (for analytics).
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "answer": "[ERROR] GROQ_API_KEY missing.",
            "usage": None,
        }

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions based on the provided "
                "context and conversation history. Remember information shared in previous "
                "messages. If the context contains relevant information that can help answer "
                "the question (even if not exact), use it to provide a helpful answer. "
                "Only say 'Not found in the PDF' if the context truly has no relevant "
                "information at all. Always format answers in a structured, point-wise "
                "manner with clear formatting, bold headers for new points, numbered or "
                "bulleted lists, and proper spacing between points."
            ),
        }
    ]

    if conversation_history:
        for msg in conversation_history:
            messages.append(
                {"role": msg["role"], "content": msg["content"]}
            )

    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )

    return {
        "answer": resp.choices[0].message.content,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "model": resp.model,
        },
    }


# -------------------------------------------------
# 2. PDF-based answering
# -------------------------------------------------

def answer_from_pdf(
    question: str,
    retrieved,
    top_k: int = 8,
    conversation_history=None,
    max_tokens: int = 400,
):
    """
    Answer using retrieved PDF chunks.
    """

    if not retrieved:
        return {
            "answer": "Not found in the PDF.",
            "usage": None,
        }

    context_parts = []
    pages_used = set()

    for chunk, similarity in retrieved[:top_k]:
        pages_used.add(chunk["page"])
        context_parts.append(
            f"[Page {chunk['page']} | {chunk['id']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    pages_used = sorted(pages_used)

    prompt = f"""
Answer the question based on the context provided below.

Only say "Not found in the PDF" if the context truly has no relevant information.

Context:
{context}

Question:
{question}
"""

    result = groq_chat(
        prompt,
        max_tokens=max_tokens,
        conversation_history=conversation_history,
    )

    answer = result["answer"]

    if "Not found in the PDF" not in answer and pages_used:
        answer += "\n\nSources: " + ", ".join(
            f"Page {p}" for p in pages_used
        )

    return {
        "answer": answer,
        "usage": result["usage"],
    }


# -------------------------------------------------
# 3. General knowledge fallback
# -------------------------------------------------

def answer_general(
    question: str,
    conversation_history=None,
    max_tokens: int = 250,
):
    """
    General LLM answer without PDF.
    """

    prompt = f"""
Answer the question clearly and concisely.

Question:
{question}
"""

    return groq_chat(
        prompt,
        max_tokens=max_tokens,
        conversation_history=conversation_history,
    )