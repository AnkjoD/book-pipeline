"""
Context Builder for Evidence-Grounded RAG.

Formats retrieved documents into a clean, numbered context block:
[Doc 1] Title: ... | Author: ... | Year: ...
Content: ...

Preserves provenance (book_id, chunk_index, source) for the verifier layer.
"""
from typing import Any, Dict, List, Tuple


def build_context(
    documents: List[Dict[str, Any]],
    max_chars: int = 8000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Format a list of retrieved documents into an LLM-ready context block.

    Args:
        documents: Output from retriever.search()
        max_chars: Character ceiling to avoid blowing up the context window.

    Returns:
        (context_text, provenance_list)
        - context_text: String injected into LLM prompt
        - provenance_list: List of dicts mapping 1-based doc index to doc metadata
    """
    if not documents:
        return "No relevant documents found.", []

    doc_blocks: List[str] = []
    provenance_list: List[Dict[str, Any]] = []
    current_len = 0

    for idx, doc in enumerate(documents, start=1):
        book_id = doc.get("book_id", "unknown")
        title = doc.get("title", "Untitled")
        authors = doc.get("author_names", "Unknown")
        meta = doc.get("metadata") or {}
        year = meta.get("release_year") or "N/A"
        rating = meta.get("rating")
        rating_str = f"{rating:.2f}" if isinstance(rating, (int, float)) else "N/A"

        # Evidence text contains the meat of the document
        evidence = (doc.get("evidence") or "").strip()
        # Fallback if evidence is empty
        if not evidence:
            desc = meta.get("description") or ""
            evidence = f"Title: {title}\nAuthor: {authors}\nDescription: {desc}"

        header = f"[Doc {idx}] Title: {title} | Author(s): {authors} | Year: {year} | Rating: {rating_str}"
        block = f"{header}\n{evidence}\n"

        # Check character budget
        if current_len + len(block) > max_chars and doc_blocks:
            break

        doc_blocks.append(block)
        current_len += len(block)

        provenance_list.append({
            "doc_id": idx,
            "book_id": book_id,
            "title": title,
            "author_names": authors,
            "release_year": year,
            "source": doc.get("source", "unknown"),
            "chunk_index": doc.get("chunk_index", 0),
        })

    context_text = "\n---\n".join(doc_blocks)
    return context_text, provenance_list
