"""
End-to-End Selective Deterministic-First Evidence-Grounded RAG Pipeline.

Orchestrates:
1. Retriever: SQL / Vector / Mixed search with hard constraint filtering.
2. Context Builder: Format numbered [Doc N] blocks and provenance map.
3. Generator: Local Ollama LLM call enforcing source citations.
4. Verifier: Validates grounding, guards against ghost citations, handles abstain.
"""

from typing import Any, Dict, Optional

from src.serve.rag.context_builder import build_context
from src.serve.rag.generator import generate_answer
from src.serve.rag.query_understanding import QueryClarification, QueryUnderstandingError
from src.serve.rag.retriever import search
from src.serve.rag.verifier import verify_answer


def _catalog_answer(provenance: list[dict]) -> dict:
    """Render SQL catalog facts without asking an LLM to rewrite them."""
    lines = []
    citations = []
    for source in provenance:
        citation = f"Doc {source['doc_id']}"
        citations.append(citation)
        rating = source.get("rating")
        rating_text = f"{rating:.2f}" if isinstance(rating, (int, float)) else "N/A"
        lines.append(
            f"- **{source['title']}** — tác giả/credit: {source.get('author_names') or 'N/A'}; "
            f"năm: {source['release_year']}; số trang: {source.get('pages_clean')}; "
            f"điểm: {rating_text}. [{citation}]"
        )
    return {"answer": "\n".join(lines), "citations": citations, "has_answer": bool(lines)}


def run_rag_pipeline(
    question: str,
    limit: int = 5,
    model: Optional[str] = None,
    max_retries: int = 1,
    retrieval_mode: str = "baseline",
) -> Dict[str, Any]:
    """
    Execute full RAG pipeline for a given user question.

    Returns:
        {
            "question": str,
            "answer": str,
            "verification_status": "ACCEPT" | "REGENERATE" | "ABSTAIN" | "ERROR",
            "reason": str,
            "cited_books": list[dict],
            "retrieved_count": int,
            "citations": list[str],
            "raw_provenance": list[dict],
        }
    """
    # 1. Retrieve evidence documents
    try:
        if retrieval_mode == "baseline":
            documents = search(question, limit=limit)
        else:
            from src.serve.rag.experimental import search_experiment

            documents = search_experiment(question, limit=limit, mode=retrieval_mode)
    except (QueryClarification, QueryUnderstandingError) as exc:
        status = "CLARIFY" if isinstance(exc, QueryClarification) else "ERROR"
        return {
            "question": question,
            "answer": str(exc),
            "verification_status": status,
            "reason": str(exc),
            "cited_books": [],
            "retrieved_count": 0,
            "citations": [],
            "raw_provenance": [],
            "error": str(exc) if status == "ERROR" else None,
        }
    except Exception:
        # Do not leak connection strings or credentials through the UI.
        message = "Không truy cập được kho dữ liệu. Kiểm tra DuckDB và dịch vụ pgvector."
        return {
            "question": question,
            "answer": message,
            "verification_status": "ERROR",
            "reason": message,
            "cited_books": [],
            "retrieved_count": 0,
            "citations": [],
            "raw_provenance": [],
            "error": message,
        }

    # 2. Build structured context block
    context_text, provenance = build_context(documents)

    # 3. Generate answer via local LLM
    kwargs = {"model": model} if model else {}
    if provenance and all(
        doc.get("source") == "sql" or doc.get("catalog_only") for doc in documents
    ):
        gen_result = _catalog_answer(provenance)
    else:
        gen_result = generate_answer(question, context_text, **kwargs)

    # 4. Verify citations against provenance
    verif = verify_answer(gen_result, provenance)

    # 5. Retry once if requested and rejected as REGENERATE
    if verif["status"] == "REGENERATE" and max_retries > 0 and not gen_result.get("error"):
        # Prompt model with explicit correction warning
        retry_context = (
            context_text
            + "\n\nWARNING: Your previous answer contained invalid citations. "
            + "Only cite [Doc N] where N corresponds to the provided documents above."
        )
        gen_result = generate_answer(question, retry_context, **kwargs)
        verif = verify_answer(gen_result, provenance)

    # Only expose the generated answer when the final verification accepts it.
    answer = gen_result.get("answer", "")
    if verif["status"] == "REGENERATE":
        answer = "Mình chưa thể tạo câu trả lời đủ tin cậy từ các nguồn tìm được."
    elif verif["status"] == "ABSTAIN":
        answer = "Mình chưa tìm thấy đủ thông tin trong kho sách để trả lời."
    error = gen_result.get("error")
    if error:
        answer = error
        verif = {
            **verif,
            "status": "ERROR",
            "reason": error,
            "cited_books": [],
            "valid_citations": [],
        }

    return {
        "question": question,
        "answer": answer,
        "verification_status": verif["status"],
        "reason": verif["reason"],
        "cited_books": verif["cited_books"],
        "retrieved_count": len(documents),
        "citations": verif["valid_citations"],
        "raw_provenance": provenance,
        "error": error,
        "retrieval_trace": [doc.get("query_trace") for doc in documents if doc.get("query_trace")],
    }
