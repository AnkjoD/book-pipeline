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
from src.serve.rag.retriever import search
from src.serve.rag.verifier import verify_answer


def run_rag_pipeline(
    question: str,
    limit: int = 5,
    model: Optional[str] = None,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """
    Execute full RAG pipeline for a given user question.

    Returns:
        {
            "question": str,
            "answer": str,
            "verification_status": "ACCEPT" | "REGENERATE" | "ABSTAIN",
            "reason": str,
            "cited_books": list[dict],
            "retrieved_count": int,
            "citations": list[str],
            "raw_provenance": list[dict],
        }
    """
    # 1. Retrieve evidence documents
    documents = search(question, limit=limit)

    # 2. Build structured context block
    context_text, provenance = build_context(documents)

    # 3. Generate answer via local LLM
    kwargs = {"model": model} if model else {}
    gen_result = generate_answer(question, context_text, **kwargs)

    # 4. Verify citations against provenance
    verif = verify_answer(gen_result, provenance)

    # 5. Retry once if requested and rejected as REGENERATE
    if verif["status"] == "REGENERATE" and max_retries > 0:
        # Prompt model with explicit correction warning
        retry_context = (
            context_text
            + "\n\nWARNING: Your previous answer contained invalid citations. "
            + "Only cite [Doc N] where N corresponds to the provided documents above."
        )
        gen_result = generate_answer(question, retry_context, **kwargs)
        verif = verify_answer(gen_result, provenance)

    return {
        "question": question,
        "answer": gen_result.get("answer", ""),
        "verification_status": verif["status"],
        "reason": verif["reason"],
        "cited_books": verif["cited_books"],
        "retrieved_count": len(documents),
        "citations": verif["valid_citations"],
        "raw_provenance": provenance,
    }
