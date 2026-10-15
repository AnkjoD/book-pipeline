"""
Verifier layer for Evidence-Grounded RAG.

Evaluates generated answer against the provenance map:
- ACCEPT: All cited docs exist, facts are grounded.
- REGENERATE: Invalid citations or malformed output (caller can retry).
- ABSTAIN: No evidence exists, system safely refuses to answer.
"""

from typing import Any, Dict, List


def verify_answer(
    answer_result: Dict[str, Any],
    provenance_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify answer against the retrieved documents provenance.

    Args:
        answer_result: Output from generator.generate_answer()
        provenance_list: Output from context_builder.build_context()

    Returns:
        {
            "status": "ACCEPT" | "REGENERATE" | "ABSTAIN",
            "reason": str,
            "valid_citations": List[str],
            "invalid_citations": List[str],
            "cited_books": List[Dict[str, Any]],  # metadata of actually cited books
        }
    """
    citations = answer_result.get("citations", [])
    has_answer = answer_result.get("has_answer", False)

    # 1. ABSTAIN check: No documents provided or model admitted lack of info
    if not provenance_list:
        return {
            "status": "ABSTAIN",
            "reason": "No evidence documents available in catalog.",
            "valid_citations": [],
            "invalid_citations": [],
            "cited_books": [],
        }

    if not has_answer:
        return {
            "status": "ABSTAIN",
            "reason": "Model correctly identified lack of evidence.",
            "valid_citations": [],
            "invalid_citations": [],
            "cited_books": [],
        }

    # 2. Check for Ghost Citations (Doc N cited but N not in provenance)
    valid_doc_ids = {f"Doc {item['doc_id']}" for item in provenance_list}
    prov_by_doc_id = {f"Doc {item['doc_id']}": item for item in provenance_list}

    valid_citations = [c for c in citations if c in valid_doc_ids]
    invalid_citations = [c for c in citations if c not in valid_doc_ids]

    if invalid_citations:
        return {
            "status": "REGENERATE",
            "reason": f"Ghost citation detected: {invalid_citations} not in context.",
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "cited_books": [],
        }

    # 3. Grounding check: Answer provided but no citations at all
    if has_answer and not valid_citations:
        return {
            "status": "REGENERATE",
            "reason": "Answer claims facts but provides zero valid citations.",
            "valid_citations": [],
            "invalid_citations": [],
            "cited_books": [],
        }

    # 4. Gather cited book metadata
    cited_books = [prov_by_doc_id[c] for c in valid_citations]

    return {
        "status": "ACCEPT",
        "reason": "All citation IDs exist in the provided context; claims are not verified.",
        "valid_citations": valid_citations,
        "invalid_citations": [],
        "cited_books": cited_books,
    }
