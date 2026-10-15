"""
Streamlit Chatbot for Evidence-Grounded Book RAG.

Displays:
- Chat interaction
- Verifier verdict badge (ACCEPT / ABSTAIN / REGENERATE)
- Collapsible evidence provenance (Title, Author, Rating, Source)
"""

import streamlit as st

from src.serve.rag.pipeline import run_rag_pipeline

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Book Pipeline RAG Assistant")
st.caption("Evidence-Grounded Book Catalog Q&A with Local LLM (Ollama)")

# Sidebar configurations
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Local Model",
        options=["qwen2.5:7b-instruct-q4_K_M", "qwen3.5:4b"],
        index=0,
    )
    doc_limit = st.slider("Document Limit (Top-K)", min_value=1, max_value=10, value=5)
    search_choices = {
        "Hiện tại": "baseline",
        "Từ khóa + ngữ nghĩa (thử nghiệm)": "hybrid",
        "Theo quan hệ GraphRAG (thử nghiệm)": "graph",
    }
    search_choice = st.selectbox("Cách tìm sách", options=list(search_choices))
    retrieval_mode = search_choices[search_choice]
    if retrieval_mode != "baseline":
        from src.serve.rag.experimental import load_corpus

        try:
            experiment_manifest, _ = load_corpus()
            st.caption(
                f"Thử nghiệm trên {experiment_manifest['books']} sách; "
                "kho tìm kiếm hiện tại được giữ riêng."
            )
        except (FileNotFoundError, ValueError):
            st.info("Chưa có kho thử nghiệm hợp lệ. Bạn có thể dùng cách tìm hiện tại.")
    st.markdown("---")
    st.markdown(
        "**Pipeline Features**:\n"
        "- 🔍 **Deterministic-First**: SQL exact search before Vector\n"
        "- 🛑 **Zero-Cost Abstain**: Blocks impossible queries (e.g. year 9999)\n"
        "- 🛡️ **Verifier Guard**: Checks ghost citations before output"
    )

# Session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "meta" in msg:
            meta = msg["meta"]
            status = meta.get("verification_status")
            badge_color = {
                "ACCEPT": "🟢 ACCEPT",
                "ABSTAIN": "🟡 ABSTAIN",
                "REGENERATE": "🔴 REGENERATE",
                "ERROR": "🔴 ERROR",
                "CLARIFY": "🟡 Cần làm rõ",
            }.get(status, status)

            st.caption(f"Status: **{badge_color}** | Retrieved: {meta.get('retrieved_count')} docs")

            cited = meta.get("cited_books", [])
            if cited:
                with st.expander(f"📖 Cited Sources ({len(cited)})"):
                    for b in cited:
                        st.markdown(
                            f"- **[Doc {b.get('doc_id')}] {b.get('title')}** "
                            f"by *{b.get('author_names')}* "
                            f"(Year: {b.get('release_year')}, Source: `{b.get('source')}`)"
                        )

# Input prompt
if prompt := st.chat_input("Hỏi bằng tiếng Việt về sách, tác giả, nội dung hoặc điều kiện tìm..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching catalog & generating verified answer..."):
            result = run_rag_pipeline(
                prompt,
                limit=doc_limit,
                model=model_choice,
                retrieval_mode=retrieval_mode,
            )

        st.markdown(result["answer"])

        status = result["verification_status"]
        badge_label = {
            "ACCEPT": "🟢 ACCEPT (Citations checked)",
            "ABSTAIN": "🟡 ABSTAIN (No Evidence)",
            "REGENERATE": "🔴 REGENERATE (Citation Mismatch)",
            "ERROR": "🔴 ERROR (Service unavailable)",
            "CLARIFY": "🟡 Cần bạn nói rõ điều kiện",
        }.get(status, status)

        st.caption(f"Status: **{badge_label}** | Retrieved: {result['retrieved_count']} docs")

        if result["cited_books"]:
            with st.expander(f"📖 Cited Sources ({len(result['cited_books'])})"):
                for b in result["cited_books"]:
                    st.markdown(
                        f"- **[Doc {b.get('doc_id')}] {b.get('title')}** "
                        f"by *{b.get('author_names')}* "
                        f"(Year: {b.get('release_year')}, Source: `{b.get('source')}`)"
                    )

    # Append assistant message with metadata
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": result,
        }
    )
