r"""Build lại collection pgvector từ Gold RAG document contract.

Chạy từ project root:
    .venv\Scripts\python.exe src\serve\rag\build_index.py
"""

import os

import duckdb
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH", "data/warehouse/gold.duckdb")
EMBED_MODEL = "all-MiniLM-L6-v2"
LIMIT = 10_000
BATCH_SIZE = 1_000
COLLECTION = "books"

PG_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
PG_PORT = os.getenv("VECTOR_DB_PORT", "5433")
PG_DB = os.getenv("VECTOR_DB_NAME", "books_rag")
PG_USER = os.getenv("VECTOR_DB_USER", "rag")
PG_PASSWORD = os.getenv("VECTOR_DB_PASSWORD", "rag")
CONNECTION_STRING = (
    f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

QUERY = """
select
    book_id,
    title,
    author_names,
    genres,
    moods,
    series_info,
    rating,
    ratings_count,
    pages_clean,
    release_year_clean,
    document_quality,
    document_hash,
    document_text
from rag_documents
where is_embedding_candidate = true
order by ratings_count desc nulls last, book_id
limit ?
"""


def _load_books(limit: int = LIMIT) -> list[dict]:
    """Đọc tối đa ``limit`` embedding candidates từ DuckDB."""
    if limit < 1:
        raise ValueError("limit must be at least 1")

    with duckdb.connect(DB_PATH, read_only=True) as connection:
        result = connection.execute(QUERY, [limit])
        columns = [column[0] for column in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]


def _make_documents(books: list[dict]) -> list[Document]:
    """Chia parent documents thành chunks và gắn metadata truy ngược nguồn."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    documents = []

    for book in books:
        chunks = splitter.split_text(book["document_text"])
        metadata = {
            "book_id": str(book["book_id"]),
            "title": book["title"] or "",
            "author_names": book["author_names"] or "",
            "genres": book["genres"] or "",
            "moods": book["moods"] or "",
            "series_info": book["series_info"] or "",
            "rating": float(book["rating"]) if book["rating"] is not None else None,
            "ratings_count": int(book["ratings_count"] or 0),
            "pages_clean": book["pages_clean"],
            "release_year_clean": book["release_year_clean"],
            "document_quality": book["document_quality"],
            "document_hash": book["document_hash"],
        }
        for chunk_index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk_index": chunk_index},
                )
            )

    return documents


def _build_index(documents: list[Document]) -> PGVector:
    """Embed chunks theo batch rồi rebuild collection ``books`` trong pgvector."""
    if not documents:
        raise ValueError("No documents to index")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        use_jsonb=True,
    )

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Indexed chunks: {min(start + len(batch), len(documents))}/{len(documents)}")

    return vectorstore


def main() -> None:
    print(f"DuckDB: {DB_PATH}")
    books = _load_books()
    print(f"Loaded books: {len(books)}")
    documents = _make_documents(books)
    print(f"Generated chunks: {len(documents)}")
    _build_index(documents)
    print(f"Built pgvector collection: {COLLECTION}")


if __name__ == "__main__":
    main()
