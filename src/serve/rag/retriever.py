"""Online retrieval module cho Selective Deterministic-First Evidence-Grounded RAG.

Hỗ trợ 3 luồng truy xuất:
  1. SQL (deterministic): Lọc chính xác rating, pages, release year, author, title trên Gold DuckDB.
  2. Vector (semantic): Tìm kiếm ngữ nghĩa trên PostgreSQL pgvector dựa trên cosine distance.
  3. Mixed (hybrid): Kết hợp lọc hard constraints và xếp hạng ngữ nghĩa.
"""

import functools
import os
import re
from typing import Any

import duckdb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

load_dotenv()

DB_PATH = os.getenv("DUCKDB_PATH", "data/warehouse/gold.duckdb")
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "books"

PG_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
PG_PORT = os.getenv("VECTOR_DB_PORT", "5433")
PG_DB = os.getenv("VECTOR_DB_NAME", "books_rag")
PG_USER = os.getenv("VECTOR_DB_USER", "rag")
PG_PASSWORD = os.getenv("VECTOR_DB_PASSWORD", "rag")
CONNECTION_STRING = (
    f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

# Danh sách từ khóa thể loại (genre keywords)
GENRE_KEYWORDS = [
    "fantasy", "sci-fi", "science fiction", "romance", "horror", "thriller",
    "mystery", "historical", "classics", "adventure", "dystopian", "non-fiction",
    "fiction", "trinh thám", "khoa học viễn tưởng", "kinh dị", "lãng mạn", "tiểu thuyết"
]

# Danh sách từ khóa ngữ nghĩa (semantic keywords / mood / theme)
SEMANTIC_KEYWORDS = [
    "dark", "u tối", "grief", "mất mát", "memory", "ký ức", "loss", "hope",
    "hy vọng", "funny", "hài hước", "fast-paced", "nhịp nhanh", "slow-paced",
    "cozy", "ấm áp", "scary", "sợ hãi", "sad", "buồn", "emotional", "philosophical",
    "triết lý", "story about", "câu chuyện về", "truyện về", "atmosphere",
    "bầu không khí", "plot", "nội dung về", "theme", "chủ đề"
]


@functools.lru_cache(maxsize=1)
def _load_vectorstore() -> PGVector:
    """Khởi tạo và cache vectorstore PGVector với model all-MiniLM-L6-v2."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


def _parse_filters(question: str) -> dict[str, Any]:
    """Phân tích các ràng buộc cấu trúc (rating, pages, year, author, title, genre) từ câu hỏi."""
    q = question.strip()
    filters: dict[str, Any] = {
        "min_rating": None,
        "max_rating": None,
        "min_pages": None,
        "max_pages": None,
        "min_year": None,
        "max_year": None,
        "exact_year": None,
        "title": None,
        "author": None,
        "genre": None,
        "is_impossible": False,
    }

    # 1. Title / Author match (ví dụ: "Dune by Frank Herbert" hoặc "sách X của Y")
    by_match = re.search(r'["\']?([^"\']+)["\']?\s+(?:by|của|bởi)\s+["\']?([^"\',]+)["\']?', q, re.IGNORECASE)
    if by_match:
        cand_title = by_match.group(1).strip()
        cand_author = by_match.group(2).strip()
        # Strip trailing punctuation (? , . !)
        cand_author = re.sub(r'[\?\.,!]+$', '', cand_author).strip()
        # Clean question prefixes from cand_title
        cand_title = re.sub(
            r'^(?:what|which|show|list|tell|find|get|any)?\s*(?:books?|all books?|cuốn sách|sách)?\s*(?:were|are)?\s*(?:written|authored)?(?:\s+|$)',
            '', cand_title, flags=re.IGNORECASE
        ).strip()
        if cand_author:
            filters['author'] = cand_author
        # Only treat as real title if not a generic question word
        if cand_title and cand_title.lower() not in ('book', 'books', 'sách', 'all', 'what', 'which'):
            filters['title'] = cand_title

    # 2. Rating constraints
    # min rating: rating trên 4.2 / rated at least 4.2 / rated above 4
    min_r = re.search(r'(?:rating|rated|đánh giá)\s+(?:at least|above|over|greater than|>=|>|từ|trên|cao hơn)\s*([0-9]+(?:\.[0-9]+)?)', q, re.IGNORECASE)
    if not min_r:
        min_r = re.search(r'rated\s+above\s*([0-9]+(?:\.[0-9]+)?)', q, re.IGNORECASE)
    if min_r:
        filters["min_rating"] = float(min_r.group(1))

    max_r = re.search(r'(?:rating|rated|đánh giá)\s+(?:under|below|less than|<=|<|dưới|nhỏ hơn)\s*([0-9]+(?:\.[0-9]+)?)', q, re.IGNORECASE)
    if max_r:
        filters["max_rating"] = float(max_r.group(1))

    # 3. Pages constraints
    max_p = re.search(r'(?:under|below|less than|dưới|ít hơn|<=|<)\s*([0-9]+)\s*(?:pages|trang)', q, re.IGNORECASE)
    if max_p:
        filters["max_pages"] = int(max_p.group(1))

    min_p = re.search(r'(?:over|above|more than|at least|trên|nhiều hơn|>=|>)\s*([0-9]+)\s*(?:pages|trang)', q, re.IGNORECASE)
    if min_p:
        filters["min_pages"] = int(min_p.group(1))

    # 4. Year constraints
    after_y = re.search(r'(?:after|sau năm|sau|từ năm)\s*([1-2][0-9]{3})', q, re.IGNORECASE)
    if after_y:
        filters["min_year"] = int(after_y.group(1))

    before_y = re.search(r'(?:before|trước năm|trước)\s*([1-2][0-9]{3})', q, re.IGNORECASE)
    if before_y:
        filters["max_year"] = int(before_y.group(1))

    in_y = re.search(r'(?:in|năm)\s+([1-2][0-9]{3})', q, re.IGNORECASE)
    if in_y and not after_y and not before_y:
        filters["exact_year"] = int(in_y.group(1))

    # 5. Genre detection
    for g in GENRE_KEYWORDS:
        if re.search(r'\b' + re.escape(g) + r'\b', q, re.IGNORECASE):
            filters["genre"] = g
            break

    # 6. Kiểm tra ràng buộc bất khả thi (impossible constraints)
    if filters["min_rating"] is not None and (filters["min_rating"] > 5.0 or filters["min_rating"] < 0.0):
        filters["is_impossible"] = True
    if filters["max_rating"] is not None and (filters["max_rating"] > 5.0 or filters["max_rating"] < 0.0):
        filters["is_impossible"] = True
    if filters["min_rating"] is not None and filters["max_rating"] is not None:
        if filters["min_rating"] > filters["max_rating"]:
            filters["is_impossible"] = True

    if filters["max_pages"] is not None and filters["max_pages"] <= 0:
        filters["is_impossible"] = True
    if filters["min_pages"] is not None and filters["min_pages"] < 0:
        filters["is_impossible"] = True
    if filters["min_pages"] is not None and filters["max_pages"] is not None:
        if filters["min_pages"] > filters["max_pages"]:
            filters["is_impossible"] = True

    # Year range sanity: must be 0-2100
    for yk in ("min_year", "max_year", "exact_year"):
        yv = filters.get(yk)
        if yv is not None and (yv > 2100 or yv < 0):
            filters["is_impossible"] = True

    # Catch years outside regex range (e.g. 9999, 3000) via broad scan
    import re as _re
    for _m in _re.finditer(r'(?:in|year|after|before|published)\s+(?:year\s+)?([0-9]{4,})', q, _re.IGNORECASE):
        _y = int(_m.group(1))
        if _y > 2100 or _y < 0:
            filters["is_impossible"] = True

    return filters


def _has_semantic_concept(question: str) -> bool:
    """Kiểm tra câu hỏi có chứa từ khóa ngữ nghĩa (mood/theme/plot) hay không."""
    q_lower = question.lower()
    for kw in SEMANTIC_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', q_lower):
            return True
    return False


def _route(question: str) -> str:
    """Phân luồng câu hỏi thành 'sql', 'vector', hoặc 'mixed'."""
    filters = _parse_filters(question)
    has_semantic = _has_semantic_concept(question)

    if filters["is_impossible"]:
        return "sql"

    has_structured = any(
        filters[k] is not None
        for k in ["min_rating", "max_rating", "min_pages", "max_pages", "min_year", "max_year", "exact_year", "title", "author", "genre"]
    )

    if has_structured and has_semantic:
        return "mixed"
    if has_structured:
        return "sql"
    return "vector"


def _vector_search(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Tìm kiếm ngữ nghĩa trên pgvector và khử trùng lặp theo parent book_id."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    vectorstore = _load_vectorstore()
    fetch_k = max(limit * 4, 20)
    raw_results = vectorstore.similarity_search_with_score(question, k=fetch_k)

    seen_books: set[str] = set()
    results: list[dict[str, Any]] = []

    for doc, distance in raw_results:
        meta = doc.metadata or {}
        book_id = str(meta.get("book_id", ""))
        if not book_id or book_id in seen_books:
            continue
        seen_books.add(book_id)

        dist = float(distance)
        similarity_score = max(0.0, 1.0 - dist)

        results.append({
            "book_id": book_id,
            "title": meta.get("title", ""),
            "author_names": meta.get("author_names", ""),
            "evidence": doc.page_content,
            "score": round(similarity_score, 4),
            "distance": round(dist, 4),
            "source": "vector",
            "chunk_index": int(meta.get("chunk_index", 0)),
            "metadata": {
                "rating": meta.get("rating"),
                "ratings_count": meta.get("ratings_count", 0),
                "pages_clean": meta.get("pages_clean"),
                "release_year_clean": meta.get("release_year_clean"),
                "genres": meta.get("genres", ""),
                "moods": meta.get("moods", ""),
                "document_hash": meta.get("document_hash", ""),
            },
        })

        if len(results) >= limit:
            break

    return results


def _sql_search(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Truy vấn chính xác trên DuckDB Gold dựa trên parsed filters."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    filters = _parse_filters(question)
    if filters["is_impossible"]:
        return []

    conditions = ["is_embedding_candidate = true"]
    params: list[Any] = []

    if filters["title"] is not None:
        conditions.append("lower(title) = lower(?)")
        params.append(filters["title"])

    if filters["author"] is not None:
        conditions.append("lower(author_names) LIKE ?")
        params.append(f"%{filters['author'].lower()}%")

    if filters["min_rating"] is not None:
        conditions.append("rating >= ?")
        params.append(filters["min_rating"])

    if filters["max_rating"] is not None:
        conditions.append("rating <= ?")
        params.append(filters["max_rating"])

    if filters["min_pages"] is not None:
        conditions.append("pages_clean >= ?")
        params.append(filters["min_pages"])

    if filters["max_pages"] is not None:
        conditions.append("pages_clean <= ?")
        params.append(filters["max_pages"])

    if filters["min_year"] is not None:
        conditions.append("release_year_clean >= ?")
        params.append(filters["min_year"])

    if filters["max_year"] is not None:
        conditions.append("release_year_clean <= ?")
        params.append(filters["max_year"])

    if filters["exact_year"] is not None:
        conditions.append("release_year_clean = ?")
        params.append(filters["exact_year"])

    if filters["genre"] is not None:
        conditions.append("lower(genres) LIKE ?")
        params.append(f"%{filters['genre'].lower()}%")

    where_clause = " AND ".join(conditions)
    query = f"""
    SELECT
        book_id,
        title,
        author_names,
        rating,
        ratings_count,
        pages_clean,
        release_year_clean,
        genres,
        moods,
        document_hash,
        document_text
    FROM rag_documents
    WHERE {where_clause}
    ORDER BY ratings_count DESC NULLS LAST, rating DESC NULLS LAST, book_id
    LIMIT ?
    """
    params.append(limit)

    with duckdb.connect(DB_PATH, read_only=True) as connection:
        rows = connection.execute(query, params).fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        results.append({
            "book_id": str(r[0]),
            "title": r[1] or "",
            "author_names": r[2] or "",
            "evidence": r[10][:500] if r[10] else "",
            "score": 1.0,
            "distance": 0.0,
            "source": "sql",
            "chunk_index": 0,
            "metadata": {
                "rating": float(r[3]) if r[3] is not None else None,
                "ratings_count": int(r[4] or 0),
                "pages_clean": r[5],
                "release_year_clean": r[6],
                "genres": r[7] or "",
                "moods": r[8] or "",
                "document_hash": r[9] or "",
            },
        })

    return results


def _mixed_search(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Tìm kiếm kết hợp lọc hard constraints và semantic ranking."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    filters = _parse_filters(question)
    if filters["is_impossible"]:
        return []

    vectorstore = _load_vectorstore()
    fetch_k = max(limit * 15, 60)
    raw_results = vectorstore.similarity_search_with_score(question, k=fetch_k)

    seen_books: set[str] = set()
    results: list[dict[str, Any]] = []

    for doc, distance in raw_results:
        meta = doc.metadata or {}
        book_id = str(meta.get("book_id", ""))
        if not book_id or book_id in seen_books:
            continue

        # Kiểm tra nghiêm ngặt các hard constraints
        rating = meta.get("rating")
        pages = meta.get("pages_clean")
        year = meta.get("release_year_clean")
        genres = (meta.get("genres") or "").lower()
        authors = (meta.get("author_names") or "").lower()

        if filters["min_rating"] is not None:
            if rating is None or rating < filters["min_rating"]:
                continue
        if filters["max_rating"] is not None:
            if rating is None or rating > filters["max_rating"]:
                continue
        if filters["min_pages"] is not None:
            if pages is None or pages < filters["min_pages"]:
                continue
        if filters["max_pages"] is not None:
            if pages is None or pages > filters["max_pages"]:
                continue
        if filters["min_year"] is not None:
            if year is None or year < filters["min_year"]:
                continue
        if filters["max_year"] is not None:
            if year is None or year > filters["max_year"]:
                continue
        if filters["exact_year"] is not None:
            if year is None or year != filters["exact_year"]:
                continue
        if filters["genre"] is not None:
            if filters["genre"].lower() not in genres:
                continue
        if filters["author"] is not None:
            if filters["author"].lower() not in authors:
                continue

        seen_books.add(book_id)
        dist = float(distance)
        similarity_score = max(0.0, 1.0 - dist)

        results.append({
            "book_id": book_id,
            "title": meta.get("title", ""),
            "author_names": meta.get("author_names", ""),
            "evidence": doc.page_content,
            "score": round(similarity_score, 4),
            "distance": round(dist, 4),
            "source": "mixed",
            "chunk_index": int(meta.get("chunk_index", 0)),
            "metadata": {
                "rating": rating,
                "ratings_count": meta.get("ratings_count", 0),
                "pages_clean": pages,
                "release_year_clean": year,
                "genres": meta.get("genres", ""),
                "moods": meta.get("moods", ""),
                "document_hash": meta.get("document_hash", ""),
            },
        })

        if len(results) >= limit:
            break

    return results


def search(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Public retrieval facade duy nhất cho chatbot và downstream consumers.

    Phân luồng tự động:
      - 'sql': nếu chỉ có ràng buộc cấu trúc hoặc lookup
      - 'vector': nếu là câu hỏi ngữ nghĩa thuần túy
      - 'mixed': nếu kết hợp cả bộ lọc cấu trúc và ngữ nghĩa
    """
    if not isinstance(question, str) or not question.strip():
        return []  # empty/invalid -> no results
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    route = _route(question)
    if route == "sql":
        return _sql_search(question, limit=limit)
    if route == "vector":
        return _vector_search(question, limit=limit)
    return _mixed_search(question, limit=limit)
