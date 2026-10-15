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

from .query_understanding import is_vietnamese, normalize_question

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
    f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?connect_timeout=5"
)

# Danh sách từ khóa thể loại (genre keywords)
GENRE_KEYWORDS = [
    "young adult",
    "fantasy",
    "sci-fi",
    "science fiction",
    "romance",
    "horror",
    "thriller",
    "mystery",
    "historical",
    "classics",
    "adventure",
    "dystopian",
    "non-fiction",
    "fiction",
    "trinh thám",
    "khoa học viễn tưởng",
    "kinh dị",
    "lãng mạn",
    "tiểu thuyết",
]

GENRE_ALIASES = {
    "trinh thám": "mystery",
    "khoa học viễn tưởng": "science fiction",
    "kinh dị": "horror",
    "lãng mạn": "romance",
    "tiểu thuyết": "fiction",
    "sci-fi": "science fiction",
}

# Danh sách từ khóa ngữ nghĩa (semantic keywords / mood / theme)
SEMANTIC_KEYWORDS = [
    "dark",
    "u tối",
    "grief",
    "mất mát",
    "memory",
    "ký ức",
    "loss",
    "hope",
    "hy vọng",
    "funny",
    "hài hước",
    "fast-paced",
    "nhịp nhanh",
    "slow-paced",
    "cozy",
    "ấm áp",
    "scary",
    "sợ hãi",
    "sad",
    "buồn",
    "emotional",
    "philosophical",
    "triết lý",
    "story about",
    "câu chuyện về",
    "truyện về",
    "atmosphere",
    "bầu không khí",
    "plot",
    "nội dung về",
    "theme",
    "chủ đề",
    "space exploration",
    "alien invasion",
    "detective",
    "murder",
    "time travel",
    "post-apocalyptic",
    "magic academy",
    "cyberpunk",
    "psychological",
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
        "min_rating_strict": False,
        "max_rating_strict": False,
        "min_pages_strict": False,
        "max_pages_strict": False,
    }

    # 1. Title / Author match (ví dụ: "Dune by Frank Herbert" hoặc "sách X của Y")
    by_match = re.search(
        r'["\']?([^"\']+)["\']?\s+(?:by|của|bởi)\s+["\']?([^"\',]+)["\']?', q, re.IGNORECASE
    )
    author_lookup = re.search(
        r"\b(?:books?|novels?)\s+(?:(?:written|authored)\s+)?by\s+([^,?!]+)",
        q,
        re.IGNORECASE,
    )
    quoted_title = re.search(r'"([^"]+)"', q)
    describes_plot = _has_semantic_concept(q) or bool(
        re.search(r"\b(?:story|stories|novel|novels|book|books)\s+about\b", q, re.IGNORECASE)
    )
    # "led by", "haunted by", etc. describe the plot, not authorship.
    # Explicit "books by ..." and quoted title lookups still identify authors.
    if author_lookup and not quoted_title:
        filters["author"] = author_lookup.group(1).strip().rstrip(".")
    elif by_match and (quoted_title or not describes_plot):
        cand_title = by_match.group(1).strip()
        cand_author = by_match.group(2).strip()
        # Strip trailing punctuation (? , . !)
        cand_author = re.sub(r"[\?\.,!]+$", "", cand_author).strip()
        # Clean question prefixes from cand_title
        cand_title = re.sub(
            r"^(?:what|which|show|list|tell|find|get|any)?\s*"
            r"(?:books?|all books?|cuốn sách|sách)?\s*(?:were|are)?\s*"
            r"(?:written|authored)?(?:\s+|$)",
            "",
            cand_title,
            flags=re.IGNORECASE,
        ).strip()
        if cand_author:
            filters["author"] = cand_author
        # Only treat as real title if not a generic question word
        if cand_title and cand_title.lower() not in (
            "book",
            "books",
            "sách",
            "all",
            "what",
            "which",
        ):
            filters["title"] = cand_title

    # An explicit quoted title can accompany numeric or semantic constraints.
    if quoted_title:
        filters["title"] = quoted_title.group(1).strip()

    # 2. Rating constraints
    # min rating: rating trên 4.2 / rated at least 4.2 / rated above 4
    min_r = re.search(
        r"(?:rating|rated|đánh giá)\s+"
        r"(?:at least|above|over|greater than|>=|>|từ|trên|cao hơn)\s*([0-9]+(?:\.[0-9]+)?)",
        q,
        re.IGNORECASE,
    )
    if not min_r:
        min_r = re.search(r"rated\s+above\s*([0-9]+(?:\.[0-9]+)?)", q, re.IGNORECASE)
    if min_r:
        filters["min_rating"] = float(min_r.group(1))
        filters["min_rating_strict"] = bool(
            re.search(r"above|over|greater than|trên|cao hơn|(?<![>])>(?!=)", min_r.group(0))
        )

    max_r = re.search(
        r"(?:rating|rated|đánh giá)\s+"
        r"(?:under|below|less than|<=|<|dưới|nhỏ hơn)\s*([0-9]+(?:\.[0-9]+)?)",
        q,
        re.IGNORECASE,
    )
    if max_r:
        filters["max_rating"] = float(max_r.group(1))
        filters["max_rating_strict"] = not bool(re.search(r"<=", max_r.group(0)))

    # 3. Pages constraints
    max_p = re.search(
        r"(?:under|below|less than|dưới|ít hơn|<=|<)\s*([0-9]+)\s*(?:pages|trang)", q, re.IGNORECASE
    )
    if max_p:
        filters["max_pages"] = int(max_p.group(1))
        filters["max_pages_strict"] = "<=" not in max_p.group(0)

    min_p = re.search(
        r"(?:over|above|more than|at least|trên|nhiều hơn|>=|>)\s*([0-9]+)\s*(?:pages|trang)",
        q,
        re.IGNORECASE,
    )
    if min_p:
        filters["min_pages"] = int(min_p.group(1))
        filters["min_pages_strict"] = not bool(re.search(r"at least|>=", min_p.group(0)))

    # 4. Year constraints
    after_y = re.search(r"(?:after|sau năm|sau|từ năm)\s*([0-9]{4,})", q, re.IGNORECASE)
    if after_y:
        filters["min_year"] = int(after_y.group(1)) + (
            0 if after_y.group(0).lower().startswith("từ năm") else 1
        )

    before_y = re.search(r"(?:before|trước năm|trước)\s*([0-9]{4,})", q, re.IGNORECASE)
    if before_y:
        filters["max_year"] = int(before_y.group(1)) - 1

    in_y = re.search(r"(?:in(?:\s+year)?|năm)\s+([0-9]{4,})", q, re.IGNORECASE)
    if in_y and not after_y and not before_y:
        filters["exact_year"] = int(in_y.group(1))

    # 5. Genre detection
    for g in GENRE_KEYWORDS:
        if re.search(r"\b" + re.escape(g) + r"\b", q, re.IGNORECASE):
            filters["genre"] = GENRE_ALIASES.get(g, g)
            break

    # 6. Kiểm tra ràng buộc bất khả thi (impossible constraints)
    if filters["min_rating"] is not None and (
        filters["min_rating"] > 5.0 or filters["min_rating"] < 0.0
    ):
        filters["is_impossible"] = True
    if filters["max_rating"] is not None and (
        filters["max_rating"] > 5.0 or filters["max_rating"] < 0.0
    ):
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
    if (
        filters["min_year"] is not None
        and filters["max_year"] is not None
        and filters["min_year"] > filters["max_year"]
    ):
        filters["is_impossible"] = True

    # Catch years outside regex range (e.g. 9999, 3000) via broad scan
    import re as _re

    for _m in _re.finditer(
        r"(?:in|year|after|before|published)\s+(?:year\s+)?([0-9]{4,})", q, _re.IGNORECASE
    ):
        _y = int(_m.group(1))
        if _y > 2100 or _y < 0:
            filters["is_impossible"] = True

    return filters


def _has_semantic_concept(question: str) -> bool:
    """Kiểm tra câu hỏi có chứa từ khóa ngữ nghĩa (mood/theme/plot) hay không."""
    q_lower = question.lower()
    for kw in SEMANTIC_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
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
        for k in [
            "min_rating",
            "max_rating",
            "min_pages",
            "max_pages",
            "min_year",
            "max_year",
            "exact_year",
            "title",
            "author",
            "genre",
        ]
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

        results.append(
            {
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
            }
        )

        if len(results) >= limit:
            break

    return results


def _sql_search(
    question: str, limit: int = 5, title_lookup: str | None = None
) -> list[dict[str, Any]]:
    """Truy vấn chính xác trên DuckDB Gold dựa trên parsed filters."""
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    filters = _parse_filters(question)
    if title_lookup is not None:
        filters["title"] = title_lookup
    if filters["is_impossible"]:
        return []

    conditions = ["is_embedding_candidate = true"]
    params: list[Any] = []

    if filters["title"] is not None:
        if title_lookup is None:
            conditions.append("lower(title) = lower(?)")
            params.append(filters["title"])
        else:
            conditions.append("(lower(title) = lower(?) OR starts_with(lower(title), lower(?)))")
            params.extend([filters["title"], filters["title"] + " "])

    if filters["author"] is not None:
        conditions.append("lower(author_names) LIKE ?")
        params.append(f"%{filters['author'].lower()}%")

    if filters["min_rating"] is not None:
        conditions.append("rating " + (">" if filters["min_rating_strict"] else ">=") + " ?")
        params.append(filters["min_rating"])

    if filters["max_rating"] is not None:
        conditions.append("rating " + ("<" if filters["max_rating_strict"] else "<=") + " ?")
        params.append(filters["max_rating"])

    if filters["min_pages"] is not None:
        conditions.append("pages_clean " + (">" if filters["min_pages_strict"] else ">=") + " ?")
        params.append(filters["min_pages"])

    if filters["max_pages"] is not None:
        conditions.append("pages_clean " + ("<" if filters["max_pages_strict"] else "<=") + " ?")
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
        results.append(
            {
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
            }
        )

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
            if (
                rating is None
                or rating < filters["min_rating"]
                or (filters["min_rating_strict"] and rating == filters["min_rating"])
            ):
                continue
        if filters["max_rating"] is not None:
            if (
                rating is None
                or rating > filters["max_rating"]
                or (filters["max_rating_strict"] and rating == filters["max_rating"])
            ):
                continue
        if filters["min_pages"] is not None:
            if (
                pages is None
                or pages < filters["min_pages"]
                or (filters["min_pages_strict"] and pages == filters["min_pages"])
            ):
                continue
        if filters["max_pages"] is not None:
            if (
                pages is None
                or pages > filters["max_pages"]
                or (filters["max_pages_strict"] and pages == filters["max_pages"])
            ):
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
        if filters["title"] is not None:
            title = str(meta.get("title", "")).casefold()
            requested_title = filters["title"].casefold()
            if title != requested_title:
                continue

        seen_books.add(book_id)
        dist = float(distance)
        similarity_score = max(0.0, 1.0 - dist)

        results.append(
            {
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
            }
        )

        if len(results) >= limit:
            break

    return results


def _prefer_exact_title(plan):
    """Use an exact catalog title when it exists; otherwise retain partial-title lookup."""
    from langchain_core.structured_query import Comparator, Comparison

    def resolve(node):
        if isinstance(node, Comparison):
            if node.attribute == "title" and node.comparator == Comparator.CONTAIN:
                with duckdb.connect(DB_PATH, read_only=True) as connection:
                    match = connection.execute(
                        "SELECT title FROM rag_documents WHERE lower(title) = lower(?) LIMIT 1",
                        [node.value],
                    ).fetchone()
                if match:
                    return node.model_copy(update={"comparator": Comparator.EQ, "value": match[0]})
            return node
        return node.model_copy(update={"arguments": [resolve(a) for a in node.arguments]})

    return plan.model_copy(update={"filter": resolve(plan.filter)}) if plan.filter else plan


def _self_query_search(question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Understand Vietnamese and execute the validated plan via SelfQueryRetriever."""
    from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
    from langchain_core.runnables import RunnableLambda

    from .query_translator import BookPGVectorTranslator
    from .query_understanding import plan_to_structured_query, understand_question
    from .reranking import rerank_enabled, rerank_with_fallback

    # Understand first: clarification must not require a running vector database.
    query_plan = understand_question(question)
    should_rerank = rerank_enabled() and not query_plan.catalog_only and not query_plan.title
    candidate_limit = max(30, limit) if should_rerank else limit
    plan = _prefer_exact_title(plan_to_structured_query(query_plan))
    translator = BookPGVectorTranslator()
    query_text, kwargs = translator.visit_structured_query(plan)
    searcher = SelfQueryRetriever(
        query_constructor=RunnableLambda(lambda inputs: plan),
        vectorstore=_load_vectorstore(),
        structured_query_translator=translator,
        search_kwargs={"k": max(20, candidate_limit * 4)},
    )
    documents = searcher.invoke(question)
    results = []
    seen = set()
    for doc in documents:
        meta = doc.metadata or {}
        book_id = str(meta.get("book_id", ""))
        if not book_id or book_id in seen:
            continue
        seen.add(book_id)
        results.append(
            {
                "book_id": book_id,
                "title": meta.get("title", ""),
                "author_names": meta.get("author_names", ""),
                "evidence": doc.page_content,
                "source": "self_query",
                "score": None,
                "distance": None,
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": dict(meta),
                "query_trace": {"semantic_query": query_text, **kwargs},
                "catalog_only": query_plan.catalog_only or not query_plan.semantic_query.strip(),
            }
        )
        if len(results) >= candidate_limit:
            break
    return rerank_with_fallback(query_text, results, limit) if should_rerank else results


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

    from .reranking import rerank_enabled, rerank_with_fallback

    if is_vietnamese(question):
        normalized = normalize_question(question)
        year = re.search(r"\bnam\s+([0-9]{4,})", normalized)
        if year and int(year.group(1)) > 2100:
            return []
        filters = _parse_filters(question)
        if filters["is_impossible"]:
            return []
        # Only the explicit, simple title-author form bypasses the query model.
        if (
            filters["title"]
            and filters["author"]
            and not re.match(r"^(tim|cho|goi|noi dung|cuon)\b", normalized)
            and not _has_semantic_concept(question)
        ):
            exact = _sql_search(question, limit=limit)
            if exact:
                return exact
        return _self_query_search(question, limit=limit)

    route = _route(question)
    if route == "sql":
        return _sql_search(question, limit=limit)
    if route == "vector":
        # Bare book names should not require an embedding model to resolve.
        if not _has_semantic_concept(question):
            exact = _sql_search(question, limit=limit, title_lookup=question.strip())
            if exact:
                return exact
        if rerank_enabled():
            return rerank_with_fallback(
                question, _vector_search(question, limit=max(30, limit)), limit
            )
        return _vector_search(question, limit=limit)
    if rerank_enabled():
        return rerank_with_fallback(question, _mixed_search(question, limit=max(30, limit)), limit)
    return _mixed_search(question, limit=limit)
