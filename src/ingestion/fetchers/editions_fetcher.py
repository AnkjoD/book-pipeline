"""
Fetcher cho entity 'editions' -- chứa ISBN, publisher, format ấn bản.
Bổ sung cho 'books' để phân tích theo ấn bản/nhà xuất bản (phục vụ ETL/RAG cá nhân).

Field CHỦ ĐỘNG BỎ (theo cùng logic với books_fetcher.py):
    - cached_image, cached_contributors, cached_tags (trùng dữ liệu, đã có ở book/contributions)
    - book_mappings, list_books, images (không cần cho RAG nội dung)
    - locked, curation_status, created_by_user_id (admin nội bộ)
"""

from ingestion.fetchers.base_fetcher import BaseFetcher

EDITIONS_QUERY = """
query FetchEditions($limit: Int!, $offset: Int!, $where: editions_bool_exp) {
  editions(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    book_id
    title
    subtitle
    alternative_titles
    edition_format
    edition_information
    physical_format
    physical_information
    isbn_10
    isbn_13
    isbn_10_valid
    isbn_13_valid
    asin
    pages
    audio_seconds
    release_date
    release_year
    language_id
    country_id
    publisher_id
    reading_format_id
    rating
    score
    users_count
    users_read_count
    lists_count
    state
    canonical_id
    original_book_id
    source
    created_at
    updated_at
  }
}
"""


class EditionsFetcher(BaseFetcher):
    entity_name = "editions"
    query = EDITIONS_QUERY
