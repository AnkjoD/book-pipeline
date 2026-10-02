"""
Fetcher cho entity 'books' — core entity của toàn bộ pipeline.

Field đã CHỦ ĐỘNG BỎ (theo quyết định trong quá trình thiết kế):
    - cached_image, cached_header_image, cached_featured_series
      (blob ảnh nặng, chỉ giữ image_id/header_image_id làm reference)
    - cached_contributors (trùng với contributions, contributions chi tiết hơn)
    - locked, collection_import_results (admin nội bộ, không phục vụ RAG/DS)
    - editions liên quan (default_audio_edition_id...) — ngoài scope hiện tại
    - book_mappings (mapping sang platform khác, không cần)

Field GIỮ cached_tags + book_characters theo yêu cầu (dù có phần trùng
taggings) — do người dùng chủ động chọn giữ nguyên.
"""

from ingestion.fetchers.base_fetcher import BaseFetcher

BOOKS_QUERY = """
query FetchBooks($limit: Int!, $offset: Int!, $where: books_bool_exp) {
  books(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    title
    subtitle
    description
    headline
    alternative_titles
    book_category_id
    literary_type_id
    release_date
    release_year
    pages
    audio_seconds
    compilation
    is_partial_book
    parent_book_id
    rating
    ratings_count
    reviews_count
    ratings_distribution
    users_count
    users_read_count
    editions_count
    slug
    links
    book_status_id
    canonical_id
    state
    activities_count
    journals_count
    lists_count
    prompts_count
    import_platform_id
    image_id
    header_image_id
    created_at
    updated_at
    contributions {
      id
      author_id
      contribution
    }
    book_characters {
      id
      character_id
      position
      character { id name }
    }
    book_series {
      series_id
      position
      series { id name slug }
    }
    cached_tags
    taggings {
      id
      tag_id
      spoiler
      tag {
        id
        tag
        tag_category_id
      }
    }
  }
}
"""


class BooksFetcher(BaseFetcher):
    entity_name = "books"
    query = BOOKS_QUERY
