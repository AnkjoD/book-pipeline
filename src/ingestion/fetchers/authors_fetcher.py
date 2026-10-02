"""Fetcher cho entity 'authors'."""

from ingestion.fetchers.base_fetcher import BaseFetcher

AUTHORS_QUERY = """
query FetchAuthors($limit: Int!, $offset: Int!, $where: authors_bool_exp) {
  authors(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    name
    name_personal
    slug
    bio
    title
    image_id
    born_date
    born_year
    death_date
    death_year
    gender_id
    location
    is_bipoc
    is_lgbtq
    links
    alternate_names
    identifiers
    canonical_id
    state
    books_count
    users_count
  }
}
"""


class AuthorsFetcher(BaseFetcher):
    entity_name = "authors"
    query = AUTHORS_QUERY
    incremental_field = None
    # LƯU Ý: schema thật của Authors KHÔNG có created_at/updated_at
    # (khác với Books) -- nên KHÔNG dùng sync_incremental() cho entity này,
    # chỉ chạy fetch_all() (mode=full) định kỳ. Xem docker-compose.yml.
