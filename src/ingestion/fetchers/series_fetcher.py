"""Fetcher cho entity 'series' (dimension table nhẹ, không phải 'book_series' join table)."""

from ingestion.fetchers.base_fetcher import BaseFetcher

SERIES_QUERY = """
query FetchSeries($limit: Int!, $offset: Int!, $where: series_bool_exp) {
  series(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    name
    slug
    description
    author_id
    books_count
    primary_books_count
    is_completed
    canonical_id
    state
  }
}
"""


class SeriesFetcher(BaseFetcher):
    entity_name = "series"
    query = SERIES_QUERY
    incremental_field = None
