"""Fetcher cho entity 'publishers' (dimension table nhẹ, đi kèm editions.publisher_id)."""

from ingestion.fetchers.base_fetcher import BaseFetcher

PUBLISHERS_QUERY = """
query FetchPublishers($limit: Int!, $offset: Int!, $where: publishers_bool_exp) {
  publishers(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    name
    slug
    parent_id
    editions_count
    state
    canonical_id
    created_at
    updated_at
  }
}
"""


class PublishersFetcher(BaseFetcher):
    entity_name = "publishers"
    query = PUBLISHERS_QUERY
