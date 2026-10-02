"""Fetcher cho entity 'tags' (dimension table nhẹ, bổ sung mô tả/category cho tag_id đã có trong taggings)."""

from ingestion.fetchers.base_fetcher import BaseFetcher

TAGS_QUERY = """
query FetchTags($limit: Int!, $offset: Int!, $where: tags_bool_exp) {
  tags(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    tag
    slug
    tag_category_id
    count
    tag_category {
    id
    category
    slug
  }
  }
}
"""


class TagsFetcher(BaseFetcher):
    entity_name = "tags"
    query = TAGS_QUERY
    incremental_field = None
