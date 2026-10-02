"""
Fetcher cho các reference/lookup table siêu nhẹ: countries, languages,
reading_formats và platforms.
Đây là các bảng tĩnh, số lượng record rất nhỏ (vài chục - vài trăm dòng),
gộp chung 1 file cho gọn thay vì tách 3 file riêng.
"""

from ingestion.fetchers.base_fetcher import BaseFetcher

COUNTRIES_QUERY = """
query FetchCountries($limit: Int!, $offset: Int!, $where: countries_bool_exp) {
  countries(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    name
    code2
    code3
    iso_3166
    region
    region_code
    sub_region
    sub_region_code
  }
}
"""

LANGUAGES_QUERY = """
query FetchLanguages($limit: Int!, $offset: Int!, $where: languages_bool_exp) {
  languages(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    language
    code2
    code3
  }
}
"""

PLATFORMS_QUERY = """
query FetchPlatforms($limit: Int!, $offset: Int!, $where: platforms_bool_exp) {
  platforms(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    name
    url
  }
}
"""

READING_FORMATS_QUERY = """
query FetchReadingFormats($limit: Int!, $offset: Int!, $where: reading_formats_bool_exp) {
  reading_formats(limit: $limit, offset: $offset, order_by: { id: asc }, where: $where) {
    id
    format
  }
}
"""


class CountriesFetcher(BaseFetcher):
    entity_name = "countries"
    query = COUNTRIES_QUERY
    incremental_field = None


class LanguagesFetcher(BaseFetcher):
    entity_name = "languages"
    query = LANGUAGES_QUERY
    incremental_field = None


class ReadingFormatsFetcher(BaseFetcher):
    entity_name = "reading_formats"
    query = READING_FORMATS_QUERY
    incremental_field = None
    supports_count = False


class PlatformsFetcher(BaseFetcher):
    entity_name = "platforms"
    query = PLATFORMS_QUERY
    incremental_field = None
