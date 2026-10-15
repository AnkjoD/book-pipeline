"""
HardcoverClient — GraphQL client cho Hardcover API.
Tự retry với exponential backoff khi gặp rate limit (429) hoặc lỗi mạng/server tạm thời.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)


class HardcoverClient:
    """Client gọi GraphQL API của Hardcover, có retry tự động."""

    ENDPOINT = "https://api.hardcover.app/v1/graphql"

    def __init__(self, api_key: str, max_retries: int = 5, timeout: int = 30):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        token = api_key.strip()
        authorization = token if token.lower().startswith("bearer ") else f"Bearer {token}"
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": authorization,
            }
        )

    def execute_query(self, query: str, variables: dict | None = None) -> dict:
        """
        Gọi 1 GraphQL query, trả về toàn bộ response body (bao gồm key "data").
        Tự retry tối đa max_retries lần với exponential backoff khi gặp
        rate limit (429), lỗi server (5xx), hoặc mất mạng.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                resp = self.session.post(self.ENDPOINT, json=payload, timeout=self.timeout)
                elapsed = time.time() - start

                if resp.status_code == 429:
                    last_error = RuntimeError("Hardcover API rate limit (429)")
                    wait = min(2**attempt, 60)
                    logger.warning(
                        f"Rate limit (429) — chờ {wait}s (lần {attempt}/{self.max_retries})"
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    last_error = RuntimeError(f"Hardcover API server error ({resp.status_code})")
                    wait = min(2**attempt, 60)
                    logger.warning(
                        f"Lỗi server {resp.status_code} — chờ {wait}s "
                        f"(lần {attempt}/{self.max_retries})"
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait)
                    continue

                resp.raise_for_status()
                body = resp.json()

                if "errors" in body:
                    logger.error(f"GraphQL error: {body['errors']}")
                    raise RuntimeError(f"GraphQL error: {body['errors']}")

                logger.debug(f"Query OK — {elapsed:.2f}s")
                return body

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                wait = min(2**attempt, 60)
                logger.warning(f"Lỗi mạng: {e} — chờ {wait}s (lần {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    time.sleep(wait)

        raise RuntimeError(
            f"Query thất bại sau {self.max_retries} lần retry. Lỗi cuối: {last_error}"
        )

    def count(self, entity: str, where: dict | None = None) -> int:
        """
        Đếm tổng số record của 1 entity — dùng để tính % tiến độ và ETA trước khi fetch.
        """
        query = f"""
        query Count($where: {entity}_bool_exp) {{
          {entity}_aggregate(where: $where) {{
            aggregate {{ count }}
          }}
        }}
        """
        variables = {"where": where} if where else {}
        result = self.execute_query(query, variables)
        return result["data"][f"{entity}_aggregate"]["aggregate"]["count"]
