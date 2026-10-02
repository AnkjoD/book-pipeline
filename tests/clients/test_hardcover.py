from clients.hardcover import HardcoverClient


def test_client_accepts_raw_token():
    client = HardcoverClient("abc123")
    assert client.session.headers["Authorization"] == "Bearer abc123"


def test_client_does_not_duplicate_bearer_prefix():
    client = HardcoverClient("Bearer abc123")
    assert client.session.headers["Authorization"] == "Bearer abc123"
