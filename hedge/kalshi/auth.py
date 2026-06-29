"""Kalshi API-key authentication: RSA-PSS request signing.

Kalshi authenticates every request with three headers:

    KALSHI-ACCESS-KEY        your API Key ID (a UUID)
    KALSHI-ACCESS-TIMESTAMP  current Unix time in MILLISECONDS, as a string
    KALSHI-ACCESS-SIGNATURE  base64(RSA-PSS sign(timestamp + METHOD + path))

The signed message is the concatenation, with NO separators, of:
    timestamp_ms + HTTP_METHOD + request_path
where:
  * HTTP_METHOD is uppercase ("GET", "POST", "DELETE").
  * request_path INCLUDES the "/trade-api/v2" prefix but EXCLUDES the query
    string. e.g. for GET /trade-api/v2/portfolio/orders?limit=5 you sign
    "/trade-api/v2/portfolio/orders".

The signature uses RSA-PSS with SHA-256, MGF1-SHA256, and salt length equal to
the digest length (32 bytes). Getting the salt length wrong is the single most
common cause of Kalshi rejecting otherwise-correct signatures.
"""

from __future__ import annotations

import base64
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def load_private_key(pem_path: str, password: bytes | None = None) -> rsa.RSAPrivateKey:
    """Load an RSA private key from a PKCS#8 PEM file.

    This is the key Kalshi shows you exactly once when you create an API key.
    """
    with open(pem_path, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=password)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise TypeError(
            f"{pem_path} is not an RSA private key (got {type(key).__name__})"
        )
    return key


def _strip_query(path: str) -> str:
    """Return the path without its query string (Kalshi signs path only)."""
    return path.split("?", 1)[0]


class KalshiAuth:
    """Signs requests with a Kalshi API key + RSA private key.

    Example::

        key = load_private_key("secrets/kalshi_private_key.pem")
        auth = KalshiAuth(key_id="...", private_key=key)
        headers = auth.headers("GET", "/trade-api/v2/portfolio/balance")
    """

    def __init__(self, key_id: str, private_key: rsa.RSAPrivateKey):
        self.key_id = key_id
        self.private_key = private_key

    def sign(self, timestamp_ms: str, method: str, path: str) -> str:
        """Return the base64 RSA-PSS signature for one request."""
        message = (timestamp_ms + method.upper() + _strip_query(path)).encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,  # 32 bytes — must be this
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("ascii")

    def headers(self, method: str, path: str) -> dict[str, str]:
        """Build the three auth headers for a request.

        `path` should be the full request path including "/trade-api/v2"; any
        query string is stripped automatically before signing.
        """
        timestamp_ms = str(int(time.time() * 1000))
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": self.sign(timestamp_ms, method, path),
        }
