"""Auth smoke test.

Two modes, run automatically:

1. OFFLINE (always runs): generate a throwaway RSA key, sign a sample request
   with KalshiAuth, and verify the signature with the matching public key. This
   proves the RSA-PSS signing in auth.py is internally correct without needing
   Kalshi credentials.

2. LIVE (runs only if credentials are present): hit the demo API's public
   exchange-status endpoint and the authenticated balance endpoint to prove the
   full request path + headers are accepted by Kalshi.

Credentials for the live check (env vars take precedence over config.yaml):
    KALSHI_KEY_ID             your API Key ID
    KALSHI_PRIVATE_KEY_PATH   path to the RSA private-key PEM
    KALSHI_BASE_URL           optional; defaults to the demo base URL

Run:  python scripts/test_auth.py
"""

from __future__ import annotations

import base64
import os
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from hedge.kalshi import KalshiAuth, KalshiClient, load_private_key
from hedge.kalshi.client import DEMO_BASE


def offline_check() -> bool:
    print("[offline] generating throwaway RSA key + signing a sample request...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    auth = KalshiAuth(key_id="test-key-id", private_key=key)

    ts = "1718000000000"
    method = "GET"
    path = "/trade-api/v2/portfolio/balance"
    sig_b64 = auth.sign(ts, method, path)

    # Verify exactly as Kalshi would: same message, same PSS params.
    message = (ts + method + path).encode("utf-8")
    try:
        key.public_key().verify(
            base64.b64decode(sig_b64),
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[offline] FAIL: signature did not verify: {e}")
        return False

    # Sanity: query string must be stripped before signing. PSS uses a random
    # salt, so signatures aren't byte-equal across calls; instead verify that a
    # signature over path+query validates against the stripped message.
    sig_with_query = auth.sign(ts, method, path + "?limit=5")
    try:
        key.public_key().verify(
            base64.b64decode(sig_with_query),
            message,  # the STRIPPED message
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[offline] FAIL: query string was not stripped before signing: {e}")
        return False

    hdrs = auth.headers(method, path)
    assert set(hdrs) == {
        "KALSHI-ACCESS-KEY",
        "KALSHI-ACCESS-TIMESTAMP",
        "KALSHI-ACCESS-SIGNATURE",
    }, f"unexpected headers: {sorted(hdrs)}"

    print("[offline] PASS: signature verifies, query stripped, headers correct.")
    return True


def live_check() -> bool | None:
    key_id = os.environ.get("KALSHI_KEY_ID")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    base_url = os.environ.get("KALSHI_BASE_URL", DEMO_BASE)

    if not key_id or not key_path:
        print(
            "[live]  SKIP: set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH to run "
            "the live demo-API check."
        )
        return None
    if not os.path.exists(key_path):
        print(f"[live]  SKIP: private key not found at {key_path}")
        return None

    print(f"[live]  hitting {base_url} ...")
    auth = KalshiAuth(key_id=key_id, private_key=load_private_key(key_path))
    client = KalshiClient(auth, base_url=base_url)

    try:
        status = client.get_exchange_status()
        print(f"[live]  exchange status OK: {status}")
        balance = client.get_balance()
        print(f"[live]  authenticated balance OK: {balance}")
    except Exception as e:  # noqa: BLE001
        print(f"[live]  FAIL: {e}")
        return False

    print("[live]  PASS: Kalshi accepted the signed requests.")
    return True


def main() -> int:
    ok_offline = offline_check()
    print()
    ok_live = live_check()

    if not ok_offline:
        return 1
    if ok_live is False:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
