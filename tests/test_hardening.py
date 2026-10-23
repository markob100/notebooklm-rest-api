"""Checks for the API-key gate and the loud master-token mint failure.

Plain asserts, no pytest needed:  .venv/bin/python tests/test_hardening.py
(pytest also collects it). Talks to nothing outside the process: the master
token is junk, so the mint fails before any network call.
"""
import asyncio
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "test-key-0123456789abcdef"

os.environ.pop("RAILWAY_ENVIRONMENT", None)
os.environ["NOTEBOOKLM_REST_API_KEY"] = KEY
os.environ["NOTEBOOKLM_MASTER_TOKEN_JSON"] = "not a master token"
os.environ["NOTEBOOKLM_HOME"] = tempfile.mkdtemp(prefix="nlm-test-home-")
sys.path.insert(0, ROOT)

import app as sidecar  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(sidecar.app)


def test_health_is_open():
    r = client.get("/health")
    assert r.status_code == 200, r.text


def test_docs_are_off():
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 401, path  # gate first
        assert client.get(path, headers={"X-API-Key": KEY}).status_code == 404, path


def test_every_v1_route_requires_the_key():
    v1_routes = [r for r in sidecar.app.routes if getattr(r, "path", "").startswith("/v1")]
    assert len(v1_routes) >= 20, len(v1_routes)
    for r in v1_routes:
        calls = [d.call for d in r.dependant.dependencies]
        assert sidecar.require_api_key in calls, r.path
    open_routes = {getattr(r, "path", "") for r in sidecar.app.routes
                   if not getattr(r, "path", "").startswith("/v1")}
    assert open_routes == {"/health"}, open_routes


def test_missing_wrong_and_non_ascii_keys_are_401():
    for headers in ({}, {"X-API-Key": "wrong"}, {"X-API-Key": b"\xc3\xa9t\xc3\xa9"}):
        r = client.get("/v1/notebooks", headers=headers)
        assert r.status_code == 401, (headers, r.status_code, r.text)
        assert r.json()["detail"]["code"] == "API_KEY_INVALID", r.text


def test_key_is_checked_before_the_body_is_read():
    # Without the middleware these were 422 (bad JSON) and an upload spooled
    # to disk before the router dependency ran.
    r = client.post("/v1/notebooks", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 401, r.text
    r = client.post("/v1/notebooks/nb/sources/file", files={"file": ("a.txt", b"x" * 1024)})
    assert r.status_code == 401, r.text
    assert r.json()["detail"]["code"] == "API_KEY_INVALID", r.text


def test_unknown_paths_fail_closed():
    assert client.get("/nope").status_code == 401
    assert client.get("/nope", headers={"X-API-Key": KEY}).status_code == 404


def test_mint_log_redacts_token_shaped_text():
    msg = sidecar._redact("bad token aas_et/AKppINaVeryLongSecretValue1234567890 for x")
    assert "AKppIN" not in msg and "[redacted]" in msg, msg
    assert sidecar._redact("Unreadable master_token.json: Expecting value") == \
        "Unreadable master_token.json: Expecting value"


def test_failed_mint_is_503_and_cooled_down():
    calls = {"n": 0}
    real = sidecar._mint_from_master_token

    async def counting():
        calls["n"] += 1
        return await real()

    sidecar._mint_from_master_token = counting
    logged = []
    real_error = sidecar.log.error
    sidecar.log.error = lambda msg, *a, **k: (logged.append(msg), real_error(msg, *a, **k))
    try:
        sidecar._mint_failed_at = 0.0
        for _ in range(2):
            r = client.get("/v1/notebooks", headers={"X-API-Key": KEY})
            assert r.status_code == 503, r.text
            assert r.json()["detail"]["code"] == "AUTH_MINT_FAILED", r.text
        # The second request fell inside MINT_RETRY_SECONDS: no second attempt.
        assert calls["n"] == 1, calls
        # It failed on the junk RECORD, not on a missing library name: an
        # ImportError here means the venv predates the pin (or a bump moved
        # the private master-token helpers) and the test proves nothing.
        assert logged and all("ImportError" not in m and "ModuleNotFound" not in m for m in logged), logged
    finally:
        sidecar._mint_from_master_token = real
        sidecar.log.error = real_error


def test_refuses_to_boot_on_railway_without_a_key():
    env = {k: v for k, v in os.environ.items() if k != "NOTEBOOKLM_REST_API_KEY"}
    env["RAILWAY_ENVIRONMENT"] = "production"
    r = subprocess.run([sys.executable, "-c", "import app"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "NOTEBOOKLM_REST_API_KEY is not set" in r.stderr, r.stderr


def test_no_master_token_on_railway_is_a_loud_503():
    env = {k: v for k, v in os.environ.items() if k != "NOTEBOOKLM_MASTER_TOKEN_JSON"}
    env["RAILWAY_ENVIRONMENT"] = "production"
    code = (
        "from fastapi.testclient import TestClient; import app; "
        "r = TestClient(app.app).get('/v1/notebooks', headers={'X-API-Key': %r}); "
        "print(r.status_code, r.json()['detail']['code'])" % KEY
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "503 AUTH_MINT_FAILED", r.stdout
    assert "[auth] NLM_MINT_FAILED: NOTEBOOKLM_MASTER_TOKEN_JSON is not set" in r.stderr, r.stderr


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
