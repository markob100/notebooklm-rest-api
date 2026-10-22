# Google NotebookLM REST API wrapper
# Namhyeon Go <gnh1201@catswords.re.kr>
# https://github.com/gnh1201/notebooklm-rest-api
import asyncio
import hmac
import logging
import os
import re
import sys
import time
import uuid
import tempfile
from typing import Any, Optional, Literal, Dict

from fastapi import APIRouter, FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from notebooklm import (
    NotebookLMClient,
    RPCError,
    AuthError,
    RateLimitError,
    NetworkError,
    ServerError,
    ClientError,
    NotebookLMError,
)  # notebooklm-py :contentReference[oaicite:2]{index=2}

# Added in notebooklm-py 0.8.0: wait_for_completion / download waits raise a
# typed timeout instead of returning a pending status. Base class covers both
# ArtifactPendingTimeoutError and ArtifactInProgressTimeoutError. Guarded so the
# module still imports against older pins.
try:
    from notebooklm.exceptions import ArtifactTimeoutError
except ImportError:  # pragma: no cover - older notebooklm-py
    class ArtifactTimeoutError(Exception):
        """Placeholder when the installed notebooklm-py predates 0.8.0."""

# Enum imports for string → enum conversion in artifact generation
try:
    from notebooklm.enums import (
        InfographicStyle, InfographicOrientation, InfographicDetail,
        AudioFormat, AudioLength,
        VideoFormat, VideoStyle,
        SlideDeckFormat, SlideDeckLength,
        ReportFormat,
        QuizQuantity, QuizDifficulty,
    )
    _ENUMS_AVAILABLE = True
except ImportError:
    # Fallback: try importing from top-level module
    try:
        from notebooklm import (
            InfographicStyle, InfographicOrientation, InfographicDetail,
            AudioFormat, AudioLength,
            VideoFormat, VideoStyle,
            SlideDeckFormat, SlideDeckLength,
            ReportFormat,
            QuizQuantity, QuizDifficulty,
        )
        _ENUMS_AVAILABLE = True
    except ImportError:
        _ENUMS_AVAILABLE = False


# ----------------------------
# Config / Security
# ----------------------------
# Own handlers rather than the root logger's: uvicorn configures only its own
# loggers, and without a handler an INFO line here would be dropped. Warnings
# and errors go to stderr, which Railway shows at error level; the rest stays
# on stdout.
log = logging.getLogger("notebooklm-rest-api")
if not log.handlers:
    _fmt = logging.Formatter("%(levelname)s %(message)s")
    _out = logging.StreamHandler(sys.stdout)
    _out.addFilter(lambda r: r.levelno < logging.WARNING)
    _out.setFormatter(_fmt)
    _err = logging.StreamHandler(sys.stderr)
    _err.setLevel(logging.WARNING)
    _err.setFormatter(_fmt)
    log.addHandler(_out)
    log.addHandler(_err)
    log.setLevel(logging.INFO)
    log.propagate = False

# Shared secret every /v1 caller must send as X-API-Key. On Railway the service
# refuses to boot without it: this process holds a full-account Google
# credential, so "reachable on the private network" is not enough on its own.
API_KEY = os.environ.get("NOTEBOOKLM_REST_API_KEY", "")
if not API_KEY:
    if os.environ.get("RAILWAY_ENVIRONMENT"):
        raise RuntimeError(
            "NOTEBOOKLM_REST_API_KEY is not set; refusing to start on Railway without an API key"
        )
    log.warning("[auth] NOTEBOOKLM_REST_API_KEY is not set; /v1 routes are OPEN (local development only)")
AUTH_STORAGE_PATH = os.environ.get("NOTEBOOKLM_STORAGE_PATH")  # optional override

# Contents of master_token.json — {version, email, master_token, android_id} —
# produced once by `notebooklm login --master-token`. notebooklm-py reads that
# record from a file only; there is no env var for it upstream, so we carry it
# in one of ours and materialise it at boot.
#
# Why bother: NOTEBOOKLM_AUTH_JSON pins a *snapshot* of a cookie jar, and
# __Secure-1PSIDTS rotates on use — Google issues a new value and invalidates
# the old one. So the snapshot dies whenever the same account is exercised
# anywhere else, which is exactly how prod broke on 5 Aug. A master token is
# durable and re-mints a fresh jar on demand over plain HTTP (gpsoauth), with
# no browser, so the sidecar can heal itself instead of waiting for a re-push.
MASTER_TOKEN_JSON = os.environ.get("NOTEBOOKLM_MASTER_TOKEN_JSON")

_minted = False  # a jar has been minted from the master token in this process
_mint_failed_at = 0.0  # monotonic time of the last failed mint; 0 = none since the last success

# A failing mint is retried at most this often. Every attempt is a gpsoauth
# exchange of a full-account credential, and hammering Google with a bad one
# is a good way to get the account challenged.
MINT_RETRY_SECONDS = 60


_KEY_REJECTED = {
    "code": "API_KEY_INVALID",
    "message": "Missing or wrong X-API-Key for the NotebookLM sidecar",
    "recoverable": False,
}


def api_key_ok(x_api_key: Optional[str]) -> bool:
    """True when the key matches, or when no key is configured (local dev only).

    Constant time, as bytes: compare_digest on str raises for non-ASCII input,
    which would turn a junk header into a 500.
    """
    if not API_KEY:
        return True
    return x_api_key is not None and hmac.compare_digest(
        x_api_key.encode("utf-8"), API_KEY.encode("utf-8")
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """Router-level copy of the gate the middleware applies (see _api_key_gate).

    The rejection carries its own code because a 401 from this service already
    means "Google rejected the session" (AUTH_EXPIRED / AUTH_INVALID); a caller
    debugging cookies over what is really a key mismatch would waste an hour.
    """
    if not api_key_ok(x_api_key):
        raise HTTPException(status_code=401, detail=_KEY_REJECTED)


# Shared client, lazily built on first use and reused for the process lifetime.
#
# notebooklm-py 0.8.0 made ``NotebookLMClient.from_storage()`` an async context
# manager: the bare ``await`` form still returns an object, but it is NOT
# initialized and every call raises "Client not initialized. Use 'async with'
# context." We therefore enter the context manager explicitly and hold the
# entered client, rather than re-creating (and leaking) one per request.
#
# A single shared client also matches how this sidecar is driven: reg-monitor-app
# caps NotebookLM at 1 concurrent job because auth is session-based, so there is
# no concurrency win from per-request clients — only extra sockets.
_client: Optional[NotebookLMClient] = None
_client_cm: Any = None
_client_lock = asyncio.Lock()


def _redact(msg: str) -> str:
    """Bound an exception message and blank anything token-shaped in it.

    The mint handles a master token and mints cookies; a library error that
    echoed either must not reach Railway logs. Diagnostic words survive
    ("Unreadable master_token.json: Expecting value: line 1 column 1").
    """
    return re.sub(r"[A-Za-z0-9_\-./+=:%]{20,}", "[redacted]", msg)[:200]


async def _mint_from_master_token() -> bool:
    """Mint a fresh cookie jar from the master token and make it the active auth.

    Returns True when the profile now holds a freshly-minted jar. A failure
    never kills the process, but it is loud: an error line starting
    ``[auth] NLM_MINT_FAILED`` (grep for that prefix), and get_client() answers
    503 AUTH_MINT_FAILED instead of building a client. It used to fall back to
    NOTEBOOKLM_AUTH_JSON quietly; since the 22 Oct 2026 rotation that variable
    is a placeholder jar, so the fallback could only surface later as a
    confusing Google-side auth error.
    """
    global _minted, _mint_failed_at
    if not MASTER_TOKEN_JSON:
        return False
    try:
        from notebooklm.paths import get_master_token_path, get_storage_path
        from notebooklm._auth.master_token import (
            read_master_token, mint_cookies, persist_minted_jar,
        )

        token_path = get_master_token_path()
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(MASTER_TOKEN_JSON, encoding="utf-8")
        os.chmod(token_path, 0o600)

        record = read_master_token(token_path)
        if not record:
            log.error("[auth] NLM_MINT_FAILED: NOTEBOOKLM_MASTER_TOKEN_JSON is not a readable master-token record")
            _mint_failed_at = time.monotonic()
            return False

        # mint_cookies is itself a coroutine — it off-threads the blocking
        # gpsoauth call internally. Wrapping it in asyncio.to_thread yields a
        # coroutine object rather than a jar, which fails later and confusingly.
        jar = await mint_cookies(
            record["email"], record["master_token"], record["android_id"]
        )
        storage_path = get_storage_path()
        # notebooklm-py 0.8.1 added ``refuse_unknown_owner`` (default True): it
        # refuses to replace an existing storage_state.json that carries NO
        # recorded owner. This sidecar is single-account and its storage is
        # written only by this mint (owner recorded) or by an operator-supplied
        # jar (no owner), so the refusal could only ever demote us to the stale
        # env snapshot — the exact failure the master token exists to heal.
        # Pass False, as the library's own remint path does; ``force`` stays
        # off so a jar recorded to a DIFFERENT account is still refused.
        try:
            persist_minted_jar(
                storage_path, jar, email=record["email"], refuse_unknown_owner=False
            )
        except TypeError:  # pragma: no cover - notebooklm-py < 0.8.1 lacks the kwarg
            persist_minted_jar(storage_path, jar, email=record["email"])
        os.chmod(storage_path, 0o600)

        # The env jar takes precedence over the profile everywhere in 0.8.0, so
        # a stale snapshot would shadow what we just minted. Drop it in-process
        # only.
        os.environ.pop("NOTEBOOKLM_AUTH_JSON", None)
        _minted = True
        _mint_failed_at = 0.0
        log.info(f"[auth] minted a fresh cookie jar from the master token ({len(jar)} cookies)")
        return True
    except Exception as e:  # noqa: BLE001 - never let auth setup kill the process
        # Type and a bounded message only: the record itself never goes to a log.
        log.error(f"[auth] NLM_MINT_FAILED: {type(e).__name__}: {_redact(str(e))}")
        _mint_failed_at = time.monotonic()
        return False


async def _close_client() -> None:
    """Exit the shared client's context manager, ignoring teardown errors."""
    global _client, _client_cm
    cm, _client, _client_cm = _client_cm, None, None
    if cm is not None:
        try:
            await cm.__aexit__(None, None, None)
        except Exception:
            pass


async def reset_client() -> None:
    """Drop the shared client so the next request rebuilds it.

    Called when auth fails mid-flight: the cookies backing the live client may
    have been rotated underneath us (Railway var update + redeploy, or an
    operator re-running ``notebooklm login``), and a rebuild picks them up
    without a process restart.

    Also clears the minted flag, so a master-token deployment re-mints on the
    next request rather than rebuilding a client around the same dead jar.
    """
    global _minted
    async with _client_lock:
        await _close_client()
        _minted = False


async def get_client() -> NotebookLMClient:
    """Return the shared NotebookLMClient, building it on first use.

    Auth precedence (delegated to notebooklm-py):
      - explicit path (``AUTH_STORAGE_PATH`` env → ``NOTEBOOKLM_STORAGE_PATH``)
      - ``NOTEBOOKLM_AUTH_JSON``
      - ``NOTEBOOKLM_HOME/storage_state.json``
      - ``~/.notebooklm/storage_state.json``

    The client is built lazily rather than at startup so that bad or missing
    credentials surface as a 401/500 on the failing request instead of
    preventing the service from booting (and failing its healthcheck).

    BYO-cookies (per-request user sessions) is deferred — see reg-monitor-app
    BACKLOG item #64.
    """
    global _client, _client_cm
    if _client is not None:
        return _client
    async with _client_lock:
        # Re-check under the lock: a concurrent caller may have built it.
        if _client is not None:
            return _client
        # Mint before building. On a cold start this replaces whatever stale
        # snapshot is lying around; after an auth failure reset_client() has
        # cleared the flag, so this is also the self-heal path.
        #
        # A configured master token that will not mint is the whole story, so
        # say so (503, retryable) rather than build a client around whatever
        # else is on disk and let Google's rejection arrive later as a 401 that
        # looks like expired cookies. A recent failure answers without another
        # attempt (MINT_RETRY_SECONDS).
        if MASTER_TOKEN_JSON and not _minted:
            recently_failed = (
                _mint_failed_at and time.monotonic() - _mint_failed_at < MINT_RETRY_SECONDS
            )
            if recently_failed or not await _mint_from_master_token():
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "AUTH_MINT_FAILED",
                        "message": "Could not mint Google cookies from NOTEBOOKLM_MASTER_TOKEN_JSON; "
                                   "see the sidecar log lines starting [auth] NLM_MINT_FAILED",
                        "recoverable": True,
                    },
                )
        try:
            cm = (
                NotebookLMClient.from_storage(AUTH_STORAGE_PATH)
                if AUTH_STORAGE_PATH
                else NotebookLMClient.from_storage()
            )
            client = await cm.__aenter__()
            _client_cm, _client = cm, client
            return client
        except ValueError as e:
            await _close_client()
            msg = str(e)
            code = "AUTH_INVALID"
            if "expired" in msg.lower() or "redirected" in msg.lower():
                code = "AUTH_EXPIRED"
            raise HTTPException(
                status_code=401,
                detail={"code": code, "message": msg, "recoverable": True},
            )
        except FileNotFoundError as e:
            await _close_client()
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "AUTH_NOT_CONFIGURED",
                    "message": str(e),
                    "recoverable": False,
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            await _close_client()
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "INIT_FAILED",
                    "message": f"Failed to initialize NotebookLM client: {e}",
                    "recoverable": False,
                },
            )


def map_notebooklm_error(e: Exception) -> HTTPException:
    """Map notebooklm-py exceptions to structured HTTPException responses.

    Returns a JSON ``detail`` body with:
      - ``code``: one of AUTH_EXPIRED / AUTH_INVALID / RATE_LIMITED /
        SERVER_ERROR / CLIENT_ERROR / NETWORK / RPC_ERROR
      - ``message``: human-readable message
      - ``recoverable``: whether a retry (or re-auth for AUTH_*) is sensible
      - ``retry_after``: seconds to wait before retry (RATE_LIMITED only)
      - ``status_code``: upstream HTTP status where available

    Status codes:
      - 401 on AuthError
      - 429 on RateLimitError (with Retry-After surfaced)
      - 502 on ServerError / generic RPCError (upstream is unhealthy)
      - 504 on NetworkError / timeout (we couldn't reach upstream)
      - 400 on ClientError (bad input to upstream)
    """
    if isinstance(e, AuthError):
        msg = str(e)
        # The client is now shared for the process lifetime, so a session that
        # has gone stale would otherwise poison every subsequent request. Drop
        # it so the next call rebuilds from whatever credentials are current.
        # Scheduled rather than awaited: this mapper is sync and is called from
        # except blocks that cannot themselves await.
        try:
            asyncio.get_running_loop().create_task(reset_client())
        except RuntimeError:
            pass
        # AuthError.recoverable is a class attr; default False but instances
        # may override. We treat all as recoverable with re-auth.
        code = "AUTH_EXPIRED" if getattr(e, "recoverable", False) else "AUTH_INVALID"
        # Fallback on message content when recoverable attr isn't set
        if "expired" in msg.lower():
            code = "AUTH_EXPIRED"
        return HTTPException(
            status_code=401,
            detail={
                "code": code,
                "message": msg,
                "recoverable": True,
                "method_id": getattr(e, "method_id", None),
            },
        )
    if isinstance(e, RateLimitError):
        detail = {
            "code": "RATE_LIMITED",
            "message": str(e),
            "recoverable": True,
            "retry_after": getattr(e, "retry_after", None),
            "method_id": getattr(e, "method_id", None),
        }
        headers = {}
        if detail["retry_after"] is not None:
            headers["Retry-After"] = str(detail["retry_after"])
        return HTTPException(status_code=429, detail=detail, headers=headers or None)
    if isinstance(e, ServerError):
        return HTTPException(
            status_code=502,
            detail={
                "code": "SERVER_ERROR",
                "message": str(e),
                "recoverable": True,
                "upstream_status": getattr(e, "status_code", None),
                "method_id": getattr(e, "method_id", None),
            },
        )
    if isinstance(e, ClientError):
        return HTTPException(
            status_code=400,
            detail={
                "code": "CLIENT_ERROR",
                "message": str(e),
                "recoverable": False,
                "upstream_status": getattr(e, "status_code", None),
                "method_id": getattr(e, "method_id", None),
            },
        )
    if isinstance(e, NetworkError):
        return HTTPException(
            status_code=504,
            detail={
                "code": "NETWORK",
                "message": str(e),
                "recoverable": True,
                "method_id": getattr(e, "method_id", None),
            },
        )
    if isinstance(e, RPCError):
        # Any other RPC error — generic upstream failure
        return HTTPException(
            status_code=502,
            detail={
                "code": "RPC_ERROR",
                "message": str(e),
                "recoverable": False,
                "method_id": getattr(e, "method_id", None),
            },
        )
    if isinstance(e, NotebookLMError):
        return HTTPException(
            status_code=502,
            detail={
                "code": "NOTEBOOKLM_ERROR",
                "message": str(e),
                "recoverable": False,
            },
        )
    # Unknown — surface generically
    return HTTPException(
        status_code=500,
        detail={
            "code": "UNKNOWN",
            "message": str(e),
            "recoverable": False,
        },
    )


# Backward-compat shim: existing call sites use `map_rpc_error(e)`. Keep
# that spelling working and route through the typed mapper so all call
# sites benefit without touching every endpoint.
def map_rpc_error(e: Exception) -> HTTPException:  # noqa: D401 — back-compat
    """Backward-compatible alias for map_notebooklm_error()."""
    return map_notebooklm_error(e)


# ----------------------------
# String → Enum conversion
# ----------------------------
# Maps (artifact_type, option_key) → enum class.
# notebooklm-py expects enum instances (e.g. InfographicStyle.LANDSCAPE),
# but the REST API receives plain strings ("LANDSCAPE").
_ENUM_MAP: Dict[str, Dict[str, Any]] = {}
if _ENUMS_AVAILABLE:
    _ENUM_MAP = {
        "infographic": {
            "style": InfographicStyle,
            "orientation": InfographicOrientation,
            "detail_level": InfographicDetail,
        },
        "audio": {
            "audio_format": AudioFormat,
            "audio_length": AudioLength,
        },
        "video": {
            "video_format": VideoFormat,
            "video_style": VideoStyle,
        },
        "video-explainer": {
            "video_format": VideoFormat,
            "video_style": VideoStyle,
        },
        "slide_deck": {
            "slide_format": SlideDeckFormat,
            "slide_length": SlideDeckLength,
        },
        "slide-detailed": {
            "slide_format": SlideDeckFormat,
            "slide_length": SlideDeckLength,
        },
        "report": {
            "report_format": ReportFormat,
        },
        "quiz": {
            "quantity": QuizQuantity,
            "difficulty": QuizDifficulty,
        },
        "flashcards": {
            "quantity": QuizQuantity,
            "difficulty": QuizDifficulty,
        },
    }


def _convert_enums(artifact_type: str, opts: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string option values to enum instances where applicable.

    Also normalises option keys that differ between the UI and notebooklm-py:
      - ``targetLanguage`` (camelCase, used across the app's export UI to
        align with ElevenLabs / internal artefacts) is renamed to
        ``language``, which every ``generate_*`` method in notebooklm-py
        accepts as a strict kwarg (``language: str = "en"``). Without this
        rename the call would 500 with ``TypeError: unexpected keyword
        argument 'targetLanguage'``.
    """
    # Key normalisation — run before enum conversion so camelCase doesn't
    # mask an enum mapping. Currently only targetLanguage needs renaming.
    if "targetLanguage" in opts:
        opts = {**opts, "language": opts["targetLanguage"]}
        opts.pop("targetLanguage", None)

    type_map = _ENUM_MAP.get(artifact_type, {})
    if not type_map:
        return opts
    converted = {}
    for key, value in opts.items():
        enum_cls = type_map.get(key)
        if enum_cls is not None and isinstance(value, str):
            try:
                converted[key] = enum_cls[value]  # lookup by name, e.g. InfographicStyle["LANDSCAPE"]
            except KeyError:
                # Try case-insensitive match
                upper = value.upper()
                try:
                    converted[key] = enum_cls[upper]
                except KeyError:
                    # Pass through as-is — library will raise its own error
                    converted[key] = value
        else:
            converted[key] = value
    return converted


# ----------------------------
# Models
# ----------------------------
class NotebookCreateReq(BaseModel):
    title: str


class NotebookRenameReq(BaseModel):
    new_title: str


class SourceAddUrlReq(BaseModel):
    url: str
    wait: bool = True


class SourceAddTextReq(BaseModel):
    title: str
    content: str


class SourceAddYoutubeReq(BaseModel):
    url: str
    wait: bool = True


class ChatAskReq(BaseModel):
    question: str
    # optional persona fields could be added if you want


class ArtifactGenerateReq(BaseModel):
    # A simple unified generator:
    # audio/video/report/quiz/flashcards/slide_deck/infographic/data_table/mind_map
    type: Literal[
        "audio",
        "video",
        "video-explainer",
        "cinematic-video",
        "report",
        "quiz",
        "flashcards",
        "slide_deck",
        "slide-detailed",
        "infographic",
        "data_table",
        "mind_map",
    ]
    # Options are passed through as-is to the underlying generate_* calls where applicable.
    # (The library supports many per-type options; keep this generic.)
    options: Dict[str, Any] = {}


class TaskPollResp(BaseModel):
    ok: bool
    status: Any


# ----------------------------
# App
# ----------------------------
# No /docs, /redoc or /openapi.json: they would sit outside the key gate and
# publish a map of every route.
app = FastAPI(
    title="NotebookLM REST API (powered by notebooklm-py)",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Every /v1 route lives on this router and so needs X-API-Key; only /health,
# on the app itself, is open. Routes added with @app instead would be open, so
# don't. The router is mounted at the bottom of this file, once its routes exist.
v1 = APIRouter(dependencies=[Depends(require_api_key)])


class _ApiKeyGate:
    """Check the key before anything reads the body.

    FastAPI parses a route's body (JSON, multipart uploads spooled to disk)
    BEFORE it resolves dependencies, so the router dependency alone would let a
    caller without the key make us spool an upload, or get a 422 instead of a
    401. Every path but /health is gated here, so an unknown path fails closed
    too; the router dependency stays as a second line. Pure ASGI rather than
    @app.middleware("http"), so large artefact downloads stream untouched.
    """

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") != "/health":
            key = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"x-api-key"), None)
            if not api_key_ok(key):
                await JSONResponse(status_code=401, content={"detail": _KEY_REJECTED})(scope, receive, send)
                return
        await self.inner(scope, receive, send)


app.add_middleware(_ApiKeyGate)


@app.on_event("shutdown")
async def _shutdown_client():
    """Release the shared NotebookLM client's HTTP pool on process shutdown."""
    await _close_client()


@app.get("/health")
async def health():
    return {"ok": True}


# ----------------------------
# Auth
# ----------------------------
@v1.post("/v1/auth/refresh")
async def refresh_auth_endpoint(
    client: NotebookLMClient = Depends(get_client),
):
    """Refresh CSRF/session tokens against the current NotebookLM session.

    Used by long-lived callers (BYO-auth extension, background workers) to
    proactively rotate the short-lived ``SNlM0e`` and ``FdrFJe`` tokens
    before they go stale, without re-authenticating from scratch.

    Auth precedence follows ``get_client``:
      - ``X-NLM-Auth-Cookies`` header (per-request BYO cookies) — refresh
        is scoped to the caller's own session.
      - Shared storage session — refreshes the sidecar's default session.

    Returns:
        ``{"ok": True, "csrf_token": str, "session_id": str}`` on success.

    Error envelope matches ``map_notebooklm_error`` (structured detail with
    ``code`` / ``message`` / ``recoverable``). Typical failure is
    ``AUTH_EXPIRED`` (401) when the underlying cookies themselves are
    stale and a full re-login is needed.
    """
    # NB: no ``async with client`` here. The client is shared for the process
    # lifetime (see get_client), so entering its context manager would close the
    # shared session on exit and break every subsequent request.
    try:
        tokens = await client.refresh_auth()
        return {
            "ok": True,
            "csrf_token": tokens.csrf_token,
            "session_id": tokens.session_id,
        }
    except ValueError as e:
        # refresh_auth() raises ValueError on Google-auth redirect
        # (cookies stale) and on failed regex extraction (page shape
        # changed). Surface both as 401 AUTH_EXPIRED — the caller's
        # remedy is to re-login regardless.
        await reset_client()
        msg = str(e)
        low = msg.lower()
        code = "AUTH_EXPIRED" if ("expired" in low or "re-authenticate" in low) else "AUTH_INVALID"
        raise HTTPException(
            status_code=401,
            detail={
                "code": code,
                "message": msg,
                "recoverable": True,
            },
        )
    except HTTPException:
        # Already a structured HTTPException from get_client — propagate.
        raise
    except Exception as e:
        # Route through the typed mapper so AuthError / RateLimitError /
        # NetworkError etc. all come back with the shared envelope.
        if isinstance(e, AuthError):
            await reset_client()
        raise map_notebooklm_error(e)


# ----------------------------
# Notebooks
# ----------------------------
@v1.get("/v1/notebooks")
async def list_notebooks():
    client = await get_client()
    try:
        nbs = await client.notebooks.list()
        return {"ok": True, "items": [nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__ for nb in nbs]}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.post("/v1/notebooks")
async def create_notebook(req: NotebookCreateReq):
    client = await get_client()
    try:
        nb = await client.notebooks.create(req.title)
        return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.get("/v1/notebooks/{notebook_id}")
async def get_notebook(notebook_id: str):
    client = await get_client()
    try:
        nb = await client.notebooks.get(notebook_id)
        return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.delete("/v1/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    client = await get_client()
    try:
        ok = await client.notebooks.delete(notebook_id)
        return {"ok": True, "deleted": bool(ok)}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.patch("/v1/notebooks/{notebook_id}/rename")
async def rename_notebook(notebook_id: str, req: NotebookRenameReq):
    client = await get_client()
    try:
        nb = await client.notebooks.rename(notebook_id, req.new_title)
        return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.get("/v1/notebooks/{notebook_id}/summary")
async def get_notebook_summary(notebook_id: str):
    client = await get_client()
    try:
        summary = await client.notebooks.get_summary(notebook_id)
        return {"ok": True, "summary": summary}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.get("/v1/notebooks/{notebook_id}/description")
async def get_notebook_description(notebook_id: str):
    client = await get_client()
    try:
        desc = await client.notebooks.get_description(notebook_id)
        return {"ok": True, "description": desc.model_dump() if hasattr(desc, "model_dump") else desc.__dict__}
    except RPCError as e:
        raise map_rpc_error(e)


# ----------------------------
# Sources
# ----------------------------
@v1.get("/v1/notebooks/{notebook_id}/sources")
async def list_sources(notebook_id: str):
    client = await get_client()
    try:
        items = await client.sources.list(notebook_id)
        return {"ok": True, "items": [s.model_dump() if hasattr(s, "model_dump") else s.__dict__ for s in items]}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.post("/v1/notebooks/{notebook_id}/sources/url")
async def add_source_url(notebook_id: str, req: SourceAddUrlReq):
    client = await get_client()
    try:
        src = await client.sources.add_url(notebook_id, req.url, wait=req.wait)
        return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
    except TypeError:
        # some versions may not accept wait=; fall back
        try:
            src = await client.sources.add_url(notebook_id, req.url)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)
    except RPCError as e:
        raise map_rpc_error(e)


@v1.post("/v1/notebooks/{notebook_id}/sources/youtube")
async def add_source_youtube(notebook_id: str, req: SourceAddYoutubeReq):
    client = await get_client()
    try:
        src = await client.sources.add_youtube(notebook_id, req.url, wait=req.wait)
        return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
    except TypeError:
        try:
            src = await client.sources.add_youtube(notebook_id, req.url)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)
    except RPCError as e:
        raise map_rpc_error(e)


@v1.post("/v1/notebooks/{notebook_id}/sources/text")
async def add_source_text(notebook_id: str, req: SourceAddTextReq):
    client = await get_client()
    try:
        src = await client.sources.add_text(notebook_id, req.title, req.content)
        return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.post("/v1/notebooks/{notebook_id}/sources/file")
async def add_source_file(
    notebook_id: str,
    upload: UploadFile = File(...),
    mime_type: Optional[str] = Form(None),
):
    # Save to temp file first
    suffix = os.path.splitext(upload.filename or "")[1] or ".bin"
    tmp_path = os.path.join(tempfile.gettempdir(), f"nb_{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        f.write(await upload.read())

    client = await get_client()
    try:
        src = await client.sources.add_file(notebook_id, tmp_path, mime_type=mime_type)
        return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
    except RPCError as e:
        raise map_rpc_error(e)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@v1.get("/v1/notebooks/{notebook_id}/sources/{source_id}/fulltext")
async def get_source_fulltext(notebook_id: str, source_id: str):
    client = await get_client()
    try:
        ft = await client.sources.get_fulltext(notebook_id, source_id)
        return {"ok": True, "fulltext": ft.model_dump() if hasattr(ft, "model_dump") else ft.__dict__}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.get("/v1/notebooks/{notebook_id}/sources/{source_id}/guide")
async def get_source_guide(notebook_id: str, source_id: str):
    client = await get_client()
    try:
        guide = await client.sources.get_guide(notebook_id, source_id)
        return {"ok": True, "guide": guide}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.delete("/v1/notebooks/{notebook_id}/sources/{source_id}")
async def delete_source(notebook_id: str, source_id: str):
    client = await get_client()
    try:
        ok = await client.sources.delete(notebook_id, source_id)
        return {"ok": True, "deleted": bool(ok)}
    except RPCError as e:
        raise map_rpc_error(e)


# ----------------------------
# Chat
# ----------------------------
@v1.post("/v1/notebooks/{notebook_id}/chat/ask")
async def chat_ask(notebook_id: str, req: ChatAskReq):
    client = await get_client()
    try:
        result = await client.chat.ask(notebook_id, req.question)
        # result.answer is shown in docs :contentReference[oaicite:5]{index=5}
        if hasattr(result, "model_dump"):
            return {"ok": True, "result": result.model_dump()}
        return {"ok": True, "result": getattr(result, "__dict__", {"answer": getattr(result, "answer", None)})}
    except RPCError as e:
        raise map_rpc_error(e)


# ----------------------------
# Artifacts: list / generate / poll / download
# ----------------------------
@v1.get("/v1/notebooks/{notebook_id}/artifacts")
async def list_artifacts(notebook_id: str, type: Optional[str] = None):
    client = await get_client()
    try:
        items = await client.artifacts.list(notebook_id, type=type) if type else await client.artifacts.list(notebook_id)
        return {"ok": True, "items": [a.model_dump() if hasattr(a, "model_dump") else a.__dict__ for a in items]}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.post("/v1/notebooks/{notebook_id}/artifacts/generate")
async def generate_artifact(notebook_id: str, req: ArtifactGenerateReq):
    client = await get_client()
    try:
        t = req.type
        opts = req.options or {}

        # Convert string values to enum instances where the library expects enums
        opts = _convert_enums(t, opts)

        if t == "audio":
            status = await client.artifacts.generate_audio(notebook_id, **opts)
        elif t == "cinematic-video":
            # Forward language alongside instructions so non-English
            # dashboards still produce a localised cinematic video.
            cine_kwargs = {"instructions": opts.get("instructions")}
            if "language" in opts:
                cine_kwargs["language"] = opts["language"]
            status = await client.artifacts.generate_cinematic_video(
                notebook_id, **cine_kwargs
            )
        elif t == "video":
            status = await client.artifacts.generate_video(notebook_id, **opts)
        elif t == "video-explainer":
            # Force EXPLAINER format regardless of caller-supplied options.
            # Plain "video" sidecar type produces the default BRIEF format.
            if _ENUMS_AVAILABLE:
                opts["video_format"] = VideoFormat.EXPLAINER
            else:
                opts["video_format"] = "EXPLAINER"
            status = await client.artifacts.generate_video(notebook_id, **opts)
        elif t == "report":
            status = await client.artifacts.generate_report(notebook_id, **opts)
        elif t == "quiz":
            status = await client.artifacts.generate_quiz(notebook_id, **opts)
        elif t == "flashcards":
            status = await client.artifacts.generate_flashcards(notebook_id, **opts)
        elif t == "slide_deck":
            status = await client.artifacts.generate_slide_deck(notebook_id, **opts)
        elif t == "slide-detailed":
            # Force DETAILED_DECK slide_format. Download path returns pptx.
            if _ENUMS_AVAILABLE:
                opts["slide_format"] = SlideDeckFormat.DETAILED_DECK
            else:
                opts["slide_format"] = "DETAILED_DECK"
            status = await client.artifacts.generate_slide_deck(notebook_id, **opts)
        elif t == "infographic":
            status = await client.artifacts.generate_infographic(notebook_id, **opts)
        elif t == "data_table":
            status = await client.artifacts.generate_data_table(notebook_id, **opts)
        elif t == "mind_map":
            # mind_map may return dict directly in docs :contentReference[oaicite:6]{index=6}
            out = await client.artifacts.generate_mind_map(notebook_id, **opts)
            return {"ok": True, "type": t, "result": out}
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported artifact type: {t}")

        # GenerationStatus commonly contains task_id :contentReference[oaicite:7]{index=7}
        payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", {})
        return {"ok": True, "type": t, "status": payload}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.get("/v1/notebooks/{notebook_id}/artifacts/tasks/{task_id}")
async def poll_task(
    notebook_id: str,
    task_id: str,
    wait: bool = False,
    timeout: float = 3300.0,
):
    """Poll a generation task, optionally blocking until it completes.

    ``wait=false`` (the default, and what reg-monitor-app uses) returns the
    current status immediately; the caller drives its own polling schedule.

    ``wait=true`` blocks. notebooklm-py 0.8.0 defaults ``wait_for_completion``
    to a 300s timeout, which is far shorter than real generation times —
    slide decks routinely exceed five minutes and cinematic video can run for
    an hour — so we default ``timeout`` to 55 minutes instead and let callers
    override it.

    A timeout is NOT an error: the task is still running server-side. We
    surface the last known status with ``timed_out: true`` so the caller can
    keep polling rather than treating it as a failed generation.
    """
    client = await get_client()
    try:
        if wait:
            status = await client.artifacts.wait_for_completion(
                notebook_id, task_id, timeout=timeout
            )
        else:
            status = await client.artifacts.poll_status(notebook_id, task_id)

        payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", status)
        return {"ok": True, "status": payload}
    except ArtifactTimeoutError as e:
        # Still pending upstream — report last-known state, don't 500.
        try:
            status = await client.artifacts.poll_status(notebook_id, task_id)
            payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", status)
        except Exception:
            payload = {"task_id": task_id, "status": "pending", "url": None, "error": None}
        return {"ok": True, "status": payload, "timed_out": True, "message": str(e)}
    except RPCError as e:
        raise map_rpc_error(e)


@v1.get("/v1/notebooks/{notebook_id}/artifacts/download")
async def download_artifact(
    notebook_id: str,
    type: Literal[
        "audio",
        "video",
        "video-explainer",
        "cinematic-video",
        "infographic",
        "slide_deck",
        "slide-detailed",
        "report",
        "mind_map",
        "data_table",
        "quiz",
        "flashcards",
    ],
    artifact_id: Optional[str] = None,
    output_format: Optional[Literal["json", "markdown", "html"]] = None,
):
    """
    Downloads the *first completed* artifact of the given type unless artifact_id is provided.
    notebooklm-py provides type-specific download_* methods. :contentReference[oaicite:8]{index=8}
    """
    suffix_map = {
        "audio": ".mp4",
        "video": ".mp4",
        "video-explainer": ".mp4",
        "cinematic-video": ".mp4",
        "infographic": ".png",
        "slide_deck": ".pdf",
        "slide-detailed": ".pptx",
        "report": ".md",
        "mind_map": ".json",
        "data_table": ".csv",
        "quiz": ".json" if (output_format in (None, "json")) else (".md" if output_format == "markdown" else ".html"),
        "flashcards": ".json" if (output_format in (None, "json")) else (".md" if output_format == "markdown" else ".html"),
    }
    out_path = os.path.join(tempfile.gettempdir(), f"nlm_{uuid.uuid4().hex}{suffix_map[type]}")

    client = await get_client()
    try:
        if type == "audio":
            await client.artifacts.download_audio(notebook_id, out_path, artifact_id=artifact_id)
        elif type in ("video", "video-explainer", "cinematic-video"):
            await client.artifacts.download_video(notebook_id, out_path, artifact_id=artifact_id)
        elif type == "infographic":
            await client.artifacts.download_infographic(notebook_id, out_path, artifact_id=artifact_id)
        elif type == "slide_deck":
            await client.artifacts.download_slide_deck(notebook_id, out_path, artifact_id=artifact_id)
        elif type == "slide-detailed":
            # DETAILED_DECK slide_format produces pptx via download.
            await client.artifacts.download_slide_deck(
                notebook_id, out_path, artifact_id=artifact_id, output_format="pptx"
            )
        elif type == "report":
            await client.artifacts.download_report(notebook_id, out_path, artifact_id=artifact_id)
        elif type == "mind_map":
            await client.artifacts.download_mind_map(notebook_id, out_path, artifact_id=artifact_id)
        elif type == "data_table":
            await client.artifacts.download_data_table(notebook_id, out_path, artifact_id=artifact_id)
        elif type == "quiz":
            await client.artifacts.download_quiz(
                notebook_id, out_path, artifact_id=artifact_id, output_format=(output_format or "json")
            )
        elif type == "flashcards":
            await client.artifacts.download_flashcards(
                notebook_id, out_path, artifact_id=artifact_id, output_format=(output_format or "json")
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported type: {type}")

        filename = os.path.basename(out_path)
        return FileResponse(out_path, filename=filename)
    except RPCError as e:
        # Clean up file if partially created
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        raise map_rpc_error(e)


# Mounted last: include_router copies the routes registered so far.
app.include_router(v1)
