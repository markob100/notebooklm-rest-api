# Google NotebookLM REST API wrapper
# Namhyeon Go <gnh1201@catswords.re.kr>
# https://github.com/gnh1201/notebooklm-rest-api
import os
import uuid
import tempfile
from typing import Any, Optional, Literal, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse
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
API_KEY = os.environ.get("NOTEBOOKLM_REST_API_KEY", "")  # set this in production
AUTH_STORAGE_PATH = os.environ.get("NOTEBOOKLM_STORAGE_PATH")  # optional override


def require_api_key(x_api_key: Optional[str] = None):
    # Minimal API-key gate. Put this behind a real gateway (Cloudflare, Nginx, etc.) for production.
    if API_KEY:
        # FastAPI header parsing without extra imports (keep simple):
        # Prefer: from fastapi import Header; def require_api_key(x_api_key: str = Header(None)) ...
        # but we keep it minimal and rely on query param fallback too.
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")


async def get_client() -> NotebookLMClient:
    """Create a NotebookLMClient from the shared storage session.

    Auth precedence (delegated to notebooklm-py):
      - explicit path (``AUTH_STORAGE_PATH`` env → ``NOTEBOOKLM_STORAGE_PATH``)
      - ``NOTEBOOKLM_AUTH_JSON``
      - ``NOTEBOOKLM_HOME/storage_state.json``
      - ``~/.notebooklm/storage_state.json``

    BYO-cookies (per-request user sessions) is deferred — see reg-monitor-app
    BACKLOG item #64. The upstream ``notebooklm-py==0.3.4`` pinned in
    requirements.txt does not expose a stateless ``NotebookLMClient.from_cookies``
    entry point today, so any BYO path would 500 on import against the pinned
    version. The stateless entry will be vendored locally into the sidecar
    when that feature lands, rather than taking on an upstream dependency.
    """
    try:
        if AUTH_STORAGE_PATH:
            return await NotebookLMClient.from_storage(AUTH_STORAGE_PATH)
        return await NotebookLMClient.from_storage()
    except ValueError as e:
        msg = str(e)
        code = "AUTH_INVALID"
        if "expired" in msg.lower() or "redirected" in msg.lower():
            code = "AUTH_EXPIRED"
        raise HTTPException(
            status_code=401,
            detail={"code": code, "message": msg, "recoverable": True},
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "AUTH_NOT_CONFIGURED",
                "message": str(e),
                "recoverable": False,
            },
        )
    except Exception as e:
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
app = FastAPI(title="NotebookLM REST API (powered by notebooklm-py)")


@app.get("/health")
async def health():
    return {"ok": True}


# ----------------------------
# Auth
# ----------------------------
@app.post("/v1/auth/refresh")
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
    async with client:
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
            raise map_notebooklm_error(e)


# ----------------------------
# Notebooks
# ----------------------------
@app.get("/v1/notebooks")
async def list_notebooks():
    client = await get_client()
    async with client:
        try:
            nbs = await client.notebooks.list()
            return {"ok": True, "items": [nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__ for nb in nbs]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks")
async def create_notebook(req: NotebookCreateReq):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.create(req.title)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}")
async def get_notebook(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.get(notebook_id)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.delete("/v1/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            ok = await client.notebooks.delete(notebook_id)
            return {"ok": True, "deleted": bool(ok)}
        except RPCError as e:
            raise map_rpc_error(e)


@app.patch("/v1/notebooks/{notebook_id}/rename")
async def rename_notebook(notebook_id: str, req: NotebookRenameReq):
    client = await get_client()
    async with client:
        try:
            nb = await client.notebooks.rename(notebook_id, req.new_title)
            return {"ok": True, "notebook": nb.model_dump() if hasattr(nb, "model_dump") else nb.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/summary")
async def get_notebook_summary(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            summary = await client.notebooks.get_summary(notebook_id)
            return {"ok": True, "summary": summary}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/description")
async def get_notebook_description(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            desc = await client.notebooks.get_description(notebook_id)
            return {"ok": True, "description": desc.model_dump() if hasattr(desc, "model_dump") else desc.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Sources
# ----------------------------
@app.get("/v1/notebooks/{notebook_id}/sources")
async def list_sources(notebook_id: str):
    client = await get_client()
    async with client:
        try:
            items = await client.sources.list(notebook_id)
            return {"ok": True, "items": [s.model_dump() if hasattr(s, "model_dump") else s.__dict__ for s in items]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/url")
async def add_source_url(notebook_id: str, req: SourceAddUrlReq):
    client = await get_client()
    async with client:
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


@app.post("/v1/notebooks/{notebook_id}/sources/youtube")
async def add_source_youtube(notebook_id: str, req: SourceAddYoutubeReq):
    client = await get_client()
    async with client:
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


@app.post("/v1/notebooks/{notebook_id}/sources/text")
async def add_source_text(notebook_id: str, req: SourceAddTextReq):
    client = await get_client()
    async with client:
        try:
            src = await client.sources.add_text(notebook_id, req.title, req.content)
            return {"ok": True, "source": src.model_dump() if hasattr(src, "model_dump") else src.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/sources/file")
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
    async with client:
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


@app.get("/v1/notebooks/{notebook_id}/sources/{source_id}/fulltext")
async def get_source_fulltext(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            ft = await client.sources.get_fulltext(notebook_id, source_id)
            return {"ok": True, "fulltext": ft.model_dump() if hasattr(ft, "model_dump") else ft.__dict__}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/sources/{source_id}/guide")
async def get_source_guide(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            guide = await client.sources.get_guide(notebook_id, source_id)
            return {"ok": True, "guide": guide}
        except RPCError as e:
            raise map_rpc_error(e)


@app.delete("/v1/notebooks/{notebook_id}/sources/{source_id}")
async def delete_source(notebook_id: str, source_id: str):
    client = await get_client()
    async with client:
        try:
            ok = await client.sources.delete(notebook_id, source_id)
            return {"ok": True, "deleted": bool(ok)}
        except RPCError as e:
            raise map_rpc_error(e)


# ----------------------------
# Chat
# ----------------------------
@app.post("/v1/notebooks/{notebook_id}/chat/ask")
async def chat_ask(notebook_id: str, req: ChatAskReq):
    client = await get_client()
    async with client:
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
@app.get("/v1/notebooks/{notebook_id}/artifacts")
async def list_artifacts(notebook_id: str, type: Optional[str] = None):
    client = await get_client()
    async with client:
        try:
            items = await client.artifacts.list(notebook_id, type=type) if type else await client.artifacts.list(notebook_id)
            return {"ok": True, "items": [a.model_dump() if hasattr(a, "model_dump") else a.__dict__ for a in items]}
        except RPCError as e:
            raise map_rpc_error(e)


@app.post("/v1/notebooks/{notebook_id}/artifacts/generate")
async def generate_artifact(notebook_id: str, req: ArtifactGenerateReq):
    client = await get_client()
    async with client:
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


@app.get("/v1/notebooks/{notebook_id}/artifacts/tasks/{task_id}")
async def poll_task(notebook_id: str, task_id: str, wait: bool = False):
    client = await get_client()
    async with client:
        try:
            if wait:
                status = await client.artifacts.wait_for_completion(notebook_id, task_id)
            else:
                status = await client.artifacts.poll_status(notebook_id, task_id)

            payload = status.model_dump() if hasattr(status, "model_dump") else getattr(status, "__dict__", status)
            return {"ok": True, "status": payload}
        except RPCError as e:
            raise map_rpc_error(e)


@app.get("/v1/notebooks/{notebook_id}/artifacts/download")
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
    async with client:
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
