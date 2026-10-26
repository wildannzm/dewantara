"""FastAPI application exposing the Dewantara AI detection API.

Surfaces mirror the original production service (``api.dewantara.cloud``):

* ``GET /``       service banner
* ``GET /health`` liveness probe
* ``WS  /ws``     one JPEG frame in, one plain-text answer out

The service is public and unauthenticated by design (the original API was too), so the
defences here are about resource safety rather than identity: origin allow-list on the
WebSocket handshake, frame-size and pixel limits, bounded concurrency, and idle timeouts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.detector import NO_DETECTION, BisindoDetector

logger = logging.getLogger("dewantara.ai")

SERVICE_BANNER = {"message": "Dewantara AI Server Ready", "status": "active"}
WELCOME_MESSAGE = {"status": "connected", "message": "Welcome"}


def is_allowed_origin(origin: str | None, allowed_origins: tuple[str, ...]) -> bool:
    """Browser origins must be allow-listed; non-browser clients send no Origin header."""
    return origin is None or "*" in allowed_origins or origin in allowed_origins


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepting settings keeps the app testable."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=settings.log_level)
        app.state.detector = BisindoDetector(
            model_path=settings.model_path,
            input_size=settings.input_size,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            max_image_pixels=settings.max_image_pixels,
        )
        app.state.inference_slots = asyncio.Semaphore(settings.max_concurrent_inferences)
        logger.info(
            "Dewantara AI %s ready (model=%s, input=%d, conf=%.2f, slots=%d)",
            __version__,
            settings.model_path.name,
            settings.input_size,
            settings.confidence_threshold,
            settings.max_concurrent_inferences,
        )
        yield
        logger.info("Dewantara AI stopped")

    app = FastAPI(title="Dewantara AI", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.state.settings = settings

    @app.get("/")
    async def service_banner() -> dict[str, str]:
        return SERVICE_BANNER

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def detect(websocket: WebSocket) -> None:
        settings = websocket.app.state.settings
        origin = websocket.headers.get("origin")
        if not is_allowed_origin(origin, settings.allowed_origins):
            logger.warning("Rejected handshake from origin %r", origin)
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        detector: BisindoDetector = websocket.app.state.detector
        slots: asyncio.Semaphore = websocket.app.state.inference_slots
        await websocket.accept()
        await websocket.send_text(json.dumps(WELCOME_MESSAGE))

        try:
            await _serve_frames(websocket, detector, slots, settings)
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("WebSocket session failed")
            with suppress(Exception):
                await websocket.close()

    return app


async def _serve_frames(
    websocket: WebSocket,
    detector: BisindoDetector,
    slots: asyncio.Semaphore,
    settings: Settings,
) -> None:
    while True:
        try:
            message = await asyncio.wait_for(
                websocket.receive(), timeout=settings.idle_timeout_seconds
            )
        except TimeoutError:
            logger.info("Closing idle session")
            await websocket.close(code=status.WS_1001_GOING_AWAY)
            return

        if message["type"] == "websocket.disconnect":
            return

        frame = message.get("bytes")
        if frame is None:
            await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
            return

        if len(frame) > settings.max_frame_bytes:
            logger.warning("Rejected oversized frame (%d bytes)", len(frame))
            await websocket.send_text(NO_DETECTION)
            continue

        # Inference is CPU-bound and blocking; keep the event loop responsive and cap
        # how many frames are worked on at once so one client cannot starve the others.
        async with slots:
            answer = await asyncio.to_thread(_safe_predict, detector, frame)
        await websocket.send_text(answer)


def _safe_predict(detector: BisindoDetector, frame: bytes) -> str:
    """Never let a malformed frame take down an active session."""
    try:
        return detector.predict(frame)
    except Exception:
        logger.exception("Inference failed for a %d-byte frame", len(frame))
        return NO_DETECTION


app = create_app()
