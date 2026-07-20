"""FastAPI + WebSocket server for the live BMS dashboard.

Serves the static frontend and one WebSocket endpoint that streams
engine snapshots at ~5 Hz while accepting UI commands (fault injection,
load mode, speed, balancer toggle, reset) on the same socket.

Run:  python dashboard/server.py   (from anywhere; sys.path is fixed up)
Then open http://127.0.0.1:8420
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from dashboard.engine import SimEngine  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
engine: SimEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SimEngine()
    engine.start()
    yield
    engine.running = False


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json({
        "type": "config",
        "rows": 6,
        "cols": 8,
        "modes": ["drive", "rest", "charge", "discharge"],
    })

    async def sender() -> None:
        while True:
            await ws.send_json(engine.snapshot())
            await asyncio.sleep(0.2)

    task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive_json()
            if isinstance(msg, dict):
                engine.command(msg)
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8420, log_level="warning")
