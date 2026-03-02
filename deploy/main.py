"""
deploy/main.py
==============
FastAPI bridge — ForexRL multi-model inference server.

Endpoints
─────────
  POST /{algo}/v{n}/predict    — versioned inference (e.g. /rppo/v1/predict)
  POST /predict                — backward-compat alias → default model
  POST /trade/close            — EA reports a closed trade
  GET  /models                 — list loaded models
  GET  /health                 — server + model health
  GET  /metrics                — live performance snapshot
  GET  /history                — closed trade log
  WS   /ws                     — WebSocket for dashboard
  GET  /                       — dashboard HTML

Run:
  uvicorn deploy.main:app --host 0.0.0.0 --port 8000
  # From project root with BOT_ROOT auto-detected.
"""

from __future__ import annotations

import sys, os, time, json, asyncio, logging, uuid
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import uvicorn

from contextlib import asynccontextmanager
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Path as FPath,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Project root ──────────────────────────────────────────────────────────────
BOT_ROOT = Path(os.getenv("BOT_ROOT", str(Path(__file__).parent.parent))).resolve()
sys.path.insert(0, str(BOT_ROOT))

from deploy.predictor import ModelRegistry
from deploy.state import SessionStore, PerformanceTracker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger("bridge")

# ── App ───────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────
    log.info("ForexRL Bridge starting...")
    log.info(f"BOT_ROOT = {BOT_ROOT}")
    found = registry.discover()
    if not found:
        log.error(
            "No models found! Place your model at:\n"
            f"  {BOT_ROOT}/models/rppo/v1/final_model.zip\n"
            "Then restart the server."
        )
    else:
        registry.load_all()
        log.info(f"ForexRL Bridge ready — http://0.0.0.0:8000")
        log.info(
            f"Models: {[m['algo']+'/'+m['version'] for m in registry.list_models()]}"
        )
    yield
    # ── Shutdown ──────────────────────────────────────────────────
    log.info("ForexRL Bridge shutting down.")


app = FastAPI(
    title="ForexRL Bridge",
    version="2.0.0",
    description="Multi-model RPPO/PPO inference server for MT5 EA",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Singletons ────────────────────────────────────────────────────────────────
registry = ModelRegistry(BOT_ROOT)
sessions = SessionStore()
perf = PerformanceTracker()

_ws_clients: set[WebSocket] = set()
_start_time = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════


class Candle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 1.0


class PredictRequest(BaseModel):
    session_id: str
    symbol: str = "EURUSD"
    candles: list[Candle]
    current_position: int = Field(0, ge=-1, le=1)
    account_balance: float = 10_000
    starting_balance: float = 10_000
    reset_lstm: bool = False


class PredictResponse(BaseModel):
    action: int  # -1 SHORT / 0 FLAT / +1 LONG
    action_index: int  # raw model output 0/1/2 (debug)
    risk_pct: float
    # ── Fixed SL/TP mode (UseModelExit=false) ──────────────────────────────
    sl_pips: float  # structure-aware SL distance
    tp_pips: float  # fixed TP = sl_pips × MIN_RR_RATIO
    sl_price: float  # exact SL price level
    sl_type: str  # "ob_low"/"stl"/"atr_default"/"atr_cap"
    # ── Model-exit mode (UseModelExit=true) ────────────────────────────────
    catastrophic_sl_pips: float  # wide black-swan SL — rarely fires
    catastrophic_sl_price: float  # exact catastrophic SL price level
    # ── Common ─────────────────────────────────────────────────────────────
    atr_pips: float
    confidence: float
    bias_aligned: float
    killzone_active: int
    signal_id: str
    model: str
    timestamp: str


class TradeCloseRequest(BaseModel):
    session_id: str
    signal_id: str
    pnl_pct: float
    close_reason: str = "tp"


# ══════════════════════════════════════════════════════════════════════════════
# SHARED INFERENCE LOGIC
# ══════════════════════════════════════════════════════════════════════════════


def _build_df(candles: list[Candle]) -> pd.DataFrame:
    df = pd.DataFrame([c.dict() for c in candles])
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("date").rename(columns={"volume": "volume"})[
        ["open", "high", "low", "close", "volume"]
    ]
    return df


def _ratchet_risk(balance: float, starting: float) -> float:
    ratio = balance / max(starting, 1.0)
    if ratio >= 6.0:
        return 0.07
    if ratio >= 3.0:
        return 0.10
    return 0.15


async def _run_prediction(
    req: PredictRequest,
    algo: str,
    version: str,
) -> PredictResponse:
    """Core inference logic shared by all predict endpoints."""
    if len(req.candles) < 100:
        raise HTTPException(400, "Need ≥100 candles for feature pipeline")

    t0 = time.perf_counter()
    df = _build_df(req.candles)

    result = registry.predict(
        algo=algo,
        version=version,
        session_id=req.session_id,
        df=df,
        current_position=req.current_position,
        reset_lstm=req.reset_lstm,
    )

    risk_pct = _ratchet_risk(req.account_balance, req.starting_balance)
    signal_id = str(uuid.uuid4())[:8].upper()
    ts = datetime.now(timezone.utc).isoformat()
    model_key = f"{algo}/{version}"
    latency_ms = (time.perf_counter() - t0) * 1000

    resp = PredictResponse(
        action=result["action"],
        action_index=result["action_index"],
        risk_pct=risk_pct,
        sl_pips=result["sl_pips"],
        tp_pips=result["tp_pips"],
        sl_price=result["sl_price"],
        sl_type=result["sl_type"],
        catastrophic_sl_pips=result["catastrophic_sl_pips"],
        catastrophic_sl_price=result["catastrophic_sl_price"],
        atr_pips=result["atr_pips"],
        confidence=result["confidence"],
        bias_aligned=result["bias_aligned"],
        killzone_active=result["killzone_active"],
        signal_id=signal_id,
        model=model_key,
        timestamp=ts,
    )

    if result["action"] != 0:
        perf.register_signal(
            signal_id,
            {
                "action": result["action"],
                "risk_pct": risk_pct,
                "confidence": result["confidence"],
                "symbol": req.symbol,
                "timestamp": ts,
                "balance": req.account_balance,
                "model": model_key,
            },
        )
    else:
        perf.register_flat()

    perf.update_latency(latency_ms)

    asyncio.create_task(
        broadcast(
            {
                "type": "prediction",
                "signal_id": signal_id,
                "action": result["action"],
                "action_index": result["action_index"],
                "risk_pct": risk_pct,
                "confidence": result["confidence"],
                "bias_aligned": result["bias_aligned"],
                "killzone": result["killzone_active"],
                "atr_pips": result["atr_pips"],
                "sl_pips": result["sl_pips"],
                "sl_type": result["sl_type"],
                "balance": req.account_balance,
                "multiplier": round(
                    req.account_balance / max(req.starting_balance, 1), 3
                ),
                "latency_ms": round(latency_ms, 1),
                "model": model_key,
                "timestamp": ts,
                **perf.snapshot(),
            }
        )
    )

    log.info(
        f"[{model_key}][{req.session_id}] "
        f"action={result['action']:+d}(idx={result['action_index']})  "
        f"risk={risk_pct:.0%}  conf={result['confidence']:.2f}  "
        f"{latency_ms:.0f}ms"
    )
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/{algo}/v{version}/predict", response_model=PredictResponse)
async def predict_versioned(
    algo: str = FPath(..., description="Algorithm: rppo, ppo, a2c"),
    version: str = FPath(..., description="Model version number, e.g. 1"),
    req: PredictRequest = ...,
) -> PredictResponse:
    """
    Versioned prediction endpoint.
    Example: POST /rppo/v1/predict
    """
    version_str = f"v{version}"
    try:
        return await _run_prediction(req, algo, version_str)
    except KeyError as e:
        log.error(f"Model not found: {e}")
        raise HTTPException(404, str(e))
    except ValueError as e:
        log.error(f"Prediction ValueError: {e}")
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        log.error(f"Prediction RuntimeError: {e}")
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(
            f"Prediction unexpected error: {type(e).__name__}: {e}", exc_info=True
        )
        raise HTTPException(500, str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict_default(req: PredictRequest) -> PredictResponse:
    """
    Backward-compatible endpoint — routes to the default (first loaded) model.
    Useful for existing EAs before adding model selection.
    """
    if not registry.default_key:
        raise HTTPException(503, "No models loaded")
    algo, version = registry.default_key.split("/", 1)
    try:
        return await _run_prediction(req, algo, version)
    except ValueError as e:
        log.error(f"Prediction ValueError: {e}")
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        log.error(f"Prediction RuntimeError: {e}")
        raise HTTPException(503, str(e))
    except Exception as e:
        log.error(
            f"Prediction unexpected error: {type(e).__name__}: {e}", exc_info=True
        )
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TRADE CLOSE
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/trade/close")
async def trade_close(req: TradeCloseRequest) -> dict:
    perf.record_close(req.signal_id, req.pnl_pct, req.close_reason)
    asyncio.create_task(
        broadcast(
            {
                "type": "trade_close",
                "signal_id": req.signal_id,
                "pnl_pct": req.pnl_pct,
                "close_reason": req.close_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **perf.snapshot(),
            }
        )
    )
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# INFO ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/models")
async def list_models() -> JSONResponse:
    """List all available models with their status."""
    return JSONResponse(
        {
            "models": registry.list_models(),
            "default": registry.default_key,
            "total": len(registry.list_models()),
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    models = registry.list_models()
    return JSONResponse(
        {
            "status": "ok" if registry.any_loaded() else "degraded",
            "models_loaded": sum(1 for m in models if m["loaded"]),
            "models_total": len(models),
            "default_model": registry.default_key,
            "active_sessions": sum(m.get("active_sessions", 0) for m in models),
            "uptime_s": int(time.time() - _start_time),
            "ws_clients": len(_ws_clients),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.get("/metrics")
async def metrics() -> JSONResponse:
    return JSONResponse(perf.snapshot())


@app.get("/history")
async def history(limit: int = 100, offset: int = 0) -> JSONResponse:
    return JSONResponse(perf.trade_history(limit=limit, offset=offset))


# ── Dashboard ─────────────────────────────────────────────────────────────────
_dashboard_html: str | None = None


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    global _dashboard_html
    if _dashboard_html is None:
        p = Path(__file__).parent / "dashboard.html"
        _dashboard_html = p.read_text(encoding="utf-8")
    return HTMLResponse(_dashboard_html)


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════════════════════════════


async def broadcast(msg: dict) -> None:
    dead = set()
    payload = json.dumps(msg)
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.add(ws)
    log.info(f"Dashboard connected ({len(_ws_clients)} clients)")
    await ws.send_text(json.dumps({"type": "snapshot", **perf.snapshot()}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        log.info(f"Dashboard disconnected ({len(_ws_clients)} clients)")


if __name__ == "__main__":
    uvicorn.run("deploy.main:app", host="0.0.0.0", port=8000, reload=False)
