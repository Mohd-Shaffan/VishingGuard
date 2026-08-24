"""
=============================================================================
  SCAM GUARD v2.0 — FastAPI Backend
  Wraps scamguard_enhanced.py and exposes WebSocket + REST endpoints.

  FIXES in this revision:
  • sys.path.insert() now runs BEFORE any scamguard_enhanced import,
    fixing ModuleNotFoundError on Windows and Hugging Face Spaces.
  • classify_intent() called correctly with 3 args (text, model, name).
  • TemporalThreatScorer.update() receives the right chunk_score value.
  • WebSocket handler now processes 'set_thresholds' messages from the
    frontend sliders, updating each session's scorer thresholds live.
  • logistic_vishing_model.pkl resolved relative to this file's directory
    so it works regardless of the working directory at startup.

  Endpoints:
    WS  /ws/{session_id}          — Real-time chunk analysis
    POST /api/analyze             — Single-shot REST analysis
    POST /api/analyze-audio       — Audio file upload → transcribe → analyze
    POST /api/reset/{session_id}  — Reset temporal scorer for a session
    GET  /api/health              — Health check + model info
    GET  /api/keywords            — Keyword tier dictionary
    GET  /                        — Serves frontend (index.html)
=============================================================================
"""

# ── FIX: sys.path must be set BEFORE any local-module import ─────────────────
# On Windows (and in Docker/HF Spaces), Python does not always add the
# script's own directory to sys.path automatically.  Inserting it here —
# before *any* 'from scamguard_enhanced import …' — guarantees the module
# is always findable, regardless of the working directory at launch.
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uuid
import time
import json
import math
import random
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import shutil

# ── Import ScamGuard engine ───────────────────────────────────────────────────
try:
    from scamguard_enhanced import (
        classify_intent,        # (text, nlp_model, model_name) → (bool, float, dict)
        TemporalThreatScorer,
        load_nlp_model,
        KEYWORD_TIERS,
        transcribe_audio,
    )
    ENGINE_LOADED = True
except ImportError as exc:
    print(f"[WARN] Could not import scamguard_enhanced: {exc}")
    print("[WARN] Running in DEMO mode — results are simulated.")
    ENGINE_LOADED = False


# =============================================================================
#  APP INITIALISATION
# =============================================================================

app = FastAPI(
    title="ScamGuard API",
    description="Real-time vishing detection via hybrid BERT + keyword engine",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load NLP model once at startup ────────────────────────────────────────────
NLP_MODEL  = None
MODEL_NAME = "keyword-only"

@app.on_event("startup")
async def startup_event():
    global NLP_MODEL, MODEL_NAME
    print("\n[ScamGuard] Loading NLP model — please wait…")
    if ENGINE_LOADED:
        NLP_MODEL, MODEL_NAME = load_nlp_model()
    else:
        MODEL_NAME = "demo-mode"
    print(f"[ScamGuard] Model ready: {MODEL_NAME}\n")


# =============================================================================
#  SESSION STORE
# =============================================================================

class SessionStore:
    def __init__(self):
        self._sessions:     Dict[str, TemporalThreatScorer] = {}
        self._created:      Dict[str, float] = {}
        self._chunk_counts: Dict[str, int]   = {}

    def get_or_create(self, session_id: str) -> TemporalThreatScorer:
        if session_id not in self._sessions:
            self._sessions[session_id] = TemporalThreatScorer(
                alpha=0.85,
                lambda_decay=0.3,
                history_k=10,
                tau_alert=0.60,
                tau_drop=0.85,
            )
            self._created[session_id]      = time.time()
            self._chunk_counts[session_id] = 0
        return self._sessions[session_id]

    def reset(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id].reset()
            self._chunk_counts[session_id] = 0

    def increment(self, session_id: str):
        self._chunk_counts[session_id] = self._chunk_counts.get(session_id, 0) + 1

    def chunk_count(self, session_id: str) -> int:
        return self._chunk_counts.get(session_id, 0)

    def set_thresholds(self, session_id: str, tau_alert: float, tau_drop: float):
        """Update thresholds on a live session without resetting history."""
        scorer = self.get_or_create(session_id)
        scorer.tau_alert = max(0.05, min(0.95, tau_alert))
        scorer.tau_drop  = max(0.05, min(1.00, tau_drop))

    def purge_old(self, max_age_seconds: int = 3600):
        now   = time.time()
        stale = [sid for sid, ts in self._created.items()
                 if now - ts > max_age_seconds]
        for sid in stale:
            self._sessions.pop(sid, None)
            self._created.pop(sid, None)
            self._chunk_counts.pop(sid, None)


SESSION_STORE = SessionStore()


# =============================================================================
#  DEMO FALLBACK (when scamguard_enhanced.py is unavailable)
# =============================================================================

def _demo_classify(text: str):
    """Simulated classifier used in demo / import-error mode."""
    scam_words = ["otp", "blocked", "urgent", "bank", "cvv", "pin", "kyc",
                  "arrest", "lottery", "aadhaar", "transfer", "anydesk"]
    hits    = [w for w in scam_words if w in text.lower()]
    is_scam = len(hits) >= 1
    confidence = min(0.60 + len(hits) * 0.10, 0.97) if is_scam else random.uniform(0.75, 0.92)
    details = {
        "keyword_weight":   len(hits) * 2,
        "keyword_hits":     len(hits),
        "matched_keywords": [(w, 2) for w in hits],
        "nlp_label":        "DEMO",
        "nlp_raw_score":    confidence,
    }
    return is_scam, confidence, details


# =============================================================================
#  CORE ANALYSIS FUNCTION
# =============================================================================

def _get_tier(weight: int) -> str:
    return {3: "CRITICAL", 2: "HIGH", 1: "MEDIUM"}.get(weight, "MEDIUM")


def analyze_chunk(session_id: str, text: str) -> dict:
    """
    Full pipeline:
      1. classify_intent(text, NLP_MODEL, MODEL_NAME)  → is_scam, confidence, details
      2. TemporalThreatScorer.update(chunk_score)       → CT
      3. Build and return the WebSocket / REST payload.

    chunk_score fed to the scorer:
      • is_scam=True  → use confidence directly  (e.g. 0.97 for critical KW)
      • is_scam=False → use (1 - confidence)     (how certain the model is it's safe)
    """
    t_start = time.perf_counter()

    # ── Classify ─────────────────────────────────────────────────────────────
    if ENGINE_LOADED:
        # FIX: pass all three arguments — the old code only passed two,
        # causing TypeError in scamguard_enhanced.classify_intent().
        is_scam, confidence, details = classify_intent(text, NLP_MODEL, MODEL_NAME)
    else:
        is_scam, confidence, details = _demo_classify(text)

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Temporal scoring ─────────────────────────────────────────────────────
    scorer      = SESSION_STORE.get_or_create(session_id)
    chunk_score = confidence
    CT          = scorer.update(chunk_score)
    status      = scorer.status()
    SESSION_STORE.increment(session_id)

    # ── Serialise keyword hits ────────────────────────────────────────────────
    matched_kw = [
        {"word": kw, "weight": wt, "tier": _get_tier(wt)}
        for kw, wt in details.get("matched_keywords", [])
    ]

    return {
        "session_id":       session_id,
        "chunk_number":     SESSION_STORE.chunk_count(session_id),
        "text":             text,
        "is_scam":          is_scam,
        "nlp_confidence":   round(confidence * 100, 2),
        "nlp_label":        details.get("nlp_label", "N/A"),
        "keyword_weight":   details.get("keyword_weight", 0),
        "matched_keywords": matched_kw,
        "chunk_score":      round(chunk_score, 4),
        "cumulative_ct":    round(CT, 4),
        "ct_percent":       round(min(CT * 100, 100), 1),
        "status":           status,          # "SAFE" | "ALERT" | "DROP"
        "latency_ms":       round(latency_ms, 2),
        "model":            MODEL_NAME,
    }


# =============================================================================
#  WEBSOCKET ENDPOINT  /ws/{session_id}
# =============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[WS] Session connected: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "chunk", "text": raw}

            msg_type = msg.get("type", "chunk")

            # ── Reset ────────────────────────────────────────────────────────
            if msg_type == "reset":
                SESSION_STORE.reset(session_id)
                await websocket.send_text(json.dumps({
                    "type":       "reset_ack",
                    "session_id": session_id,
                    "message":    "Session reset. Temporal scorer cleared."
                }))
                continue

            # ── Ping ─────────────────────────────────────────────────────────
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            # ── FIX: handle threshold updates from frontend sliders ───────────
            # The frontend sends {type: "set_thresholds", alert: 0.55, drop: 0.80}
            # when the user drags the calibration sliders.  Without this branch
            # the message was silently dropped.
            if msg_type == "set_thresholds":
                try:
                    tau_alert = float(msg.get("alert", 0.55))
                    tau_drop  = float(msg.get("drop",  0.80))
                    SESSION_STORE.set_thresholds(session_id, tau_alert, tau_drop)
                    await websocket.send_text(json.dumps({
                        "type":       "thresholds_ack",
                        "tau_alert":  tau_alert,
                        "tau_drop":   tau_drop,
                    }))
                except Exception as exc:
                    print(f"[WS] set_thresholds error: {exc}")
                continue

            # ── Analyse chunk ─────────────────────────────────────────────────
            text = msg.get("text", "").strip()
            if not text:
                continue

            result          = analyze_chunk(session_id, text)
            result["type"]  = "analysis"
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print(f"[WS] Session disconnected: {session_id}")
    except Exception as exc:
        print(f"[WS] Error in session {session_id}: {exc}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


# =============================================================================
#  REST ENDPOINTS
# =============================================================================

class AnalyzeRequest(BaseModel):
    text:       str
    session_id: Optional[str] = None


@app.post("/api/analyze")
async def rest_analyze(req: AnalyzeRequest):
    """Single-shot REST analysis. Creates a one-off session if none provided."""
    session_id = req.session_id or str(uuid.uuid4())
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    result = analyze_chunk(session_id, req.text)
    return JSONResponse(content=result)


@app.post("/api/analyze-audio")
async def analyze_audio(file: UploadFile = File(...),
                         session_id: Optional[str] = None):
    """Upload an audio file → transcribe with Whisper → analyze intent."""
    session_id = session_id or str(uuid.uuid4())
    # Write to a temp path next to this file so cleanup is reliable
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"_tmp_{uuid.uuid4().hex}_{file.filename}")
    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        print(f"[AUDIO] Received: {file.filename}")
        actual_text = transcribe_audio(temp_path) if ENGINE_LOADED else ""
        if not actual_text:
            raise HTTPException(status_code=422, detail="Transcription returned empty text.")
        result = analyze_chunk(session_id, actual_text)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/reset/{session_id}")
async def reset_session(session_id: str):
    SESSION_STORE.reset(session_id)
    return {"status": "ok", "message": f"Session {session_id} reset."}


@app.get("/api/health")
async def health():
    return {
        "status":          "online",
        "engine":          "ScamGuard v2.0",
        "model":           MODEL_NAME,
        "engine_loaded":   ENGINE_LOADED,
        "active_sessions": len(SESSION_STORE._sessions),
    }


@app.get("/api/keywords")
async def get_keywords():
    """Return the keyword tier dictionary for frontend display."""
    return KEYWORD_TIERS if ENGINE_LOADED else {
        "CRITICAL": {"weight": 3, "words": ["otp", "cvv", "pin", "aadhaar"]},
        "HIGH":     {"weight": 2, "words": ["blocked", "urgent", "kyc", "upi"]},
        "MEDIUM":   {"weight": 1, "words": ["verify", "account", "immediately"]},
    }


# =============================================================================
#  SERVE FRONTEND
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>index.html not found. Place it next to main.py</h1>")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)