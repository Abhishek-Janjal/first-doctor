import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from jinja2 import FileSystemLoader
import pdfkit
from state import ( AssessmentRequest, 
                   AssessmentResponse, 
                   ErrorResponse,
                   StartResponse,
                   MessageRequest,
                   MessageResponse,
                   StateResponse,
                   SummaryResponse,
                   PPGRequest,)
from agents.PredDoc_Agent import medical_assessment_graph
from agents.HoD_Agent import initialise
from agents.Vitals_Agent.pipeline import run_vitals_pipeline
import time
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict
from state import AgentState, fresh_state

from config import settings
from jinja2 import Environment, FileSystemLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# App-level singletons — populated during lifespan startup
# ═════════════════════════════════════════════════════════════════════════════

_graph     = None
_retriever = None

# In-memory session store: {session_id: AgentState}
# Swap for Redis / DB in production
_sessions: Dict[str, AgentState] = {}


# ═════════════════════════════════════════════════════════════════════════════
# Lifespan — runs dataset loading + index build before serving traffic
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _retriever
    logger.info("Starting up — loading datasets and building retrieval index ...")
    _retriever, _graph = initialise()
    logger.info("Startup complete. Ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="HealthFlow AI",
    description="Multi-agent medical assessment pipeline powered by LangGraph + Llama 3.3",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # (e.g. your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _require_session(session_id: str) -> AgentState:
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return state


def _last_agent_message(state: AgentState) -> str:
    return next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "assistant"),
        "",
    )


def _top_candidates(state: AgentState, n: int = 5) -> list:
    return [
        {
            "answer": c["answer"],
            "source": c.get("source", "unknown"),
            "score":  round(s, 4),
        }
        for c, s in zip(state["candidates"][:n], state["candidate_scores"][:n])
    ]

# ═════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    """Quick liveness check."""
    return {"status": "ok", "service": "HealthFlow AI"}


@app.get("/health", tags=["Health"])
def health():
    """Confirms graph is loaded and ready."""
    return {
        "status":      "ok",
        "graph_ready": medical_assessment_graph is not None,
    }


@app.post(
    "/assess",
    response_model=AssessmentResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Assessment"],
    summary="Run a full medical assessment for a patient",
)
def run_assessment(request: AssessmentRequest):
    """
    Accepts a free-text symptom description and patient name.
    Runs the full 5-node LangGraph pipeline and returns the structured report.
    """
    start = time.perf_counter()

    try:
        result: dict = medical_assessment_graph.invoke({
            "patient_name":  request.patient_name,
            "patient_input": request.patient_input,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    elapsed = round(time.perf_counter() - start, 2)

    return AssessmentResponse(
        patient_name      = result.get("patient_name", request.patient_name),
        symptoms_list     = result.get("symptoms_list", []),
        ml_prediction     = result.get("ml_prediction", ""),
        predicted_disease = result.get("predicted_disease", ""),
        clinical_support  = result.get("clinical_support", {}),
        processing_time_s = elapsed,
    )

@app.get("/retriever/stats")
async def retriever_stats():
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready.")
    stats = _retriever.pine_index.describe_index_stats()
    return {
        "total_vectors": stats["total_vector_count"],
        "dimension":     stats.get("dimension"),
        "index_name":    settings.PINECONE_INDEX,
    }


@app.post("/session/start", response_model=StartResponse)
async def start_session():
    """
    Creates a new intake session and returns the opening greeting.
    The agent doesn't invoke the graph yet — that happens on the first /message.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent not ready yet.")

    session_id           = str(uuid.uuid4())
    state                = fresh_state()
    greeting             = "Hello! I'm here to help understand your symptoms. Please describe what's been bothering you today."
    state["messages"].append({"role": "assistant", "content": greeting})
    _sessions[session_id] = state

    logger.info(f"Session started: {session_id}")
    return StartResponse(session_id=session_id, agent_message=greeting)


@app.post("/session/{session_id}/message", response_model=MessageResponse)
async def send_message(session_id: str, body: MessageRequest):
    """
    Accepts a user message, runs one turn of the graph, returns the agent reply.

    The graph processes: extract → retrieve → (ask | conclude) → END.
    State is persisted in _sessions between turns.
    """
    state = _require_session(session_id)

    if state["done"]:
        raise HTTPException(
            status_code=400,
            detail="Session is complete. Retrieve the summary at GET /session/{id}/summary.",
        )

    if not body.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    # Append user turn and increment counter
    state["messages"].append({"role": "user", "content": body.message.strip()})
    state["turn_count"] += 1

    # Run one graph turn — graph pauses at END after ask or conclude
    updated_state             = _graph.invoke(state)
    _sessions[session_id]     = updated_state

    agent_message = _last_agent_message(updated_state)

    logger.info(
        f"[{session_id}] turn={updated_state['turn_count']} "
        f"candidates={len(updated_state['candidates'])} "
        f"confidence={updated_state['confidence']:.2f} "
        f"done={updated_state['done']}"
    )

    return MessageResponse(
        session_id      = session_id,
        agent_message   = agent_message,
        turn_count      = updated_state["turn_count"],
        candidate_count = len(updated_state["candidates"]),
        confidence      = round(updated_state["confidence"], 3),
        done            = updated_state["done"],
    )


@app.get("/session/{session_id}/state", response_model=StateResponse)
async def get_session_state(session_id: str):
    """Debug endpoint — full internal state snapshot."""
    state = _require_session(session_id)

    present = [k for k, v in state["symptoms"].items() if v == "present"]
    absent  = [k for k, v in state["symptoms"].items() if v == "absent"]

    return StateResponse(
        session_id        = session_id,
        turn_count        = state["turn_count"],
        confidence        = round(state["confidence"], 3),
        candidate_count   = len(state["candidates"]),
        top_candidates    = _top_candidates(state),
        symptoms_present  = present,
        symptoms_absent   = absent,
        questions_asked   = state["questions_asked"],
        done              = state["done"],
    )


@app.get("/session/{session_id}/summary", response_model=SummaryResponse)
async def get_summary(session_id: str):
    """
    Returns the structured clinical summary once the session is complete.
    This is the handoff payload for the downstream predictor agent.
    """
    state = _require_session(session_id)

    if not state["done"]:
        raise HTTPException(
            status_code=400,
            detail="Session is not complete yet. Continue sending messages.",
        )

    return SummaryResponse(
        session_id     = session_id,
        final_summary  = state["final_summary"] or "",
        top_candidates = _top_candidates(state, n=3),
        symptoms       = state["symptoms"],
    )

@app.post("/analyze_ppg")
async def analyze_ppg(req: PPGRequest):
    try:
        result = run_vitals_pipeline(
            ppg_signal=req.signal,
            ppg_fps=req.fps,
            audio_transcript="",
            audio_duration=0.0,
        )

        if result.get("error"):
            return {"status": "error", "message": result["error"]}

        return {
            "status": "ok",
            "heart_rate": result.get("heart_rate"),
            "hrv_rmssd": result.get("hrv_rmssd"),
            "spo2_est": result.get("spo2_est"),
            "ppg_quality": result.get("ppg_quality"),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@app.post("/generate_report")
async def generate_report():
    try:
        # Load JSON data
        with open("data.json") as f:
            data = json.load(f)
        
        # Setup Jinja2 environment and render template
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("template.html")
        rendered = template.render(**data)

        # PDF config and generation
        config = pdfkit.configuration(wkhtmltopdf="/usr/local/bin/wkhtmltopdf")
        pdfkit.from_string(rendered, "patient_report.pdf", configuration=config)

        # Return json data so frontend can build a dashboard
        return {
            "status": "success",
            "data": data,
            "download_url": "http://localhost:8000/download_report"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/download_report")
async def download_report():
    # Return the generated PDF
    return FileResponse("patient_report.pdf", media_type="application/pdf", filename="patient_report.pdf")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    _require_session(session_id)
    del _sessions[session_id]
    logger.info(f"Session deleted: {session_id}")
    return {"deleted": session_id}