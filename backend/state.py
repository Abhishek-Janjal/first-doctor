

from __future__ import annotations
from typing import Any, Dict, Optional, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ── Request / Response models ─────────────────────────────────────────────────

class AssessmentRequest(BaseModel):
    patient_name:  str = Field(..., min_length=1, example="Sara")
    patient_input: str = Field(..., min_length=10, example="I have fever, headache and chills for 3 days.")


class AssessmentResponse(BaseModel):
    patient_name:      str
    symptoms_list:     list[str]
    ml_prediction:     str
    predicted_disease: str
    clinical_support: dict
    processing_time_s: float

class ErrorResponse(BaseModel):
    detail: str
    stage:  Optional[str] = None



class MedicalState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    patient_name:      str           # e.g. "Sara"
    patient_input:     str           # raw free-text from patient

    # ── Intermediate ──────────────────────────────────────────────────────────
    symptoms_list:     list[str]     # extracted symptom tokens
    ml_prediction:     str           # raw KNN output before LLM refinement
    predicted_disease: str           # final disease name (LLM-refined)
    api_disease_info:  str           # Wikipedia summary
    precautions:       str           # numbered precaution list

    # ── HoD bridge inputs ─────────────────────────────────────────────────────
    hod_summary:       Optional[str]  # HoD final_summary text handed off here
    hod_symptoms:      Optional[dict] # HoD symptoms dict {symptom: "present"/"absent"}
    patient_meta:      Optional[dict] # {name, age, weight, height, id, date}
    vitals:            Optional[list] # Pass-through — populated by future Vital Agent

    # ── Orchestration intermediates (orchestrate_summary → build_final_report) ─
    _patient_summary:  Optional[dict] # patient_summary block from orchestrate_summary
    _summary:          Optional[list] # summary block (single-element list of paragraph)

    # ── Output ────────────────────────────────────────────────────────────────
    clinical_support:  dict          # structured output from orchestrate_summary
    final_summary:     str           # full patient-facing medical report
    final_report:      Optional[dict] # complete assembled output JSON

    # ── Error slot (any node can write here) ──────────────────────────────────
    error:             Optional[str]


class AgentState(TypedDict):
   
    messages: List[Dict]                # Full conversation history as plain dicts {role, content}
    symptoms: Dict[str, str]            # Dynamically extracted symptoms: {symptom_name: "present"/"absent"/literal_value}
    candidates: List[Dict]              # Current narrowed candidate cases — shrinks each turn
    candidate_scores: List[float]       # Symptom re-scores, parallel list to candidates
    questions_asked: List[str]          # Questions the agent has already asked — prevents repeats
    turn_count: int                     # Number of completed user turns
    confidence: float                   # 0.0–1.0 confidence derived from gap between top-1 and top-2 candidate scores
    done: bool                          # True once the graph has reached a conclusion
    final_summary: Optional[str]        # Structured clinical summary emitted at conclusion — consumed by predictor

class StartResponse(BaseModel):
    session_id:    str
    agent_message: str


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    session_id:    str
    agent_message: str
    turn_count:    int
    candidate_count: int
    confidence:    float
    done:          bool


class StateResponse(BaseModel):
    session_id:       str
    turn_count:       int
    confidence:       float
    candidate_count:  int
    top_candidates:   list
    symptoms_present: list
    symptoms_absent:  list
    questions_asked:  list
    done:             bool


class SummaryResponse(BaseModel):
    session_id:    str
    final_summary: str
    top_candidates: list
    symptoms:      dict


class ReportRequest(BaseModel):
    """Body sent to POST /session/{session_id}/report."""
    name:   str   = Field(..., example="John Doe")
    age:    Any   = Field(..., example=45)
    weight: str   = Field(..., example="70 kg")
    height: str   = Field(..., example="5'10")
    id:     str   = Field(..., example="52879")
    date:   str   = Field(..., example="2026-04-18")
    language: str = Field(default="en", description="Target language code for the generated report (en, hi, mr)")
    vitals: list  = Field(default_factory=list,
                          description="Populated by Vital Agent — pass-through only")


class FinalReportResponse(BaseModel):
    """Complete structured medical report returned by PredDoc_Agent."""
    name:            str
    age:             Any
    weight:          str
    height:          str
    id:              str
    date:            str
    patient_summary: dict
    summary:         list
    vitals:          list
    clinical_support: dict



def fresh_state() -> AgentState:
    """Return a blank state for a new session."""
    return {
        "messages":          [],
        "symptoms":          {},
        "candidates":        [],
        "candidate_scores":  [],
        "questions_asked":   [],
        "turn_count":        0,
        "confidence":        0.0,
        "done":              False,
        "final_summary":     None,
    }


class VitalsState(TypedDict):
    # inputs
    ppg_signal: Optional[list[float]]
    ppg_fps: Optional[float]
    audio_transcript: Optional[str]
    audio_duration: Optional[float]

    # ppg outputs
    heart_rate: Optional[float]
    hrv_rmssd: Optional[float]
    spo2_est: Optional[float]
    ppg_quality: Optional[str]

    # audio outputs
    respiratory_class: Optional[str]
    resp_confidence: Optional[float]
    resp_note: Optional[str]

    # final
    clinical_summary: Optional[str]
    error: Optional[str]
    
class PPGRequest(BaseModel):
    signal: list[float]
    fps: float