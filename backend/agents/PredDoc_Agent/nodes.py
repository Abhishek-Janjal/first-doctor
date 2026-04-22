from __future__ import annotations

import ast
import json
import requests
import pandas as pd
import numpy as np
from datetime import date
from joblib import load

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from state import MedicalState
from config import settings
from fuzzywuzzy import process

LLM = ChatGroq(
    model=settings.model_name,
    temperature=0.3,
    api_key=settings.GROQ_API_KEY,
)

KNN_MODEL = load(settings.KNN_MODEL_PATH)
LABEL_ENCODER = load(settings.LABEL_ENCODER_PATH)

disease_df     = pd.read_csv(settings.DISEASE_DATASET_PATH)
precautions_df = pd.read_csv(settings.PRECAUTION_PATH)
KNOWN_SYMPTOMS: list[str] = list(disease_df.columns[1:])

# ── Internal helpers ──────────────────────────────────────────────────────────

def _llm_call(system: str, human: str) -> str:
    try:
        resp = LLM.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return resp.content.strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e


def _fuzzy_map(symptom: str) -> str | None:
    """Maps a raw symptom string to the nearest known symptom."""
    match, score = process.extractOne(symptom, KNOWN_SYMPTOMS)
    return match if score >= settings.FUZZY_THRESHOLD else None


def _parse_symptoms(raw: str) -> list[str]:
    """Safely coerce LLM list output → Python list."""
    try:
        start, end = raw.find("["), raw.rfind("]") + 1
        if start != -1 and end > start:
            return ast.literal_eval(raw[start:end])
    except Exception:
        pass
    cleaned = raw.replace("[", "").replace("]", "").replace('"', "").replace("'", "")
    return [s.strip() for s in cleaned.split(",") if s.strip()]


def _safe_json(raw: str) -> dict:
    """Extract and parse the first JSON object found in a string."""
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"No valid JSON found in LLM output:\n{raw}")


# ── Node 1 : Extract Symptoms ─────────────────────────────────────────────────

def extract_symptoms(state: MedicalState) -> dict:
    try:
        patient_input = state.get("patient_input", "")
        if not patient_input:
            return {"symptoms_list": [], "error": "No patient input provided"}
        system = (
            "You are a medical symptom extraction specialist. "
            "Your ONLY job is to read the patient input and extract a clean list of medical symptoms. "
            "Return ONLY a Python-style list of symptom strings. "
            'Example output: ["fever", "headache", "nausea"] '
            "Do NOT add any explanation, commentary, or extra text. Just the list."
        )
        raw = _llm_call(system, patient_input)
        symptoms = _parse_symptoms(raw)
        print(f"  [extract_symptoms] → {symptoms}")
        return {"symptoms_list": symptoms}
    except Exception as e:
        return {"symptoms_list": [], "error": f"Symptom extraction failed: {e}"}


# ── Node 2 : Predict Disease ──────────────────────────────────────────────────

def predict_disease(state: MedicalState) -> dict:
    """
    Node 2 — Disease Predictor
    Runs KNN model, then lets LLM refine the prediction.
    Returns ml_prediction and predicted_disease.
    """
    symptoms = state.get("symptoms_list", [])

    # ── KNN inference ────────────────────────────────────────────────────────
    mapped = [m for s in symptoms if (m := _fuzzy_map(s))]
    if mapped:
        vec = np.zeros(len(KNOWN_SYMPTOMS))
        for sym in mapped:
            vec[KNOWN_SYMPTOMS.index(sym)] = 1
        raw_pred   = KNN_MODEL.predict(vec.reshape(1, -1))
        ml_disease = LABEL_ENCODER.inverse_transform(raw_pred.reshape(-1, 1))[0][0]
    else:
        ml_disease = "Unknown — symptoms not recognised in dataset"

    print(f"  [predict_disease] KNN → {ml_disease}")

    # ── LLM refinement ───────────────────────────────────────────────────────
    system = (
        "You are a medical disease prediction expert. "
        "You receive a list of symptoms and a preliminary ML model prediction. "
        "Your job is to confirm or refine the predicted disease based on medical knowledge. "
        "Return ONLY the name of the most likely disease. No explanation needed."
    )
    human = (
        f"Symptoms: {symptoms}\n"
        f"ML Model Prediction: {ml_disease}\n"
        "Based on these symptoms and the ML prediction, what is the most likely disease?"
    )
    final_disease = _llm_call(system, human)
    print(f"  [predict_disease] LLM-refined → {final_disease}")

    return {"ml_prediction": str(ml_disease), "predicted_disease": final_disease}


# ── Node 3 : Suggest Precautions ──────────────────────────────────────────────

def suggest_precautions(state: MedicalState) -> dict:
    """
    Node 3 — Precaution Suggester
    Looks up CSV dataset, enriches with Wikipedia context, asks LLM.
    """
    disease = state.get("predicted_disease", "")

    # ── CSV lookup ───────────────────────────────────────────────────────────
    df = precautions_df.copy()
    df["Disease"] = df["Disease"].str.lower().str.strip()
    row = df[df["Disease"] == disease.lower().strip()]
    if not row.empty:
        csv_precautions = ", ".join(row.iloc[0, 1:].dropna().astype(str).tolist())
    else:
        csv_precautions = "No dataset entry found — using medical knowledge."

    system = (
        "You are a medical precautions specialist. "
        "You receive a predicted disease and dataset-based precautions. "
        "Provide a clear, numbered list of 5 key medical precautions the patient should follow. "
        "Be concise and practical. Use simple language the patient can understand."
    )
    human = (
        f"Predicted Disease: {disease}\n"
        f"Dataset Precautions: {csv_precautions}\n"
        "Provide 5 numbered precautions the patient must follow."
    )
    precautions = _llm_call(system, human)
    print(f"  [suggest_precautions] → {precautions[:80]}...")
    return {"precautions": precautions}


# ── Node 4 : Orchestrate Final Summary ───────────────────────────────────────

def orchestrate_summary(state: MedicalState) -> dict:
    """
    Node 4 — Healthcare Manager (Orchestrator)

    Uses the HoD_Agent's final_summary + symptoms dict (when available)
    alongside PredDoc predictions to compose the complete structured report.

    Returns:
        patient_summary  — structured intake summary (demographics, complaints, etc.)
        summary          — single-element list with the narrative paragraph
        clinical_support — indicators + 3 focus action items
    """

    # ── Decide which symptom source to use ───────────────────────────────────
    hod_summary   = state.get("hod_summary") or ""
    hod_symptoms  = state.get("hod_symptoms") or {}
    symptoms_list = state.get("symptoms_list", [])
    predicted_disease = state.get("predicted_disease", "")
    precautions   = state.get("precautions", "")

    # Build present / absent lists from HoD symptoms dict if available,
    # otherwise fall back to the PredDoc symptoms_list (all treated as present).
    if hod_symptoms:
        symptoms_present = [
            k.replace("_", " ").title()
            for k, v in hod_symptoms.items()
            if v == "present"
        ]
        symptoms_absent = [
            k.replace("_", " ").title()
            for k, v in hod_symptoms.items()
            if v == "absent"
        ]
    else:
        symptoms_present = [s.title() for s in symptoms_list]
        symptoms_absent  = []

    # ── Build rich context for LLM ────────────────────────────────────────────
    context_parts = []
    if hod_summary:
        context_parts.append(f"CLINICAL INTAKE SUMMARY (from HoD Agent):\n{hod_summary}")
    context_parts.append(f"Predicted Disease: {predicted_disease}")
    context_parts.append(f"Symptoms Present: {symptoms_present}")
    context_parts.append(f"Symptoms Absent: {symptoms_absent}")
    context_parts.append(f"Precautions:\n{precautions}")
    context = "\n\n".join(context_parts)

    system = (
        "You are a clinical AI documentation assistant.\n"
        "Return ONLY valid JSON — no markdown, no explanation, no extra text.\n\n"
        "Required JSON structure:\n"
        "{\n"
        '  "patient_summary": {\n'
        '    "demographics": "string — age/sex/relevant demographics or Not specified",\n'
        '    "chief_complaint": "string — primary presenting complaint",\n'
        '    "symptoms_present": ["string", ...],\n'
        '    "symptoms_absent": ["string", ...],\n'
        '    "duration_onset": "string — e.g. 5 days or Not specified",\n'
        '    "relevant_history": "string — prior conditions, meds, exposures"\n'
        "  },\n"
        '  "summary": [\n'
        '    "string — single concise narrative paragraph (2-4 sentences) summarising '
        'the full clinical picture"\n'
        "  ],\n"
        '  "clinical_support": {\n'
        '    "indicators": "string — 1-line pattern description (e.g. Patterns commonly associated with X / Y / Z).",\n'
        '    "focus": [\n'
        '      "string — actionable clinical check 1",\n'
        '      "string — actionable clinical check 2",\n'
        '      "string — actionable clinical check 3"\n'
        "    ]\n"
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Do NOT include patient name or ID in indicators\n"
        "- symptoms_present / symptoms_absent must be string arrays\n"
        "- focus must contain exactly 3 items\n"
        "- summary must be a single-element array with one narrative string\n"
        "- Output ONLY the JSON object"
    )

    response = _llm_call(system, context)
    print("  [orchestrate_summary] RAW LLM OUTPUT:", response[:200], "...")

    parsed = _safe_json(response)

    return {
        "clinical_support": parsed.get("clinical_support", {}),
        # Stash patient_summary and summary in clinical_support temporarily —
        # build_final_report will pull them out properly.
        "_patient_summary": parsed.get("patient_summary", {}),
        "_summary":         parsed.get("summary", []),
    }


# ── Node 5 : Build Final Report ───────────────────────────────────────────────

def build_final_report(state: MedicalState) -> dict:
    """
    Node 5 — Assembles the complete final report JSON.

    Merges:
      - patient_meta  (name, age, weight, height, id, date)
      - patient_summary
      - summary
      - vitals        (pass-through — untouched)
      - clinical_support

    The assembled dict is stored in state["final_report"].
    """
    patient_meta    = state.get("patient_meta") or {}
    vitals          = state.get("vitals") or []
    clinical_support = state.get("clinical_support") or {}

    # Extract the intermediate fields set by orchestrate_summary.
    # These are stored in state because LangGraph merges node returns into state.
    patient_summary = state.get("_patient_summary") or {}
    summary         = state.get("_summary") or []

    report = {
        "name":            patient_meta.get("name", ""),
        "age":             patient_meta.get("age", ""),
        "weight":          patient_meta.get("weight", ""),
        "height":          patient_meta.get("height", ""),
        "id":              patient_meta.get("id", ""),
        "date":            patient_meta.get("date", str(date.today())),
        "patient_summary": patient_summary,
        "summary":         summary,
        "vitals":          vitals,          # Never touched — future Vital Agent owns this
        "clinical_support": clinical_support,
    }

    print("  [build_final_report] Report assembled successfully.")
    return {"final_report": report}
