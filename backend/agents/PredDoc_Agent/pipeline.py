"""
graph/pipeline.py
-----------------
Assembles and compiles the medical assessment StateGraph.

Usage (standalone):
    from graph import medical_assessment_graph
    result = medical_assessment_graph.invoke({
        "patient_name": "Sara",
        "patient_input": "I have fever, headache and chills..."
    })

Usage (as a subgraph node inside a parent LangGraph):
    from graph import medical_assessment_graph

    parent = StateGraph(ParentState)
    parent.add_node("medical_assessment", medical_assessment_graph)
    # The compiled subgraph is a valid node — LangGraph handles I/O mapping.
    # Use an input/output schema or add_node with state key mapping if needed.
"""

from langgraph.graph import StateGraph, END

from state import MedicalState
from .nodes import (
    extract_symptoms,
    predict_disease,
    suggest_precautions,
    orchestrate_summary,
    build_final_report,
)


def build_graph() -> StateGraph:
    """
    Constructs the medical assessment graph.

    Linear pipeline (no branching needed here — every step is required):

        extract_symptoms
             ↓
        predict_disease
             ↓
        suggest_precautions  ← CSV + LLM
             ↓
        orchestrate_summary  ← Full structured JSON (patient_summary, summary, clinical_support)
             ↓
        build_final_report   ← Merge patient_meta + vitals → final_report
             ↓
             END

    Returns the compiled graph (a Runnable — supports .invoke / .stream).
    """
    builder = StateGraph(MedicalState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("extract_symptoms",    extract_symptoms)
    builder.add_node("predict_disease",     predict_disease)
    builder.add_node("suggest_precautions", suggest_precautions)
    builder.add_node("orchestrate_summary", orchestrate_summary)
    builder.add_node("build_final_report",  build_final_report)

    # ── Wire edges (linear) ───────────────────────────────────────────────────
    builder.set_entry_point("extract_symptoms")
    builder.add_edge("extract_symptoms",    "predict_disease")
    builder.add_edge("predict_disease",     "suggest_precautions")
    builder.add_edge("suggest_precautions", "orchestrate_summary")
    builder.add_edge("orchestrate_summary", "build_final_report")
    builder.add_edge("build_final_report",  END)

    return builder


# Pre-compiled singleton — import this anywhere
medical_assessment_graph = build_graph().compile()
