import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from config import settings
from state import AgentState

if TYPE_CHECKING:
    # Retriever is injected at runtime by pipeline.py to avoid circular imports
    pass


# ── LLM singleton (shared across all nodes) ───────────────────────────────────

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMP,
            api_key=settings.GROQ_API_KEY,
        )
    return _llm


def _call_llm(system: str, user: str) -> str:
    resp = get_llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return resp.content.strip()


# ── Retriever accessor — set by pipeline.py after index is ready ──────────────

_retriever = None


def set_retriever(retriever) -> None:
    """Called once during startup to inject the retriever into the node layer."""
    global _retriever
    _retriever = retriever


def _get_retriever():
    if _retriever is None:
        raise RuntimeError(
            "Retriever not initialised. Call set_retriever() before invoking nodes."
        )
    return _retriever


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — extract_symptoms
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACT_SYS = """
You are a clinical NLP system. Extract ALL medical symptoms, signs, history,
and demographics from the conversation.

Return ONLY a flat JSON object — no nesting, no explanation:
{"fever": "present", "knee_pain": "present", "cough": "absent", "age": "21", "sex": "male"}

Rules:
- snake_case keys
- Values: "present", "absent", or a literal string (age, duration, severity)
- Include ALL previously known symptoms — carry them over, do not drop any
- Output ONLY the JSON object
"""


def extract_symptoms(state: AgentState) -> AgentState:
    """
    Parses the last few turns and updates state['symptoms'].
    Merges new findings into the existing symptom dict.
    """
    recent    = state["messages"][-4:]
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

    user_prompt = f"""
Already known symptoms:
{json.dumps(state["symptoms"], indent=2)}

Conversation:
{conv_text}

Return updated JSON with ALL symptoms (carry over old + add new).
"""
    raw = _call_llm(_EXTRACT_SYS, user_prompt)
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        extracted = json.loads(raw)
        if isinstance(extracted, dict):
            state["symptoms"].update(extracted)
    except json.JSONDecodeError:
        # Attempt to salvage a partial JSON substring
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                state["symptoms"].update(json.loads(match.group()))
            except Exception:
                pass  # keep existing symptoms unchanged

    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — retrieve_and_rank
# ─────────────────────────────────────────────────────────────────────────────

def _symptom_rescore(case: dict, symptoms: dict) -> float:
    """
    Re-score a retrieved candidate against confirmed/absent symptoms.
    Starts from the RRF score (scaled) and adds a symptom overlap bonus.
    """
    text      = (case["retrieval_text"] + " " + case.get("answer", "")).lower()
    base      = case.get("_rrf_score", 0.0) * 10   # scale RRF to comparable range
    sym_score = 0.0

    for sym, val in symptoms.items():
        sym_clean = sym.replace("_", " ")
        if val == "present":
            sym_score += 1.5 if sym_clean in text else 0.0
        elif val == "absent":
            sym_score -= 0.5 if sym_clean in text else 0.0

    return base + sym_score


def retrieve_and_rank(state: AgentState) -> AgentState:
    """
    Turn 1  — builds a query from confirmed symptoms + raw user text,
              runs hybrid BM25+Pinecone retrieval, populates candidates.
    Turn 2+ — re-scores existing candidates against latest symptoms,
              drops bottom tier to narrow K progressively.
    """
    retriever = _get_retriever()
    symptoms  = state["symptoms"]

    if not state["candidates"]:
        # ── First turn: full hybrid retrieval ────────────────────────────────
        present_syms = " ".join(
            k.replace("_", " ") for k, v in symptoms.items() if v == "present"
        )
        last_user = next(
            (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
            "",
        )
        query = f"{present_syms} {last_user}".strip() or last_user
        state["candidates"] = retriever.hybrid_retrieve(query, k=settings.TOP_K_INITIAL)

    # ── Re-score all current candidates ──────────────────────────────────────
    scored = [
        (_symptom_rescore(c, symptoms), c)
        for c in state["candidates"]
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    scores = [s for s, _ in scored]
    cands  = [c for _, c in scored]

    # ── Compute confidence: normalised gap between rank-1 and rank-2 ─────────
    if len(scores) >= 2 and scores[0] > 0:
        gap = scores[0] - scores[1]
        state["confidence"] = min(gap / (abs(scores[0]) + 1e-9), 1.0)
    elif len(scores) == 1:
        state["confidence"] = 1.0
    else:
        state["confidence"] = 0.0

    # ── Narrow: drop candidates below 40% of top score (after turn 1) ────────
    if state["turn_count"] > 1 and scores and scores[0] > 0:
        threshold = scores[0] * 0.4
        filtered  = [(s, c) for s, c in zip(scores, cands) if s >= threshold]
        if len(filtered) >= settings.TOP_K_MIN:
            scores = [s for s, _ in filtered]
            cands  = [c for _, c in filtered]

    state["candidates"]       = cands
    state["candidate_scores"] = scores

    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — ask_question
# ─────────────────────────────────────────────────────────────────────────────

_QUESTION_SYS = """
You are a clinical intake AI. Your goal is to narrow a differential diagnosis
by asking the patient focused, discriminating questions.

You will receive:
- Known symptoms extracted so far
- The current top candidate diagnoses
- Questions already asked (do NOT repeat these)

Task: Ask the SINGLE most discriminating question that best separates the
top candidates from each other.

Rules:
- ONE question only — no preamble, no explanation
- Patient-friendly language — no jargon
- Prefer yes/no or short-answer format
- Output ONLY the question text
"""


def ask_question(state: AgentState) -> AgentState:
    """
    Generates the next question using the current candidate set as context.
    Appends the question to messages and questions_asked.
    """
    cand_labels  = [c["answer"][:80] for c in state["candidates"][:6]]
    cand_summary = "\n".join(f"- {l}" for l in cand_labels)

    user_prompt = f"""
Known symptoms:
{json.dumps(state["symptoms"], indent=2)}

Current candidate diagnoses:
{cand_summary}

Questions already asked:
{chr(10).join(state["questions_asked"]) if state["questions_asked"] else "None yet"}

Ask the single best next question.
"""
    question = _call_llm(_QUESTION_SYS, user_prompt)
    state["messages"].append({"role": "assistant", "content": question})
    state["questions_asked"].append(question)

    return state


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — conclude
# ─────────────────────────────────────────────────────────────────────────────

_CONCLUDE_SYS = """
You are a clinical documentation AI. Produce a structured intake summary
for a downstream diagnostic model.

Output exactly this format:

PATIENT SUMMARY
---------------
Demographics      : ...
Chief complaint   : ...
Symptoms present  : ...
Symptoms absent   : ...
Duration / onset  : ...
Relevant history  : ...

TOP DIFFERENTIAL (retrieval-ranked):
1. ...
2. ...
3. ...

⚠️  Intake data only. Requires physician review.
"""


def conclude(state: AgentState) -> AgentState:
    """
    Emits a structured clinical summary and sets state['done'] = True.
    The final_summary field is the handoff payload to the predictor agent.
    """
    conv     = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in state["messages"])
    top3     = state["candidates"][:3]
    top3_txt = "\n".join(f"- {c['answer']}" for c in top3)

    user_prompt = f"""
Conversation:
{conv}

Extracted symptoms:
{json.dumps(state["symptoms"], indent=2)}

Top retrieved candidates:
{top3_txt}
"""

    summary               = _call_llm(_CONCLUDE_SYS, user_prompt)
    state["final_summary"] = summary
    state["messages"].append({"role": "assistant", "content": summary})
    state["done"]          = True

    return state