import json
import re
import os
import time
import logging
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec

from langgraph.graph import StateGraph, END

from config import settings
from state import AgentState
from .nodes import (
    extract_symptoms,
    retrieve_and_rank,
    ask_question,
    conclude,
    set_retriever,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset loaders
# ═════════════════════════════════════════════════════════════════════════════

def _load_gretel() -> List[Dict]:
    """
    gretelai/symptom_to_diagnosis
    Patient-voice descriptions with disease labels.
    Primary retrieval source — closest to real patient input.
    """
    logger.info("Loading gretelai/symptom_to_diagnosis ...")
    ds = load_dataset("gretelai/symptom_to_diagnosis", split="train")

    cases = []
    for row in ds:
        cases.append({
            "id":             f"gretel_{len(cases)}",
            "source":         "gretel",
            "retrieval_text": row["input_text"],
            "answer":         row["output_text"],
            "question":       row["input_text"],
            "options":        {},
        })

    logger.info(f"  ✓ {len(cases)} records from Gretel")
    return cases


def _load_mts_dialog() -> List[Dict]:
    """
    abachaa/MTS-Dialog (with har1/MTS_Dialogue-Clinical_Note as fallback)
    Real doctor-patient conversations.
    Only patient turns are stored as retrieval_text.
    """
    logger.info("Loading MTS-Dialog ...")

    ds = None
    for hf_id in ("abachaa/MTS-Dialog", "har1/MTS_Dialogue-Clinical_Note"):
        try:
            ds = load_dataset(hf_id, split="train")
            logger.info(f"  Loaded from {hf_id}")
            break
        except Exception as e:
            logger.warning(f"  {hf_id} unavailable: {e}")

    if ds is None:
        logger.warning("  Skipping MTS-Dialog — Gretel alone will be used.")
        return []

    cases = []
    for row in ds:
        dialogue = row.get("dialogue", row.get("conversation", ""))
        note     = row.get("section_text", row.get("note", ""))
        if not dialogue:
            continue

        patient_lines = [
            line[8:].strip()
            for line in dialogue.split("\n")
            if line.lower().startswith("patient:")
        ]
        retrieval_text = " ".join(patient_lines) if patient_lines else dialogue

        answer = "Unknown"
        for seg in ("cc:", "chief complaint:", "assessment:", "diagnosis:"):
            if seg in note.lower():
                idx    = note.lower().index(seg) + len(seg)
                answer = note[idx: idx + 120].split("\n")[0].strip()
                break

        cases.append({
            "id":             f"mts_{len(cases)}",
            "source":         "mts_dialog",
            "retrieval_text": retrieval_text,
            "answer":         answer,
            "question":       dialogue,
            "options":        {},
        })

    logger.info(f"  ✓ {len(cases)} records from MTS-Dialog")
    return cases


def _load_usmle(path: str) -> List[Dict]:
    """Optional USMLE dev.jsonl — clinical accuracy anchor."""
    if not path or not os.path.exists(path):
        logger.info("  USMLE path not set or not found — skipping.")
        return []

    cases = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            cases.append({
                "id":             f"usmle_{len(cases)}",
                "source":         "usmle",
                "retrieval_text": row["question"],
                "answer":         row["answer"],
                "question":       row["question"],
                "options":        row.get("options", {}),
            })

    logger.info(f"  ✓ {len(cases)} records from USMLE")
    return cases


def load_all_cases() -> List[Dict]:
    cases = (
        _load_gretel()
        + _load_mts_dialog()
        + _load_usmle(settings.USMLE_JSONL_PATH)
    )
    logger.info(f"Total cases loaded: {len(cases)}")
    return cases


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HybridRetriever
# ═════════════════════════════════════════════════════════════════════════════

class HybridRetriever:
    """
    Combines BM25 sparse retrieval with Pinecone dense (cosine) retrieval.
    Merges results using Reciprocal Rank Fusion (RRF).

    Usage
    -----
    retriever = HybridRetriever(cases)
    results   = retriever.hybrid_retrieve("fever and knee pain", k=10)
    """

    def __init__(self, cases: List[Dict]):
        self.cases      = cases
        self.id_to_idx  = {c["id"]: i for i, c in enumerate(cases)}
        texts           = [c["retrieval_text"] for c in cases]

        # ── Dense encoder ─────────────────────────────────────────────────────
        logger.info(f"Loading embedding model: {settings.EMBED_MODEL}")
        self.embedder = SentenceTransformer(settings.EMBED_MODEL)

        # ── BM25 sparse index ─────────────────────────────────────────────────
        logger.info("Building BM25 index ...")
        corpus_tokens = [self._tokenize(t) for t in texts]
        self.bm25     = BM25Okapi(corpus_tokens)
        logger.info(f"  BM25 ready — {len(corpus_tokens)} docs")

        # ── Pinecone dense index ──────────────────────────────────────────────
        self.pine_index = self._init_pinecone(texts)

    # ── Tokenisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Medical-aware tokeniser for BM25.
        Keeps hyphenated terms intact (e.g. 'non-productive', 'well-defined').
        """
        tokens = re.findall(r"[a-z][a-z\-]*[a-z]|[a-z]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    # ── Pinecone setup ────────────────────────────────────────────────────────

    def _init_pinecone(self, texts: List[str]):
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        existing = [idx.name for idx in pc.list_indexes()]
        if settings.PINECONE_INDEX not in existing:
            logger.info(f"Creating Pinecone index '{settings.PINECONE_INDEX}' ...")
            pc.create_index(
                name      = settings.PINECONE_INDEX,
                dimension = settings.EMBED_DIM,
                metric    = "cosine",
                spec      = ServerlessSpec(
                    cloud  = settings.PINECONE_CLOUD,
                    region = settings.PINECONE_REGION,
                ),
            )
            while not pc.describe_index(settings.PINECONE_INDEX).status["ready"]:
                logger.info("  Waiting for Pinecone index to be ready ...")
                time.sleep(3)
            logger.info("  Index created.")

        index        = pc.Index(settings.PINECONE_INDEX)
        current_count = index.describe_index_stats()["total_vector_count"]

        if current_count < len(self.cases):
            self._upsert(index, texts)
        else:
            logger.info(
                f"Pinecone already has {current_count} vectors — skipping upsert."
            )

        return index

    def _upsert(self, index, texts: List[str]) -> None:
        logger.info(f"Embedding and upserting {len(texts)} vectors ...")
        embeddings = self.embedder.encode(
            texts,
            batch_size          = 128,
            show_progress_bar   = True,
            normalize_embeddings= True,
            convert_to_numpy    = True,
        )

        vectors = [
            (
                self.cases[i]["id"],
                embeddings[i].tolist(),
                {
                    "case_id":        self.cases[i]["id"],
                    "source":         self.cases[i]["source"],
                    "answer":         self.cases[i]["answer"][:200],
                    "retrieval_text": self.cases[i]["retrieval_text"][:500],
                },
            )
            for i in range(len(self.cases))
        ]

        for start in tqdm(
            range(0, len(vectors), settings.UPSERT_BATCH_SIZE), desc="Upserting"
        ):
            index.upsert(vectors=vectors[start : start + settings.UPSERT_BATCH_SIZE])

        time.sleep(2)
        final = index.describe_index_stats()["total_vector_count"]
        logger.info(f"  Upsert complete — Pinecone now has {final} vectors.")

    # ── Individual retrievers ─────────────────────────────────────────────────

    def bm25_retrieve(self, query: str, k: int = settings.TOP_K_BM25) -> List[Tuple[float, Dict]]:
        """BM25 sparse retrieval. Returns [(score, case), ...] sorted desc."""
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores  = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.cases[i]) for i in top_idx if scores[i] > 0]

    def dense_retrieve(self, query: str, k: int = settings.TOP_K_DENSE) -> List[Tuple[float, Dict]]:
        """
        Dense cosine retrieval via Pinecone.
        Returns [(score, case), ...] sorted desc.
        """
        q_emb  = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        result = self.pine_index.query(
            vector           = q_emb,
            top_k            = k,
            include_metadata = True,
        )
        hits = []
        for match in result["matches"]:
            case_id = match["id"]
            if case_id in self.id_to_idx:
                idx = self.id_to_idx[case_id]
                hits.append((float(match["score"]), self.cases[idx]))
        return hits

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_hits:  List[Tuple[float, Dict]],
        dense_hits: List[Tuple[float, Dict]],
        top_n:      int,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion:
            RRF(doc) = Σ  1 / (k + rank)
        Docs ranking well in both lists naturally rise to the top.
        """
        rrf: Dict[str, float] = {}

        for rank, (_, case) in enumerate(bm25_hits):
            cid      = case["id"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (settings.RRF_K + rank + 1)

        for rank, (_, case) in enumerate(dense_hits):
            cid      = case["id"]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (settings.RRF_K + rank + 1)

        sorted_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)

        results = []
        for cid in sorted_ids[:top_n]:
            idx  = self.id_to_idx[cid]
            case = dict(self.cases[idx])
            case["_rrf_score"] = round(rrf[cid], 6)
            results.append(case)

        return results

    # ── Public interface ──────────────────────────────────────────────────────

    def hybrid_retrieve(self, query: str, k: int = settings.TOP_K_INITIAL) -> List[Dict]:
        """Main entry point — runs BM25 + dense and fuses with RRF."""
        bm25_hits  = self.bm25_retrieve(query,  k=settings.TOP_K_BM25)
        dense_hits = self.dense_retrieve(query, k=settings.TOP_K_DENSE)
        return self._rrf_fuse(bm25_hits, dense_hits, top_n=k)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Router
# ═════════════════════════════════════════════════════════════════════════════

def _route(state: AgentState) -> str:
    """
    Conditional edge function: decides whether to ask another question
    or produce the final conclusion.
    """
    n_cands     = len(state["candidates"])
    n_questions = len(state["questions_asked"])
    confidence  = state["confidence"]

    logger.debug(
        f"[router] candidates={n_cands} | questions={n_questions} | confidence={confidence:.2f}"
    )

    if n_cands     <= settings.TOP_K_MIN:       return "conclude"
    if confidence  >= settings.CONF_THRESHOLD:  return "conclude"
    if n_questions >= settings.MAX_TURNS:       return "conclude"

    return "ask"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Graph builder
# ═════════════════════════════════════════════════════════════════════════════

def build_graph(retriever: HybridRetriever):
    """
    Wires up the LangGraph and returns a compiled graph ready for invocation.

    Graph topology
    --------------
    extract ──► retrieve ──┬──► ask ──► END   (loop: outer caller re-invokes)
                           └──► conclude ──► END
    """
    # Inject retriever into the node layer
    set_retriever(retriever)

    builder = StateGraph(AgentState)

    builder.add_node("extract",  extract_symptoms)
    builder.add_node("retrieve", retrieve_and_rank)
    builder.add_node("ask",      ask_question)
    builder.add_node("conclude", conclude)

    builder.set_entry_point("extract")
    builder.add_edge("extract", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        _route,
        {"ask": "ask", "conclude": "conclude"},
    )
    # Graph pauses at END after each question — the FastAPI layer
    # re-invokes with the updated state on the next user turn.
    builder.add_edge("ask",      END)
    builder.add_edge("conclude", END)

    return builder.compile()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Startup factory (called once by main.py)
# ═════════════════════════════════════════════════════════════════════════════

_retriever_singleton: HybridRetriever | None = None
_graph_singleton = None


def get_retriever() -> HybridRetriever:
    return _retriever_singleton


def initialise() -> tuple:
    """
    Loads datasets, builds retriever and graph.
    Called once during FastAPI lifespan startup.
    Returns (retriever, compiled_graph).
    """
    global _retriever_singleton, _graph_singleton

    if _graph_singleton is not None:
        return _retriever_singleton, _graph_singleton

    cases                 = load_all_cases()
    _retriever_singleton  = HybridRetriever(cases)
    _graph_singleton      = build_graph(_retriever_singleton)

    logger.info("Pipeline initialised.")
    return _retriever_singleton, _graph_singleton