from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── Core ─────────────────────────────────────────────
    GROQ_API_KEY: str
    PINECONE_API_KEY: str 

    root_dir: Path
    model_name: str = "llama-3.3-70b-versatile"
    flow_path: str = "data/fever_flow.json"

    # ── Pinecone ─────────────────────────────────────────
    PINECONE_INDEX: str = "medical-diagnostic-agent"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_CLOUD: str = "aws"

    # ── Embeddings ───────────────────────────────────────
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    EMBED_DIM: int = 384

    # ── Retrieval ────────────────────────────────────────
    TOP_K_INITIAL: int = 10
    TOP_K_BM25: int = 20
    TOP_K_DENSE: int = 20
    TOP_K_MIN: int = 3
    RRF_K: int = 60

    # ── Agent ────────────────────────────────────────────
    MAX_TURNS: int = 8
    CONF_THRESHOLD: float = 0.75
    LLM_MODEL : str = "llama-3.3-70b-versatile"
    LLM_TEMP: float = 0.3

    # ── Dataset ──────────────────────────────────────────
    USMLE_JSONL_PATH: str = ""
    UPSERT_BATCH_SIZE: int = 100

    FUZZY_THRESHOLD: int = 70

    # ── Computed Paths (IMPORTANT) ───────────────────────
    @property
    def KNN_MODEL_PATH(self) -> Path:
        return self.root_dir / "first_doctor/training/ml-training/save-models/disease_predictor(KNN).joblib"

    @property
    def LABEL_ENCODER_PATH(self) -> Path:
        return self.root_dir / "first_doctor/training/ml-training/save-models/label_encoder(KNN).joblib"

    @property
    def DISEASE_DATASET_PATH(self) -> Path:
        return self.root_dir / "first_doctor/backend/data/Final_Augmented_dataset_Diseases_and_Symptoms.csv"

    @property
    def PRECAUTION_PATH(self) -> Path:
        return self.root_dir / "first_doctor/backend/data/Disease precaution.csv"


# ── Singleton instance ──────────────────────────────────
try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(
        "Config failed. Check your .env (especially GROQ_API_KEY & root_dir)."
    ) from e