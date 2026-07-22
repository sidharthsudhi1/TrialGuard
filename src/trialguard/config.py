from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Tracing — SDK v3 reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"       # legacy alias
    langfuse_base_url: str = "https://cloud.langfuse.com"   # SDK v3 canonical
    tracing_enabled: bool = True

    # Vector store
    database_url: str = ""
    # ivfflat probes: default 1 scans a single list and loses ~64% recall vs exact
    # (measured, phase5_vectorstore). Higher probes recovers recall at near-flat
    # latency on the free tier; tune toward sqrt(lists) for the production corpus.
    pgvector_probes: int = 20

    # ClinicalTrials.gov
    ctgov_api_base: str = "https://clinicaltrials.gov/api/v2"
    ctgov_page_size: int = 100
    ctgov_request_delay: float = 1.5  # seconds — stay under 50 req/min

    # Scope
    condition_class: str = "oncology"

    # Hugging Face
    hf_token: str = ""


settings = Settings()
