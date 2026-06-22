from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-70b-versatile"

    # Tracing — SDK v3 reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"       # legacy alias
    langfuse_base_url: str = "https://cloud.langfuse.com"   # SDK v3 canonical
    tracing_enabled: bool = True

    # Vector store
    database_url: str = ""

    # ClinicalTrials.gov
    ctgov_api_base: str = "https://clinicaltrials.gov/api/v2"
    ctgov_page_size: int = 100
    ctgov_request_delay: float = 1.5  # seconds — stay under 50 req/min

    # Scope
    condition_class: str = "oncology"

    # Hugging Face
    hf_token: str = ""


settings = Settings()
