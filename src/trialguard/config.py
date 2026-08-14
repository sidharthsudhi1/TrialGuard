from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Provider selection (Phase 8). DeepInfra by default since the WS-3 parity
    # gate passed: FP8 quantization left citation precision unchanged (0.9057 ->
    # 0.9086 on the matched 180-trial baseline arm) and cleared every committed
    # regression floor. See data/reports/phase8_provider_parity.md.
    # Groq stays fully runnable — it reproduces Phase 3/4 from cache.
    llm_provider: str = "deepinfra"
    deepinfra_api_key: str = ""
    # The served model ID, not the alias. Requesting "...-Instruct" is silently
    # aliased to Turbo; recording the alias would let the cache key claim two
    # different builds are the same model.
    deepinfra_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    # Daily spend ceiling in USD, enforced by the cost ledger (Phase 8 WS-2).
    # The free tier used to be its own cost control; on a metered provider this
    # is what replaces it.
    daily_usd_cap: float = 2.00

    # Tracing — SDK v3 reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"       # legacy alias
    langfuse_base_url: str = "https://cloud.langfuse.com"   # SDK v3 canonical
    tracing_enabled: bool = True

    # Vector store
    database_url: str = ""
    # ivfflat probes for the ctgov_live corpus (lists=161). Bench (PHASE7 WS-4,
    # recall vs exact top-100 on 26k trials): probes=20 recovered only ~62%, the
    # knee is ~40 (~full recall, <200ms warm). Eval-cohort scopes seq-scan the
    # source subset (ivfflat bypassed) so this only affects production ctgov_live.
    pgvector_probes: int = 40

    # ClinicalTrials.gov
    ctgov_api_base: str = "https://clinicaltrials.gov/api/v2"
    ctgov_page_size: int = 100
    ctgov_request_delay: float = 1.5  # seconds — stay under 50 req/min

    # Scope
    condition_class: str = "oncology"

    # Demo serving source: "sigir" = $0 FileIndex (HF Spaces default);
    # "ctgov_live" = production pgvector + Postgres FTS (needs DATABASE_URL).
    demo_source: str = "sigir"
    # Hard cap on trials assessed per demo request (cost bound).
    demo_max_top_k: int = 5

    # Hugging Face
    hf_token: str = ""


settings = Settings()
