"""FastAPI entry for Phase 9 Stage A.

Wraps retrieve() and assess() as they are. Pins TG_PROMPT_VERSION=v4.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Interactive serving: drop the eval-default 7s analyst delay. setdefault so an
# operator can still pace deliberately.
os.environ.setdefault("TG_ANALYST_DELAY", "0")
# v4 assesses inclusion AND exclusion. Additive cache namespace.
os.environ.setdefault("TG_PROMPT_VERSION", "v4")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from trialguard.api.jobs import JobStore  # noqa: E402
from trialguard.api.rate_limit import RateLimiter  # noqa: E402
from trialguard.api.routes import router  # noqa: E402
from trialguard.api.schemas import SYNTHETIC_NOTICE  # noqa: E402
from trialguard.config import settings  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.medcpt_warm = False
    app.state.jobs = JobStore(ttl_seconds=settings.api_job_ttl_seconds)
    app.state.rate_limiters = {
        "search": RateLimiter(settings.api_search_rate_per_min),
        "assess": RateLimiter(settings.api_assess_rate_per_min),
    }
    app.state.assess_executor = ThreadPoolExecutor(
        max_workers=settings.api_assess_workers,
        thread_name_prefix="tg-assess",
    )
    # Warm MedCPT so the first /search is not a cold start. Skip when there is
    # no DATABASE_URL (local unit tests / CI without Neon).
    if settings.database_url:
        from trialguard.demo import warm_models

        warm_models()
        app.state.medcpt_warm = True
    yield
    app.state.assess_executor.shutdown(wait=False, cancel_futures=True)
    from trialguard.db.schema import close_pool

    close_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrialGuard API",
        description=(
            "Self-verifying clinical-trial eligibility. "
            f"{SYNTHETIC_NOTICE}"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    origin = settings.api_cors_origin.strip()
    if not origin or origin == "*":
        raise RuntimeError("API_CORS_ORIGIN must be a single concrete origin, not '*'.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
