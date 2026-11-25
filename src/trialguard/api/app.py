"""FastAPI entry for Phase 9 Stage A.

Wraps retrieve() and assess() as they are. Pins TG_PROMPT_VERSION=v4.
"""

from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Interactive serving: drop the eval-default 7s analyst delay. setdefault so an
# operator can still pace deliberately.
os.environ.setdefault("TG_ANALYST_DELAY", "0")
# v4 assesses inclusion AND exclusion. Additive cache namespace.
os.environ.setdefault("TG_PROMPT_VERSION", "v4")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from trialguard.api.jobs import make_job_store
from trialguard.api.rate_limit import RateLimiter
from trialguard.api.routes import router
from trialguard.api.schemas import SYNTHETIC_NOTICE
from trialguard.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.medcpt_warm = False
    app.state.jobs = make_job_store(
        ttl_seconds=settings.api_job_ttl_seconds,
        stale_seconds=settings.api_job_stale_seconds,
    )
    # Strong references to in-flight assess tasks; see routes.assess_start.
    app.state.assess_tasks = set()
    app.state.rate_limiters = {
        "search": RateLimiter(settings.api_search_rate_per_min),
        "assess": RateLimiter(settings.api_assess_rate_per_min),
        "trials": RateLimiter(settings.api_trial_rate_per_min),
    }
    # Open assess streams per client IP; see routes.assess_stream.
    app.state.open_streams = Counter()
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
        # Not awaited: an 80 MB fetch must not extend a boot that already takes
        # over a minute. Dense search runs on pgvector until this lands.
        from trialguard.retrieval.vector_cache import warm_in_background

        warm_in_background(settings.retrieval_vector_cache_source)
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
    # Staging only: name the machine that answered, so a load test can show the
    # proxy spread requests instead of assuming it.
    machine = os.environ.get("FLY_MACHINE_ID", "")
    if settings.api_staging_bypass_token and machine:

        @app.middleware("http")
        async def served_by(request: Request, call_next):
            response = await call_next(request)
            response.headers["X-Served-By"] = machine
            return response

    app.include_router(router)
    return app


app = create_app()
