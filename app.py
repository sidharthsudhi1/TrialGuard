"""Hugging Face Spaces entry point. Launches the TrialGuard Gradio demo.

Config comes from Space secrets (GROQ_API_KEY, optional Langfuse keys) via
pydantic-settings. Retrieval is the self-contained FileIndex — no database.
"""

import os

# Interactive demo: a single query never approaches the per-minute token window, so
# the 7s batch-pacing delay (eval default) is dropped. Set before any ratelimit import.
os.environ.setdefault("TG_ANALYST_DELAY", "0")

# Pin the demo to the free tier. Phase 8 moved the project default to a metered
# provider, which is right for eval and wrong for a public demo: unbounded traffic
# against a paid host is a billing incident, and the $0 claim in the README should
# not depend on remembering to set a Space secret. setdefault, not a hard
# assignment, so an operator can still opt into a metered demo deliberately.
os.environ.setdefault("LLM_PROVIDER", "groq")

from trialguard.demo import launch  # noqa: E402 — must follow the env default above

if __name__ == "__main__":
    launch(server_name="0.0.0.0", server_port=7860)
