"""Hugging Face Spaces entry point. Launches the TrialGuard Gradio demo.

Config comes from Space secrets (GROQ_API_KEY, optional Langfuse keys) via
pydantic-settings. Retrieval is the self-contained FileIndex — no database.
"""

from trialguard.demo import launch

if __name__ == "__main__":
    launch(server_name="0.0.0.0", server_port=7860)
