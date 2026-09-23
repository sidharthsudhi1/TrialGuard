"""python -m trialguard.api — serve the Stage A FastAPI app."""

import uvicorn

from trialguard.api.app import app


def main() -> None:
    # Container binds all interfaces; Fly's proxy is what faces the internet.
    uvicorn.run(app, host="0.0.0.0", port=8000)  # noqa: S104


if __name__ == "__main__":
    main()
