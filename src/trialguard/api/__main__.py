"""python -m trialguard.api — serve the Stage A FastAPI app."""

import uvicorn

from trialguard.api.app import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
