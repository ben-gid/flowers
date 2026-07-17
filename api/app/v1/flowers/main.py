import os
import sys
from pathlib import Path

import uvicorn

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    # Use "0.0.0.0" to listen on all interfaces, or "127.0.0.1" for local only
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    print(f"Starting server on {host}:{port}")
    uvicorn.run("api.app.v1.flowers.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
