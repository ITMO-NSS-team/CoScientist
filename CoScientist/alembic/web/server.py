"""Run the alembic pipeline dashboard locally.

Usage:
    python -m alembic.web.server
    # or
    python CoScientist/alembic/web/server.py

The pipeline writes its workdir under ``.alembic/`` relative to the current
working directory (same as the CLI), so launch this from the same place you
would run ``python CoScientist/alembic/main.py``.
"""
import sys
from pathlib import Path

# CoScientist/alembic/web/server.py -> CoScientist  (so `import alembic.*` works)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn

from alembic.web.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "alembic.web.server:app",
        host="127.0.0.1",
        port=8100,
        reload=False,
        log_level="info",
    )
