"""
Run the CoScientist web interface locally.

Usage:
    python -m CoScientist.web.server
    # or
    python CoScientist/web/server.py

Environment:
    COSCIENTIST_CONFIG    — system profile (e.g. "microfluidics"); default system.yaml
    COSCIENTIST_WEB_PORT  — port to serve on (default 8000), so a second
                            profile instance can run next to the default one
"""

import os
import sys
import uvicorn
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from CoScientist.web.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "CoScientist.web.server:app",
        host="127.0.0.1",
        port=int(os.environ.get("COSCIENTIST_WEB_PORT", "8000")),
        reload=False,
        log_level="info",
    )
