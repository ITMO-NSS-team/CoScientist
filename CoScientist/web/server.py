"""
Run the CoScientist web interface locally.

Usage:
    python -m CoScientist.web.server
    # or
    python CoScientist/web/server.py
"""

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
        port=8000,
        reload=False,
        log_level="info",
    )
