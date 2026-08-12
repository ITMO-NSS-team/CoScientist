"""Run the dedicated Codesynapse CoScientist façade service."""

import uvicorn

from CoScientist.integrations.codesynapse.app import create_app
from CoScientist.integrations.codesynapse.settings import CodesynapseIntegrationSettings


def main() -> None:
    settings = CodesynapseIntegrationSettings()
    uvicorn.run(create_app(settings), host="0.0.0.0", port=8010, log_level="info")


if __name__ == "__main__":
    main()
