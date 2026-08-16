import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ChemServiceError(RuntimeError):
    pass


class ChemServiceClient:
    def __init__(
        self,
        host: str,
        port: str,
        timeout: int = 60,
        docking_timeout: int | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        # Docking is far heavier than this client's other endpoints (PDF/figure
        # extraction, SMILES conversion) — falls back to `timeout` if the
        # caller doesn't pass a dedicated value, but should always be given one
        # in practice (see calculate_docking_score).
        self.docking_timeout = docking_timeout if docking_timeout is not None else timeout

    def _post(
        self,
        endpoint: str,
        *,
        files: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> Any:
        url = f"{self.base_url}{endpoint}"
        logger.info("Calling ChemService API: %s", url)

        try:
            response = requests.post(
                url,
                files=files,
                params=params,
                timeout=timeout if timeout is not None else self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            raise ChemServiceError(
                f"Failed to call ChemService API at {url}: {e}"
            ) from e

        try:
            payload = response.json()
        except ValueError as e:
            raise ChemServiceError(f"Invalid JSON response from {url}") from e

        if payload is None:
            raise ChemServiceError(f"Empty JSON response from {url}")

        if "data" not in payload:
            raise ChemServiceError(
                f"Response from {url} does not contain 'data': {payload}"
            )

        return payload["data"]

    def extract_reactions_from_pdf(self, pdf_file: bytes) -> Any:
        return self._post(
            "/extract_reactions_from_pdf/",
            files={"pdf_file": pdf_file},
        )

    def extract_reactions_from_figure(self, image: bytes) -> Any:
        return self._post(
            "/extract_reactions_from_figure/",
            files={"image": image},
        )

    def extract_molecules_from_pdf(self, pdf_file: bytes) -> Any:
        return self._post(
            "/extract_molecules_from_pdf/",
            files={"pdf_file": pdf_file},
        )

    def extract_molecules_from_figure(self, image: bytes) -> Any:
        return self._post(
            "/extract_molecules_from_figure/",
            files={"image": image},
        )

    def convert_image_to_smiles(self, image: bytes) -> Any:
        return self._post(
            "/convert_image_to_smiles/",
            files={"image": image},
        )

    def calculate_docking_score(self, smiles: str, pdb_id: str) -> Any:
        return self._post(
            "/docking/",
            params={"smiles": smiles, "pdb_id": pdb_id},
            timeout=self.docking_timeout,
        )