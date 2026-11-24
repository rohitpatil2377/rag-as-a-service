"""Uvicorn entrypoint for the monolith (``rag-api`` console script)."""

from __future__ import annotations

import uvicorn

from rag_common import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "rag_api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,  # structlog owns logging
        server_header=False,  # don't disclose "server: uvicorn" (version/tech leak, VAPT)
        date_header=False,
    )


if __name__ == "__main__":
    main()
