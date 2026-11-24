"""SDK error type — raised for any non-2xx API response."""

from __future__ import annotations


class RagError(RuntimeError):
    """An API call failed. Carries the HTTP status and the server's error code/message."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"[{status_code} {code}] {message}")
