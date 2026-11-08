"""Application-level exceptions for consistent API error responses."""


class ApplicationError(Exception):
    """An expected application error safe to expose to an API client."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
