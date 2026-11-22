"""Safe provider failures shared with the pipeline boundary."""

from app.schemas.contracts import ProviderError


class ProviderFailure(Exception):
    """A sanitized provider failure that is safe to persist and return to a client."""

    def __init__(self, error: ProviderError) -> None:
        self.error = error
        super().__init__(error.message)
