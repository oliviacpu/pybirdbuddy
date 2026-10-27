"""Bird Buddy session credentials."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuthTokens:
    """An immutable session snapshot, with secrets excluded from its repr."""

    access_token: str | None = field(repr=False)
    refresh_token: str = field(repr=False)
