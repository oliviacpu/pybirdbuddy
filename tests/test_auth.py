"""Regression coverage for Google sign-in and persistent Bird Buddy sessions."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from birdbuddy import queries
from birdbuddy.auth import AuthTokens
from birdbuddy.client import BirdBuddy
from birdbuddy.exceptions import AuthenticationFailedError, GraphqlError


def _issued(access="new-access", refresh="new-refresh"):
    """Return an API session payload."""
    return {"accessToken": access, "refreshToken": refresh}


def _error(code, message="Authentication failed"):
    """Return a GraphQL failure, including an untrusted server message."""
    return {"errors": [{"message": message, "extensions": {"code": code}}]}


async def test_google_signin_and_callback(graphql_mock, api_payloads, caplog):
    """Google tokens are exchanged once; session secrets never reach debug logs."""
    callback = AsyncMock()
    client = BirdBuddy(tokens_updated=callback)
    graphql_mock.return_value = {
        "data": {
            "authSocialSignIn": {
                "__typename": "Auth",
                **_issued(),
                "me": api_payloads["me"],
            }
        }
    }
    caplog.set_level(1, logger="birdbuddy")
    assert await client.sign_in_google("google-secret")
    assert client.user.id == api_payloads["me"]["user"]["id"]
    assert client.tokens == AuthTokens("new-access", "new-refresh")
    callback.assert_awaited_once_with(client.tokens)
    assert graphql_mock.call_args.kwargs["variables"] == {
        "socialSignInInput": {"provider": "GOOGLE", "token": "google-secret"}
    }
    for value in ("google-secret", "new-access", "new-refresh"):
        assert value not in caplog.text
        assert value not in repr(client.tokens)


async def test_google_rejection_redacts_server_error(graphql_mock, caplog):
    """An API error cannot echo the Google token into exceptions or logs."""
    graphql_mock.return_value = _error("AUTH_SOCIAL_INVALID_USER_ID", "google-secret")
    caplog.set_level(logging.DEBUG, logger="birdbuddy")
    with pytest.raises(AuthenticationFailedError) as error:
        await BirdBuddy().sign_in_google("google-secret")
    assert "google-secret" not in str(error.value)
    assert "google-secret" not in caplog.text


async def test_google_problem_does_not_create_session(graphql_mock):
    """A sign-in Problem never becomes an empty or partially authenticated session."""
    graphql_mock.return_value = {
        "data": {
            "authSocialSignIn": {
                "__typename": "Problem",
                "items": [{"kind": "INVALID"}],
            }
        }
    }
    client = BirdBuddy()
    with pytest.raises(AuthenticationFailedError):
        await client.sign_in_google("google-secret")
    assert client.tokens is None


async def test_rotation_is_saved_before_account_read(graphql_mock, api_payloads):
    """A restored session saves its rotated refresh token before loading entities."""
    saved = []

    async def save(tokens):
        saved.append(tokens)

    async def response(query, variables, headers):
        if query == queries.auth.REFRESH_AUTH_TOKEN:
            return {"data": {"authRefreshToken": _issued()}}
        assert saved == [AuthTokens("new-access", "new-refresh")]
        return {"data": {"me": api_payloads["me"]}}

    graphql_mock.side_effect = response
    client = BirdBuddy(refresh_token="persisted-refresh", tokens_updated=save)
    assert await client.refresh()
    restarted = BirdBuddy(refresh_token=saved[-1].refresh_token)
    await restarted.refresh()
    assert graphql_mock.call_args_list[2].kwargs["variables"] == {
        "refreshTokenInput": {"token": "new-refresh"}
    }


async def test_concurrent_requests_refresh_only_once(graphql_mock, api_payloads):
    """Simultaneous requests never race a rotating refresh token."""
    refreshes = 0

    async def response(query, variables, headers):
        nonlocal refreshes
        await asyncio.sleep(0)
        if query == queries.auth.REFRESH_AUTH_TOKEN:
            refreshes += 1
            return {"data": {"authRefreshToken": _issued()}}
        return {"data": {"me": api_payloads["me"]}}

    graphql_mock.side_effect = response
    client = BirdBuddy(refresh_token="persisted-refresh")
    assert all(await asyncio.gather(*(client.refresh() for _ in range(5))))
    assert refreshes == 1


async def test_late_expired_response_preserves_new_session(graphql_mock, api_payloads):
    """An old in-flight failure does not invalidate a freshly renewed token."""
    old_requests = 0
    refreshes = 0
    both_started = asyncio.Event()

    async def response(query, variables, headers):
        nonlocal old_requests, refreshes
        if query == queries.auth.REFRESH_AUTH_TOKEN:
            refreshes += 1
            return {"data": {"authRefreshToken": _issued()}}
        if headers["Authorization"] == "Bearer old-access":
            old_requests += 1
            if old_requests == 2:
                both_started.set()
            await both_started.wait()
            return _error("AUTH_TOKEN_EXPIRED_ERROR")
        return {"data": {"me": api_payloads["me"]}}

    graphql_mock.side_effect = response
    client = BirdBuddy(access_token="old-access", refresh_token="old-refresh")
    assert all(await asyncio.gather(client.refresh(), client.refresh()))
    assert refreshes == 1


async def test_temporary_refresh_failure_retains_session(graphql_mock):
    """Server outages propagate for retry without discarding the refresh token."""
    graphql_mock.return_value = _error("INTERNAL_SERVER_ERROR", "persisted-refresh")
    client = BirdBuddy(refresh_token="persisted-refresh")
    with pytest.raises(GraphqlError) as error:
        await client.refresh()
    assert client.tokens.refresh_token == "persisted-refresh"
    assert "persisted-refresh" not in str(error.value)


async def test_expired_refresh_requires_google_reauthentication(graphql_mock):
    """A rejected Bird Buddy refresh token asks the caller to authenticate again."""
    graphql_mock.return_value = _error("AUTH_REFRESH_TOKEN_EXPIRED_ERROR")
    client = BirdBuddy(refresh_token="expired-refresh")
    with pytest.raises(AuthenticationFailedError):
        await client.refresh()
    assert client.tokens is None


async def test_password_login_recovers_after_refresh_rejection(
    graphql_mock, api_payloads
):
    """Password users retain their automatic sign-in recovery path."""
    graphql_mock.side_effect = [
        _error("AUTH_REFRESH_TOKEN_EXPIRED_ERROR"),
        {
            "data": {
                "authEmailSignIn": {
                    "__typename": "Auth",
                    **_issued(),
                    "me": api_payloads["me"],
                }
            }
        },
        {"data": {"me": api_payloads["me"]}},
    ]
    client = BirdBuddy("user@email", "password", refresh_token="expired-refresh")
    assert await client.refresh()
    assert client.tokens == AuthTokens("new-access", "new-refresh")
