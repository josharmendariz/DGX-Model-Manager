"""Cover the four states of verify_auth.

The bug this guards: `verify_auth` used to return early whenever no API key was
configured, so on a non-loopback bind (this box serves on its tailnet address) every
mutating endpoint was effectively unauthenticated while still *looking* protected by
`Depends(verify_auth)`.
"""

import asyncio

import pytest
from fastapi import HTTPException

import app as appmod


class _Req:
    """Minimal stand-in for starlette's Request."""

    def __init__(self, token: str | None = None, method: str = "POST", path: str = "/api/x"):
        self.headers = {"authorization": f"Bearer {token}"} if token else {}
        self.method = method

        class _URL:
            pass

        self.url = _URL()
        self.url.path = path


def call(req: _Req):
    return asyncio.run(appmod.verify_auth(req))


@pytest.fixture
def key_set(monkeypatch):
    monkeypatch.setattr(appmod, "_API_KEY_HASH", appmod._hash_key("s3cret"))


def test_valid_key_allows(monkeypatch, key_set):
    assert call(_Req("s3cret")) is None


def test_wrong_key_401(monkeypatch, key_set):
    with pytest.raises(HTTPException) as exc:
        call(_Req("wrong"))
    assert exc.value.status_code == 401


def test_missing_key_401(monkeypatch, key_set):
    with pytest.raises(HTTPException) as exc:
        call(_Req())
    assert exc.value.status_code == 401


def test_no_key_on_loopback_allows(monkeypatch):
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    monkeypatch.setattr(appmod, "APP_HOST", "127.0.0.1")
    monkeypatch.delenv("MODEL_MANAGER_ALLOW_UNAUTH", raising=False)
    assert call(_Req()) is None


def test_no_key_on_public_bind_fails_closed(monkeypatch):
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    monkeypatch.setattr(appmod, "APP_HOST", "100.115.54.83")
    monkeypatch.delenv("MODEL_MANAGER_ALLOW_UNAUTH", raising=False)
    with pytest.raises(HTTPException) as exc:
        call(_Req())
    assert exc.value.status_code == 503
    # the message must name both remedies
    assert "api_key" in exc.value.detail
    assert "MODEL_MANAGER_ALLOW_UNAUTH" in exc.value.detail


def test_acknowledged_open_bind_allows_but_logs(monkeypatch, caplog):
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    monkeypatch.setattr(appmod, "APP_HOST", "100.115.54.83")
    monkeypatch.setenv("MODEL_MANAGER_ALLOW_UNAUTH", "1")
    with caplog.at_level("WARNING"):
        assert call(_Req(method="DELETE", path="/api/inventory/model")) is None
    assert "UNAUTHENTICATED" in caplog.text
    assert "DELETE" in caplog.text
    assert "/api/inventory/model" in caplog.text


def test_runtime_key_change_takes_effect_immediately(monkeypatch):
    """PUT /api/config reassigns _API_KEY_HASH; the decision must not be cached."""
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    monkeypatch.setattr(appmod, "APP_HOST", "100.115.54.83")
    monkeypatch.setenv("MODEL_MANAGER_ALLOW_UNAUTH", "1")
    assert call(_Req()) is None

    monkeypatch.setattr(appmod, "_API_KEY_HASH", appmod._hash_key("newkey"))
    with pytest.raises(HTTPException) as exc:
        call(_Req())
    assert exc.value.status_code == 401
    assert call(_Req("newkey")) is None


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "", "  ", "nope"])
def test_falsy_allow_unauth_still_fails_closed(monkeypatch, value):
    """`MODEL_MANAGER_ALLOW_UNAUTH=0` must not mean "accept the risk"."""
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    monkeypatch.setattr(appmod, "APP_HOST", "100.115.54.83")
    monkeypatch.setenv("MODEL_MANAGER_ALLOW_UNAUTH", value)
    with pytest.raises(HTTPException) as exc:
        call(_Req())
    assert exc.value.status_code == 503


@pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on", " 1 "])
def test_truthy_allow_unauth_allows(monkeypatch, value):
    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    monkeypatch.setattr(appmod, "APP_HOST", "100.115.54.83")
    monkeypatch.setenv("MODEL_MANAGER_ALLOW_UNAUTH", value)
    assert call(_Req()) is None


def _lifespan_sleeps(monkeypatch, host, env=None):
    """Run _lifespan's startup half and report whether it imposed the 10 s delay."""
    import asyncio
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(appmod, "_API_KEY_HASH", "")
    monkeypatch.setattr(appmod, "APP_HOST", host)
    if env is None:
        monkeypatch.delenv("MODEL_MANAGER_ALLOW_UNAUTH", raising=False)
    else:
        monkeypatch.setenv("MODEL_MANAGER_ALLOW_UNAUTH", env)
    monkeypatch.setattr(appmod.asyncio, "sleep", fake_sleep)
    # _lifespan rebinds and then closes the module-global client; restore it after.
    monkeypatch.setattr(appmod, "_http", appmod._http)

    async def noop():
        return None

    monkeypatch.setattr(appmod, "_alert_loop", noop)
    monkeypatch.setattr(appmod, "_pinned_autostart", noop)

    async def run():
        cm = appmod._lifespan(appmod.app)
        await cm.__aenter__()
        try:
            await cm.__aexit__(None, None, None)
        except Exception:
            pass

    asyncio.run(run())
    return 10 in slept


def test_lifespan_uses_the_shared_loopback_list(monkeypatch):
    monkeypatch.setattr(appmod, "_LOOPBACK_HOSTS", ("127.0.0.1", "::1", "localhost", "lo-test"))
    assert _lifespan_sleeps(monkeypatch, "lo-test") is False


def test_lifespan_delays_on_open_bind_and_honours_only_truthy_ack(monkeypatch):
    assert _lifespan_sleeps(monkeypatch, "100.115.54.83") is True
    assert _lifespan_sleeps(monkeypatch, "100.115.54.83", env="0") is True
    assert _lifespan_sleeps(monkeypatch, "100.115.54.83", env="1") is False
