from __future__ import annotations

import dataclasses

import pytest

from xrd import config as cfgmod
from xrd.config import Config, configure, current, override


def test_defaults_are_sane():
    c = Config()
    assert c.username
    assert c.redirect_limit > 0
    assert c.verify_tls is True
    assert c.auth_order[0] == "gsi"


def test_env_names_match_the_official_client(monkeypatch):
    monkeypatch.setenv("XRD_REQUESTTIMEOUT", "12.5")
    monkeypatch.setenv("XRD_REDIRECTLIMIT", "3")
    monkeypatch.setenv("XRD_CPCHUNKSIZE", "8192")
    c = Config()
    assert (c.request_timeout, c.redirect_limit, c.chunk_size) == (12.5, 3, 8192)


def test_unparseable_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("XRD_REDIRECTLIMIT", "not-a-number")
    default = Config.__dataclass_fields__["redirect_limit"].default_factory
    assert Config().redirect_limit == default()


def test_bearer_token_file_is_picked_up(monkeypatch):
    monkeypatch.setenv("BEARER_TOKEN_FILE", "/run/tok")
    assert Config().token_file == "/run/tok"


def test_evolve_is_a_copy():
    base = Config(username="a")
    other = base.evolve(username="b")
    assert (base.username, other.username) == ("a", "b")
    assert base is not other


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().username = "x"  # type: ignore[misc]


def test_repr_redacts_the_token():
    text = repr(Config(token="SECRET-JWT"))
    assert "SECRET-JWT" not in text
    assert "token='<redacted>'" in text


def test_repr_keeps_an_absent_token_readable():
    assert "token=None" in repr(Config())


def test_configure_replaces_the_current_config():
    saved = current()
    try:
        configure(username="scoped")
        assert current().username == "scoped"
    finally:
        cfgmod._current.set(saved)


def test_override_restores_on_exit():
    before = current()
    with override(username="temp") as c:
        assert c.username == "temp"
        assert current() is c
    assert current() is before


def test_override_restores_after_an_exception():
    before = current()
    with pytest.raises(RuntimeError), override(username="temp"):
        raise RuntimeError("boom")
    assert current() is before


@pytest.mark.parametrize(
    ("name", "attribute", "default"),
    [("XRD_REQUESTTIMEOUT", "request_timeout", 300.0), ("XRD_REDIRECTLIMIT", "redirect_limit", 16)],
)
def test_nonsense_in_the_environment_falls_back_to_the_default(
    monkeypatch, name, attribute, default
):
    """A typo in a shell profile must not make the client unusable."""
    monkeypatch.setenv(name, "half past two")
    assert getattr(Config(), attribute) == default


def test_the_username_comes_from_the_first_environment_variable_that_is_set(monkeypatch):
    for var in ("XRD_USER", "USER", "LOGNAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOGNAME", "from-logname")
    assert Config().username == "from-logname"
    monkeypatch.setenv("XRD_USER", "from-xrd")
    assert Config().username == "from-xrd"


def test_a_machine_with_no_name_for_its_user_still_gets_a_config(monkeypatch):
    """No environment and no passwd entry: ``nobody`` beats a traceback."""
    import getpass

    for var in ("XRD_USER", "USER", "LOGNAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: (_ for _ in ()).throw(KeyError("no passwd")))
    assert Config().username == "nobody"
