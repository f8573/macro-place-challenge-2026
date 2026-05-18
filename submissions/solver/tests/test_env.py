"""Tests for submissions/solver/_env.py."""

import pytest

import submissions.solver._env as _env
from submissions.solver._env import SubmoduleMissingError, require_submodule


def test_require_submodule_raises_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(_env, "_PLC_MODULE", tmp_path / "nonexistent.py")
    with pytest.raises(SubmoduleMissingError):
        require_submodule()


def test_submodule_missing_error_is_runtime_error():
    assert issubclass(SubmoduleMissingError, RuntimeError)


def test_submodule_missing_error_message_contains_setup_command():
    exc = SubmoduleMissingError()
    msg = str(exc)
    assert "git submodule update --init" in msg
    assert "external/MacroPlacement" in msg


def test_require_submodule_passes_when_present(tmp_path, monkeypatch):
    fake_plc = tmp_path / "plc_client_os.py"
    fake_plc.touch()
    monkeypatch.setattr(_env, "_PLC_MODULE", fake_plc)
    require_submodule()  # should not raise
