from __future__ import annotations

import triforces.train as train_module


def test_is_debug_enabled_truthy_values(monkeypatch) -> None:
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("DEBUG", value)
        assert train_module._is_debug_enabled() is True


def test_is_debug_enabled_falsey_values(monkeypatch) -> None:
    for value in ("", "0", "false", "no", "off", "random"):
        monkeypatch.setenv("DEBUG", value)
        assert train_module._is_debug_enabled() is False
