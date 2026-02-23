from __future__ import annotations

from omegaconf import OmegaConf

import triforces.train as train_module


def test_maybe_init_wandb_skips_when_logger_disabled(monkeypatch) -> None:
    called = {"value": False}

    def fake_init(**kwargs):
        _ = kwargs
        called["value"] = True
        return object()

    monkeypatch.setattr(train_module.wandb, "init", fake_init)

    cfg = OmegaConf.create({"logger": {"enabled": False}})
    run = train_module._maybe_init_wandb(cfg)

    assert run is None
    assert called["value"] is False


def test_maybe_init_wandb_reads_logger_config(monkeypatch) -> None:
    calls: list[dict] = []
    sentinel = object()

    def fake_init(**kwargs):
        calls.append(dict(kwargs))
        return sentinel

    monkeypatch.setattr(train_module.wandb, "init", fake_init)

    cfg = OmegaConf.create(
        {
            "logger": {
                "enabled": True,
                "project": "unit-test-project",
                "mode": "offline",
            }
        }
    )
    run = train_module._maybe_init_wandb(cfg)

    assert run is sentinel
    assert len(calls) == 1
    assert calls[0]["project"] == "unit-test-project"
    assert calls[0]["mode"] == "offline"
    assert "config" in calls[0]


def test_maybe_init_wandb_skips_when_logger_missing(monkeypatch) -> None:
    called = {"value": False}

    def fake_init(**kwargs):
        _ = kwargs
        called["value"] = True
        return object()

    monkeypatch.setattr(train_module.wandb, "init", fake_init)

    cfg = OmegaConf.create({})
    run = train_module._maybe_init_wandb(cfg)

    assert run is None
    assert called["value"] is False


def test_maybe_init_wandb_uses_run_name_when_logger_name_missing(monkeypatch) -> None:
    calls: list[dict] = []
    sentinel = object()

    def fake_init(**kwargs):
        calls.append(dict(kwargs))
        return sentinel

    monkeypatch.setattr(train_module.wandb, "init", fake_init)

    cfg = OmegaConf.create({"logger": {"enabled": True, "project": "triforces"}})
    run = train_module._maybe_init_wandb(cfg, run_name="my-run")

    assert run is sentinel
    assert len(calls) == 1
    assert calls[0]["name"] == "my-run"


def test_maybe_init_wandb_prefers_explicit_logger_name_over_run_name(
    monkeypatch,
) -> None:
    calls: list[dict] = []
    sentinel = object()

    def fake_init(**kwargs):
        calls.append(dict(kwargs))
        return sentinel

    monkeypatch.setattr(train_module.wandb, "init", fake_init)

    cfg = OmegaConf.create(
        {"logger": {"enabled": True, "project": "triforces", "name": "explicit-name"}}
    )
    run = train_module._maybe_init_wandb(cfg, run_name="fallback-name")

    assert run is sentinel
    assert len(calls) == 1
    assert calls[0]["name"] == "explicit-name"
