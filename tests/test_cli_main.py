import os
import sys

import pytest

from triforces.cli.main import main


def test_cli_help_prints_usage(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "Usage:" in out
    assert "triforces train" in out


def test_cli_unknown_command_exits_2(capsys):
    assert main(["nope"]) == 2
    err = capsys.readouterr().err
    assert "Unknown command" in err


def test_cli_train_dispatches_without_subcommand(monkeypatch):
    import triforces.cli.train_contrastive as train_contrastive

    called = {}

    def fake_train_main():
        called["argv"] = list(sys.argv)
        return 7

    monkeypatch.setattr(train_contrastive, "main", fake_train_main)
    monkeypatch.delenv("HYDRA_FULL_ERROR", raising=False)

    old_argv = sys.argv
    try:
        sys.argv = ["triforces", "train", "x=1"]
        assert main(["train", "x=1"]) == 7
        assert called["argv"] == ["triforces", "x=1"]
        assert os.environ.get("HYDRA_FULL_ERROR") == "1"
    finally:
        sys.argv = old_argv


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_cli_help_aliases(flag, capsys):
    assert main([flag]) == 0
    assert "Commands:" in capsys.readouterr().out
