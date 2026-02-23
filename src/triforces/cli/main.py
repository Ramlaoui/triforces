from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import TextIO


def _print_help(*, file: TextIO) -> None:
    print(
        "Usage: triforces <command> [args...]\n"
        "\n"
        "Commands:\n"
        "  train   Run training from any Hydra config (pretraining or supervised).\n"
        "\n"
        "Examples:\n"
        "  triforces train -cn experiments/pretraining/orb/main_triforces train.epochs=10\n"
        "  triforces train -cn experiments/supervised/orb/energy_conserving train.epochs=10\n",
        file=file,
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print_help(file=sys.stdout)
        return 0

    command = argv[0]
    if command == "train":
        # Delegate to Hydra entrypoint; Hydra reads from sys.argv.
        from . import train_contrastive

        # Always surface full Hydra stack traces in CLI mode for easier debugging.
        os.environ["HYDRA_FULL_ERROR"] = "1"
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0], *argv[1:]]
            rc = train_contrastive.main()
            return int(rc) if rc is not None else 0
        finally:
            sys.argv = old_argv

    print(f"Unknown command: {command!r}\n", file=sys.stderr)
    _print_help(file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
