from __future__ import annotations

import sys
from pathlib import Path


def run_cli(command: str) -> int:
    module_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(module_root))

    from opencvl_tools.cli import main

    arguments = list(sys.argv[1:])
    program = Path(sys.argv[0]).name
    return main(
        command,
        arguments,
        prog=program,
    )
