#!/usr/bin/env python3
"""Create the official OpenCVL split manifests from labels.json files."""

from _bootstrap import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli("make-splits"))
