from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .archives import prepare_archives
from .dataset import find_sample_by_ground_image
from .splits import create_split_manifests
from .visualization import render_sample_pair
from .watermark import watermark_sample


COMMANDS = ("prepare", "make-splits", "watermark", "plot")


def build_parser(command: str, *, prog: str | None = None) -> argparse.ArgumentParser:
    if command not in COMMANDS:
        raise ValueError(f"unknown command: {command}")

    descriptions = {
        "prepare": "Verify and extract a complete OpenCVL archive set.",
        "make-splits": "Create the official train, validation, and test splits.",
        "watermark": "Write source-watermarked images for one sample.",
        "plot": "Plot one image pair with its GT position and heading.",
    }
    parser = argparse.ArgumentParser(prog=prog, description=descriptions[command])

    if command == "prepare":
        parser.add_argument(
            "archive_directory",
            type=Path,
            help="Directory containing the OpenCVL .tar files and SHA256SUMS.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("OpenCVL"),
            help="Prepared dataset directory (default: OpenCVL).",
        )
        return parser

    parser.add_argument("root", type=Path, help="Extracted OpenCVL dataset root.")
    if command == "make-splits":
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Directory for split JSONL files and summary.json.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace existing split files.",
        )
    elif command == "watermark":
        _add_ground_image_argument(parser)
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Directory for the two watermarked image files.",
        )
    else:
        _add_ground_image_argument(parser)
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("opencvl_pair.png"),
            help="Output image path (default: opencvl_pair.png).",
        )
    return parser


def main(
    command: str,
    argv: Sequence[str] | None = None,
    *,
    prog: str | None = None,
) -> int:
    parser = build_parser(command, prog=prog)
    args = parser.parse_args(argv)
    try:
        return _run(command, args)
    except (OSError, ValueError, KeyError, IndexError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _add_ground_image_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ground-image",
        required=True,
        help="Ground image filename or relative path.",
    )


def _run(command: str, args: argparse.Namespace) -> int:
    if command == "prepare":
        output = prepare_archives(args.archive_directory, args.output)
        print(f"dataset: {output}")
        return 0

    if command == "make-splits":
        summary = create_split_manifests(
            args.root,
            args.output,
            overwrite=args.overwrite,
        )
        for split, count in sorted(summary.counts.items()):
            print(f"{split}: {count}")
        print(f"summary: {summary.output / 'summary.json'}")
        return 0

    sample = find_sample_by_ground_image(args.root, args.ground_image)
    if command == "watermark":
        for output in watermark_sample(args.root, sample, args.output):
            print(f"wrote: {output}")
        return 0

    output = render_sample_pair(sample, args.output)
    print(f"wrote: {output}")
    return 0
