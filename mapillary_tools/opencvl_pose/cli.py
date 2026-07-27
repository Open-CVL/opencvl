import argparse
import json
import os
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def load_local_env(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without requiring another package."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Correct nearby Mapillary camera poses using a ZOD front image, "
            "ZOD LiDAR, MASt3R correspondences, and COLMAP absolute pose refinement."
        )
    )
    parser.add_argument("--frame-id", required=True, help="ZOD single-frame identifier")
    parser.add_argument(
        "--zod-root", required=True, type=Path, help="Path to the ZOD dataset"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--zod-version", choices=["mini", "full"], default="full")
    parser.add_argument(
        "--radius", type=float, default=5.0, help="Mapillary search radius in metres"
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--top-match-percent", type=float, default=85.0)
    parser.add_argument("--min-lidar-matches", type=int, default=200)
    parser.add_argument("--min-colmap-inliers", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser


def main() -> None:
    load_local_env()
    args = build_parser().parse_args()
    config = PipelineConfig(
        frame_id=args.frame_id,
        zod_root=args.zod_root,
        output_dir=args.output_dir,
        zod_version=args.zod_version,
        search_radius_m=args.radius,
        max_candidates=args.max_candidates,
        top_match_percent=args.top_match_percent,
        min_lidar_matches=args.min_lidar_matches,
        min_colmap_inliers=args.min_colmap_inliers,
        device=args.device,
    )
    summary = run_pipeline(config)
    print(json.dumps(summary["selection"], indent=2))
    print(f"Results: {config.frame_output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
