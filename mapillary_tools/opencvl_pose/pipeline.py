from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .geometry import to_jsonable
from .mapillary_data import MapillaryCandidate, MapillaryClient
from .matching import Mast3rColmapEstimator, MatchResult
from .visualization import save_correspondences
from .zod_data import AnchorFrame, load_anchor


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")


def _candidate_metadata(candidate: MapillaryCandidate) -> dict:
    return {
        "image_id": candidate.image_id,
        "sequence_id": candidate.sequence_id,
        "distance_to_anchor_m": candidate.distance_to_anchor_m,
        "camera_type": candidate.camera_type,
        "raw_pose": {
            "longitude_latitude": candidate.raw_lonlat,
            "heading_degrees": candidate.raw_heading,
            "altitude_m": candidate.altitude,
        },
        "opensfm_pose": {
            "longitude_latitude": candidate.opensfm_lonlat,
            "heading_degrees": candidate.opensfm_heading,
            "rotation_rotvec": candidate.opensfm_rotation,
            "altitude_m": candidate.computed_altitude,
        },
    }


def _save_candidate_artifacts(
    frame_dir: Path,
    candidate: MapillaryCandidate,
    image: np.ndarray,
    result: MatchResult,
) -> dict[str, str]:
    candidate_dir = frame_dir / "candidates" / candidate.image_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(candidate_dir / "mapillary.jpg", quality=95)
    Image.fromarray(result.anchor_crop).save(candidate_dir / "anchor_crop.png")
    Image.fromarray(result.mapillary_crop).save(candidate_dir / "mapillary_crop.png")
    np.savez_compressed(
        candidate_dir / "correspondences.npz",
        topk_anchor_uv=result.topk_anchor_uv,
        topk_mapillary_uv=result.topk_mapillary_uv,
        fundamental_anchor_uv=result.fundamental_anchor_uv,
        fundamental_mapillary_uv=result.fundamental_mapillary_uv,
        lidar_anchor_uv=result.lidar_anchor_uv,
        lidar_mapillary_uv=result.lidar_mapillary_uv,
        lidar_projected_uv=result.lidar_projected_uv,
        points_3d=result.points_3d,
    )
    save_correspondences(
        result.anchor_crop,
        result.mapillary_crop,
        result.lidar_anchor_uv,
        result.lidar_mapillary_uv,
        candidate_dir / "verified_matches.png",
    )
    return {
        "mapillary_image": str(
            (candidate_dir / "mapillary.jpg").relative_to(frame_dir)
        ),
        "correspondences": str(
            (candidate_dir / "correspondences.npz").relative_to(frame_dir)
        ),
        "verified_matches": str(
            (candidate_dir / "verified_matches.png").relative_to(frame_dir)
        ),
    }


def _corrected_global_pose(
    anchor: AnchorFrame, result: MatchResult
) -> np.ndarray | None:
    if result.mapillary_from_anchor is None:
        return None
    return (
        result.mapillary_from_anchor
        @ np.linalg.inv(anchor.camera_to_oxts)
        @ np.linalg.inv(anchor.utm_from_oxts)
    )


def run_pipeline(config: PipelineConfig, mapillary_token: str | None = None) -> dict:
    """Run pose correction for one ZOD frame and return the saved summary."""
    token = mapillary_token or os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("Set MAPILLARY_ACCESS_TOKEN in the environment or .env file")

    frame_dir = config.frame_output_dir
    frame_dir.mkdir(parents=True, exist_ok=True)
    anchor = load_anchor(
        config.frame_id,
        config.zod_root,
        config.zod_version,
        config.lidar_min_depth_m,
        config.lidar_max_depth_m,
    )
    Image.fromarray(anchor.image).save(frame_dir / "zod_anchor.png")
    np.save(frame_dir / "zod_lidar_camera.npy", anchor.lidar_camera)

    client = MapillaryClient(token)
    candidates = client.nearby_candidates(
        anchor.latitude,
        anchor.longitude,
        config.search_radius_m,
        config.max_candidates,
    )
    if not candidates:
        raise RuntimeError(
            "No supported Mapillary images were found near the ZOD anchor"
        )

    estimator = Mast3rColmapEstimator(config)
    records = []
    for candidate in candidates:
        base_record = _candidate_metadata(candidate)
        try:
            image = client.download_image(candidate)
            result = estimator.estimate(
                anchor.image,
                image,
                anchor.lidar_camera,
                anchor.intrinsics,
                candidate,
            )
            files = _save_candidate_artifacts(frame_dir, candidate, image, result)
            mapillary_from_utm = _corrected_global_pose(anchor, result)
            record = {
                **base_record,
                "status": result.status,
                "stats": result.stats,
                "corrected_pose": None
                if mapillary_from_utm is None
                else {
                    "mapillary_camera_from_zod_camera": result.mapillary_from_anchor,
                    "mapillary_camera_from_utm": mapillary_from_utm,
                    "refined_camera_parameters": result.refined_camera_parameters,
                },
                "files": files,
            }
        except Exception as error:
            record = {
                **base_record,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        candidate_dir = frame_dir / "candidates" / candidate.image_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(candidate_dir / "result.json", record)

    accepted = [record for record in records if record["status"] == "accepted"]
    accepted.sort(
        key=lambda item: (
            item["stats"]["colmap_inliers"],
            item["stats"]["lidar_backed_matches"],
        ),
        reverse=True,
    )
    summary = {
        "frame_id": config.frame_id,
        "anchor": {
            "country_code": anchor.country_code,
            "latitude": anchor.latitude,
            "longitude": anchor.longitude,
            "utm_from_oxts": anchor.utm_from_oxts,
            "front_camera_intrinsics": anchor.intrinsics,
            "lidar_points": len(anchor.lidar_camera),
        },
        "selection": {
            "search_radius_m": config.search_radius_m,
            "minimum_lidar_matches": config.min_lidar_matches,
            "minimum_colmap_inliers": config.min_colmap_inliers,
            "candidate_count": len(records),
            "accepted_count": len(accepted),
            "best_candidate_id": accepted[0]["image_id"] if accepted else None,
        },
        "candidates": records,
    }
    _write_json(frame_dir / "summary.json", summary)
    return to_jsonable(summary)
