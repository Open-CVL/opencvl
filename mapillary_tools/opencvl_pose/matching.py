from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from .config import PipelineConfig
from .geometry import (
    associate_lidar_to_matches,
    center_crop_with_intrinsics,
    generate_depth_and_scene_maps,
    resize_with_intrinsics,
)
from .mapillary_data import MapillaryCandidate


MAST3R_WIDTH = 512
MAST3R_HEIGHT = 384


@dataclass(frozen=True)
class MatchResult:
    status: str
    topk_anchor_uv: np.ndarray
    topk_mapillary_uv: np.ndarray
    fundamental_anchor_uv: np.ndarray
    fundamental_mapillary_uv: np.ndarray
    lidar_anchor_uv: np.ndarray
    lidar_mapillary_uv: np.ndarray
    lidar_projected_uv: np.ndarray
    points_3d: np.ndarray
    anchor_crop: np.ndarray
    mapillary_crop: np.ndarray
    mapillary_from_anchor: np.ndarray | None
    refined_camera_parameters: np.ndarray | None
    colmap_inliers: int

    @property
    def stats(self) -> dict[str, int]:
        return {
            "mast3r_topk_matches": int(len(self.topk_anchor_uv)),
            "fundamental_inliers": int(len(self.fundamental_anchor_uv)),
            "lidar_backed_matches": int(len(self.points_3d)),
            "colmap_inliers": int(self.colmap_inliers),
        }


def _mast3r_root() -> Path:
    configured = os.getenv("MAST3R_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "third_party" / "mast3r"


def _import_mast3r() -> tuple[object, object, object, object]:
    root = _mast3r_root()
    if not (root / "mast3r").is_dir() or not (root / "dust3r").is_dir():
        raise RuntimeError(
            f"MASt3R was not found at {root}. Run scripts/install_mast3r.sh or set MAST3R_ROOT."
        )
    for path in (root, root / "dust3r"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    importlib.import_module("mast3r.utils.path_to_dust3r")
    model_class = importlib.import_module("mast3r.model").AsymmetricMASt3R
    fast_nn = importlib.import_module("mast3r.fast_nn").fast_reciprocal_NNs
    inference = importlib.import_module("dust3r.inference").inference
    load_images = importlib.import_module("dust3r.utils.image").load_images
    return model_class, fast_nn, inference, load_images


class Mast3rColmapEstimator:
    """MASt3R matching followed by LiDAR association and COLMAP absolute pose."""

    def __init__(self, config: PipelineConfig) -> None:
        import torch

        self.config = config
        self.device = config.device
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but CUDA is unavailable: {self.device}"
            )
        model_class, self._fast_nn, self._inference, self._load_images = (
            _import_mast3r()
        )
        self.model = (
            model_class.from_pretrained(config.model_name).to(self.device).eval()
        )

    def estimate(
        self,
        anchor_image: np.ndarray,
        mapillary_image: np.ndarray,
        lidar_camera: np.ndarray,
        anchor_intrinsics: np.ndarray,
        candidate: MapillaryCandidate,
    ) -> MatchResult:
        anchor_scale = max(
            MAST3R_WIDTH / anchor_image.shape[1], MAST3R_HEIGHT / anchor_image.shape[0]
        )
        query_scale = max(
            MAST3R_WIDTH / mapillary_image.shape[1],
            MAST3R_HEIGHT / mapillary_image.shape[0],
        )
        anchor_resized, anchor_k = resize_with_intrinsics(
            anchor_image, anchor_intrinsics, anchor_scale
        )
        query_resized, query_k = resize_with_intrinsics(
            mapillary_image, candidate.intrinsics, query_scale
        )
        depth_map, scene_map = generate_depth_and_scene_maps(
            lidar_camera, anchor_k, anchor_resized.shape[:2]
        )
        anchor_crop, _, depth_crop, scene_crop = center_crop_with_intrinsics(
            anchor_resized,
            anchor_k,
            MAST3R_HEIGHT,
            MAST3R_WIDTH,
            depth_map,
            scene_map,
        )
        query_crop, query_crop_k, _, _ = center_crop_with_intrinsics(
            query_resized, query_k, MAST3R_HEIGHT, MAST3R_WIDTH
        )
        assert depth_crop is not None and scene_crop is not None

        anchor_uv, mapillary_uv, distances = self._compute_matches(
            anchor_crop, query_crop
        )
        if len(anchor_uv) == 0:
            return self._empty_result("no_mast3r_matches", anchor_crop, query_crop)

        keep_count = max(1, int(len(anchor_uv) * self.config.top_match_percent / 100.0))
        best = np.argpartition(distances, keep_count - 1)[:keep_count]
        topk_anchor = anchor_uv[best].astype(np.float32)
        topk_mapillary = mapillary_uv[best].astype(np.float32)

        fundamental_anchor, fundamental_mapillary = self._fundamental_inliers(
            topk_anchor, topk_mapillary
        )
        points_3d, lidar_uv, matched_indices = associate_lidar_to_matches(
            fundamental_anchor,
            depth_crop,
            scene_crop,
            self.config.lidar_match_radius_px,
        )
        lidar_anchor = fundamental_anchor[matched_indices]
        lidar_mapillary = fundamental_mapillary[matched_indices]

        if len(points_3d) < self.config.min_lidar_matches:
            return MatchResult(
                status="insufficient_lidar_matches",
                topk_anchor_uv=topk_anchor,
                topk_mapillary_uv=topk_mapillary,
                fundamental_anchor_uv=fundamental_anchor,
                fundamental_mapillary_uv=fundamental_mapillary,
                lidar_anchor_uv=lidar_anchor,
                lidar_mapillary_uv=lidar_mapillary,
                lidar_projected_uv=lidar_uv,
                points_3d=points_3d,
                anchor_crop=anchor_crop,
                mapillary_crop=query_crop,
                mapillary_from_anchor=None,
                refined_camera_parameters=None,
                colmap_inliers=0,
            )

        transform, refined_parameters, colmap_inliers = self._absolute_pose(
            lidar_mapillary, points_3d, query_crop_k, candidate
        )
        status = (
            "accepted"
            if (
                transform is not None
                and colmap_inliers >= self.config.min_colmap_inliers
            )
            else "insufficient_colmap_inliers"
        )
        return MatchResult(
            status=status,
            topk_anchor_uv=topk_anchor,
            topk_mapillary_uv=topk_mapillary,
            fundamental_anchor_uv=fundamental_anchor,
            fundamental_mapillary_uv=fundamental_mapillary,
            lidar_anchor_uv=lidar_anchor,
            lidar_mapillary_uv=lidar_mapillary,
            lidar_projected_uv=lidar_uv,
            points_3d=points_3d,
            anchor_crop=anchor_crop,
            mapillary_crop=query_crop,
            mapillary_from_anchor=transform if status == "accepted" else None,
            refined_camera_parameters=refined_parameters
            if status == "accepted"
            else None,
            colmap_inliers=colmap_inliers,
        )

    def _compute_matches(
        self, anchor_crop: np.ndarray, mapillary_crop: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images = self._load_images(
            [anchor_crop, mapillary_crop], size=512, verbose=False
        )
        output = self._inference(
            [tuple(images)], self.model, self.device, batch_size=1, verbose=False
        )
        descriptor_1 = output["pred1"]["desc"].squeeze(0).detach()
        descriptor_2 = output["pred2"]["desc"].squeeze(0).detach()
        anchor_uv, mapillary_uv, distances, _ = self._fast_nn(
            descriptor_1,
            descriptor_2,
            subsample_or_initxy1=8,
            device=self.device,
            dist="dot",
            block_size=2**13,
        )
        height_1, width_1 = (int(value) for value in output["view1"]["true_shape"][0])
        height_2, width_2 = (int(value) for value in output["view2"]["true_shape"][0])
        margin = 3
        valid = (
            (anchor_uv[:, 0] >= margin)
            & (anchor_uv[:, 0] < width_1 - margin)
            & (anchor_uv[:, 1] >= margin)
            & (anchor_uv[:, 1] < height_1 - margin)
            & (mapillary_uv[:, 0] >= margin)
            & (mapillary_uv[:, 0] < width_2 - margin)
            & (mapillary_uv[:, 1] >= margin)
            & (mapillary_uv[:, 1] < height_2 - margin)
        )
        return anchor_uv[valid], mapillary_uv[valid], np.asarray(distances)[valid]

    def _fundamental_inliers(
        self, anchor_uv: np.ndarray, mapillary_uv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(anchor_uv) < 8:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        _, mask = cv2.findFundamentalMat(
            anchor_uv,
            mapillary_uv,
            cv2.FM_RANSAC,
            self.config.fundamental_ransac_px,
            0.99,
        )
        if mask is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        inliers = mask.ravel().astype(bool)
        return anchor_uv[inliers], mapillary_uv[inliers]

    def _absolute_pose(
        self,
        points_2d: np.ndarray,
        points_3d: np.ndarray,
        intrinsics: np.ndarray,
        candidate: MapillaryCandidate,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        model = "OPENCV" if candidate.camera_type == "perspective" else "OPENCV_FISHEYE"
        distortion = candidate.distortion
        camera = pycolmap.Camera(
            model=model,
            width=MAST3R_WIDTH,
            height=MAST3R_HEIGHT,
            params=[
                intrinsics[0, 0],
                intrinsics[1, 1],
                intrinsics[0, 2],
                intrinsics[1, 2],
                distortion[0],
                distortion[1],
                0.0,
                0.0,
            ],
        )
        estimation_options = pycolmap.AbsolutePoseEstimationOptions()
        estimation_options.ransac.max_error = 2.0 if model == "OPENCV" else 3.0
        estimation_options.ransac.confidence = 0.9999
        estimation_options.ransac.max_num_trials = 5000
        refinement_options = pycolmap.AbsolutePoseRefinementOptions()
        refinement_options.refine_focal_length = True
        refinement_options.refine_extra_params = True
        answer = pycolmap.estimate_and_refine_absolute_pose(
            np.asarray(points_2d, dtype=np.float64),
            np.asarray(points_3d, dtype=np.float64),
            camera,
            estimation_options=estimation_options,
            refinement_options=refinement_options,
        )
        if not answer:
            return None, None, 0
        camera_from_anchor = np.eye(4)
        camera_from_anchor[:3, :3] = answer["cam_from_world"].rotation.matrix()
        camera_from_anchor[:3, 3] = np.asarray(answer["cam_from_world"].translation)
        return camera_from_anchor, np.asarray(camera.params), int(answer["num_inliers"])

    @staticmethod
    def _empty_result(
        status: str, anchor_crop: np.ndarray, mapillary_crop: np.ndarray
    ) -> MatchResult:
        empty_uv = np.empty((0, 2), dtype=np.float32)
        return MatchResult(
            status=status,
            topk_anchor_uv=empty_uv,
            topk_mapillary_uv=empty_uv.copy(),
            fundamental_anchor_uv=empty_uv.copy(),
            fundamental_mapillary_uv=empty_uv.copy(),
            lidar_anchor_uv=empty_uv.copy(),
            lidar_mapillary_uv=empty_uv.copy(),
            lidar_projected_uv=empty_uv.copy(),
            points_3d=np.empty((0, 3), dtype=np.float32),
            anchor_crop=anchor_crop,
            mapillary_crop=mapillary_crop,
            mapillary_from_anchor=None,
            refined_camera_parameters=None,
            colmap_inliers=0,
        )
