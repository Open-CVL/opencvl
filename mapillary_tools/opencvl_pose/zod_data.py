from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import utm
from zod import ZodFrames
from zod.constants import Anonymization, Camera, Lidar
from zod.data_classes.ego_motion import OXTS_TIMESTAMP_OFFSET, interpolate_transforms
from zod.utils.geometry import transform_points

from .geometry import approximate_meridian_convergence, rotation_matrix_from_angles


@dataclass(frozen=True)
class AnchorFrame:
    frame_id: str
    country_code: str
    image: np.ndarray
    lidar_camera: np.ndarray
    intrinsics: np.ndarray
    camera_to_oxts: np.ndarray
    utm_from_oxts: np.ndarray
    latitude: float
    longitude: float


def _interpolated_oxts_pose(
    frame, frame_id: str, zod_root: Path
) -> tuple[np.ndarray, float, int, str]:
    frame_timestamp = frame.info.keyframe_time.timestamp()
    oxts_path = zod_root / "single_frames" / frame_id / "oxts.hdf5"
    with h5py.File(oxts_path, "r") as file:
        latitudes = file["posLat"][()]
        longitudes = file["posLon"][()]
        altitudes = file["posAlt"][()]
        yaws = 90.0 - file["heading"][()]
        pitches = file["pitch"][()]
        rolls = file["roll"][()]
        timestamps = (
            OXTS_TIMESTAMP_OFFSET + file["timestamp"][()] + file["leapSeconds"][()][0]
        )

    right = int(np.searchsorted(timestamps, frame_timestamp, side="right"))
    if right == 0 or right >= len(timestamps):
        raise RuntimeError(
            "ZOD frame timestamp is outside the associated OXTS time range"
        )

    poses = []
    zone_number, zone_letter = 0, ""
    for index in (right - 1, right):
        east, north, zone_number, zone_letter = utm.from_latlon(
            float(latitudes[index]), float(longitudes[index])
        )
        grid_yaw = yaws[index] + approximate_meridian_convergence(
            float(latitudes[index]), float(longitudes[index]), zone_number
        )
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix_from_angles(
            np.radians(rolls[index]),
            np.radians(pitches[index]),
            np.radians(grid_yaw),
            order="XYZ",
        )
        pose[:3, 3] = [east, north, altitudes[index]]
        poses.append(pose)

    fraction = (frame_timestamp - timestamps[right - 1]) / (
        timestamps[right] - timestamps[right - 1]
    )
    interpolated = interpolate_transforms(poses[0], poses[1], fraction)
    return interpolated, frame_timestamp, zone_number, zone_letter


def load_anchor(
    frame_id: str,
    zod_root: Path,
    version: str,
    min_depth_m: float,
    max_depth_m: float,
) -> AnchorFrame:
    """Load and calibrate the front image, compensated LiDAR, and global anchor pose."""
    frame = ZodFrames(dataset_root=str(zod_root), version=version)[frame_id]
    calibration = frame.calibration
    front_camera = calibration.cameras[Camera.FRONT]
    intrinsics = np.asarray(front_camera.intrinsics, dtype=np.float64)[:, :3]
    distortion = np.asarray(front_camera.distortion, dtype=np.float64)
    image_dimensions = tuple(int(value) for value in front_camera.image_dimensions)

    camera_to_oxts = np.asarray(
        calibration.get_extrinsics(Camera.FRONT).transform, dtype=np.float64
    )
    lidar_to_oxts = np.asarray(
        calibration.lidars[Lidar.VELODYNE].extrinsics.transform, dtype=np.float64
    )
    lidar_to_camera = np.linalg.inv(np.linalg.inv(lidar_to_oxts) @ camera_to_oxts)

    utm_from_oxts, timestamp, zone_number, zone_letter = _interpolated_oxts_pose(
        frame, frame_id, zod_root
    )
    latitude, longitude = utm.to_latlon(
        utm_from_oxts[0, 3], utm_from_oxts[1, 3], zone_number, zone_letter
    )

    image = np.asarray(frame.get_image(Anonymization.BLUR))
    rectified_intrinsics = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsics, distortion, image_dimensions, np.eye(3), balance=0.0
    )
    map_1, map_2 = cv2.fisheye.initUndistortRectifyMap(
        intrinsics,
        distortion,
        np.eye(3),
        rectified_intrinsics,
        image_dimensions,
        cv2.CV_16SC2,
    )
    image = cv2.remap(image, map_1, map_2, cv2.INTER_LINEAR)

    lidar = frame.compensate_lidar(frame.get_lidar()[0], timestamp).points
    lidar_camera = transform_points(lidar, lidar_to_camera)
    valid_depth = (lidar_camera[:, 2] > min_depth_m) & (
        lidar_camera[:, 2] < max_depth_m
    )
    lidar_camera = np.asarray(lidar_camera[valid_depth], dtype=np.float32)

    return AnchorFrame(
        frame_id=frame_id,
        country_code=frame.metadata.country_code,
        image=image,
        lidar_camera=lidar_camera,
        intrinsics=rectified_intrinsics,
        camera_to_oxts=camera_to_oxts,
        utm_from_oxts=utm_from_oxts,
        latitude=float(latitude),
        longitude=float(longitude),
    )
