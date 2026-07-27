from __future__ import annotations

import json
import math
from dataclasses import dataclass

import mapillary.interface as mly
import numpy as np
import requests
from PIL import Image
from io import BytesIO


DETAIL_FIELDS = [
    "altitude",
    "camera_parameters",
    "camera_type",
    "compass_angle",
    "computed_altitude",
    "computed_compass_angle",
    "computed_geometry",
    "computed_rotation",
    "height",
    "sequence",
    "thumb_original_url",
    "width",
]


@dataclass(frozen=True)
class MapillaryCandidate:
    image_id: str
    sequence_id: str | None
    camera_type: str
    raw_lonlat: tuple[float, float]
    raw_heading: float | None
    opensfm_lonlat: tuple[float, float] | None
    opensfm_heading: float | None
    opensfm_rotation: list[float] | None
    altitude: float | None
    computed_altitude: float | None
    width: int
    height: int
    camera_parameters: np.ndarray
    image_url: str
    distance_to_anchor_m: float

    @property
    def intrinsics(self) -> np.ndarray:
        focal = float(self.camera_parameters[0]) * max(self.width, self.height)
        return np.array(
            [
                [focal, 0.0, self.width / 2],
                [0.0, focal, self.height / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def distortion(self) -> np.ndarray:
        values = np.zeros(4, dtype=np.float64)
        available = min(2, max(0, len(self.camera_parameters) - 1))
        values[:available] = self.camera_parameters[1 : 1 + available]
        return values


def _distance_m(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    radius_m = 6_371_008.8
    phi_1, phi_2 = math.radians(lat_1), math.radians(lat_2)
    delta_phi = math.radians(lat_2 - lat_1)
    delta_lambda = math.radians(lon_2 - lon_1)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(value))


class MapillaryClient:
    def __init__(self, access_token: str, timeout_seconds: float = 60.0) -> None:
        if not access_token:
            raise ValueError("A Mapillary access token is required")
        self.timeout_seconds = timeout_seconds
        mly.set_access_token(access_token)

    def nearby_candidates(
        self, latitude: float, longitude: float, radius_m: float, limit: int
    ) -> list[MapillaryCandidate]:
        collection = mly.get_image_close_to(
            longitude=longitude,
            latitude=latitude,
            radius=radius_m,
            image_type="flat",
        ).to_dict()
        candidates: list[MapillaryCandidate] = []
        for feature in collection.get("features", []):
            try:
                candidate = self._candidate_from_feature(feature, latitude, longitude)
            except (KeyError, TypeError, ValueError, requests.RequestException):
                continue
            if candidate.camera_type in {"perspective", "fisheye"}:
                candidates.append(candidate)
        candidates.sort(key=lambda item: item.distance_to_anchor_m)
        return candidates[:limit]

    def _candidate_from_feature(
        self, feature: dict, anchor_latitude: float, anchor_longitude: float
    ) -> MapillaryCandidate:
        image_id = str(feature["properties"]["id"])
        response = json.loads(mly.image_from_key(image_id, fields=DETAIL_FIELDS))
        details = response["features"]
        if isinstance(details, list):
            details = details[0]
        properties = details["properties"]
        raw_lonlat = tuple(float(value) for value in feature["geometry"]["coordinates"])
        computed = properties.get("computed_geometry") or {}
        computed_lonlat = computed.get("coordinates")
        if computed_lonlat:
            computed_lonlat = tuple(float(value) for value in computed_lonlat)
        camera_parameters = np.asarray(
            properties["camera_parameters"], dtype=np.float64
        )
        if len(camera_parameters) < 1:
            raise ValueError("Mapillary camera parameters are empty")
        return MapillaryCandidate(
            image_id=image_id,
            sequence_id=str(feature["properties"].get("sequence_id"))
            if feature["properties"].get("sequence_id") is not None
            else None,
            camera_type=str(properties["camera_type"]),
            raw_lonlat=raw_lonlat,
            raw_heading=properties.get(
                "compass_angle", feature["properties"].get("compass_angle")
            ),
            opensfm_lonlat=computed_lonlat,
            opensfm_heading=properties.get("computed_compass_angle"),
            opensfm_rotation=properties.get("computed_rotation"),
            altitude=properties.get("altitude"),
            computed_altitude=properties.get("computed_altitude"),
            width=int(properties["width"]),
            height=int(properties["height"]),
            camera_parameters=camera_parameters,
            image_url=str(properties["thumb_original_url"]),
            distance_to_anchor_m=_distance_m(
                anchor_latitude, anchor_longitude, raw_lonlat[1], raw_lonlat[0]
            ),
        )

    def download_image(self, candidate: MapillaryCandidate) -> np.ndarray:
        response = requests.get(candidate.image_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return np.asarray(Image.open(BytesIO(response.content)).convert("RGB"))
