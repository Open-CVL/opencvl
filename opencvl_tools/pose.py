from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


MAPILLARY_POSE_SOURCES = ("mapilraw", "mapilopensfm", "mast3rpnp")


@dataclass(frozen=True)
class GroundTruthPose:
    dx: float
    dy: float
    heading: float

    def pixel(self, width: int, height: int) -> tuple[float, float]:
        """Return the ground-camera location in original aerial-image pixels."""

        return width / 2.0 + self.dx, height / 2.0 - self.dy

    @property
    def heading_degrees(self) -> float:
        return self.heading % 360.0


def select_pose(record: Mapping[str, Any]) -> GroundTruthPose | None:
    direct = _pose_from_keys(record, "dx", "dy", "heading")
    if direct is not None:
        return direct
    for source in MAPILLARY_POSE_SOURCES:
        pose = _pose_from_keys(
            record,
            f"dx_{source}",
            f"dy_{source}",
            f"heading_{source}",
        )
        if pose is not None:
            return pose
    return None


def _pose_from_keys(
    record: Mapping[str, Any],
    dx_key: str,
    dy_key: str,
    heading_key: str,
) -> GroundTruthPose | None:
    values = (
        _finite_number(record.get(dx_key)),
        _finite_number(record.get(dy_key)),
        _finite_number(record.get(heading_key)),
    )
    if any(value is None for value in values):
        return None
    dx, dy, heading = values
    return GroundTruthPose(float(dx), float(dy), float(heading))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
