from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class ReleaseProfile:
    name: str
    archive_filenames: tuple[str, ...]
    official_splits: tuple[str, ...]
    expected_counts: dict[str, Any]
    rules: dict[str, str]
    zod_group_memberships: dict[str, dict[str, tuple[str, ...]]]


@lru_cache(maxsize=1)
def load_release_profile() -> ReleaseProfile:
    resource = files("opencvl_tools").joinpath("data/release_v1.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    memberships = {
        subset: {
            split: tuple(str(group) for group in groups)
            for split, groups in split_map.items()
        }
        for subset, split_map in payload["zod_group_memberships"].items()
    }
    return ReleaseProfile(
        name=str(payload["release_name"]),
        archive_filenames=tuple(str(value) for value in payload["archives"]),
        official_splits=tuple(str(value) for value in payload["official_splits"]),
        expected_counts=dict(payload["expected_counts"]),
        rules={str(key): str(value) for key, value in payload["rules"].items()},
        zod_group_memberships=memberships,
    )
