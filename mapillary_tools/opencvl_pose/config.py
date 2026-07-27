from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings for one ZOD-anchor pose-correction run."""

    frame_id: str
    zod_root: Path
    output_dir: Path = Path("outputs")
    zod_version: str = "full"
    search_radius_m: float = 5.0
    max_candidates: int = 8
    top_match_percent: float = 85.0
    fundamental_ransac_px: float = 3.0
    lidar_match_radius_px: float = 1.0
    min_lidar_matches: int = 400
    min_colmap_inliers: int = 250
    lidar_min_depth_m: float = 2.0
    lidar_max_depth_m: float = 100.0
    device: str = "auto"
    model_name: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", str(self.frame_id))
        object.__setattr__(self, "zod_root", Path(self.zod_root).expanduser().resolve())
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.zod_version not in {"mini", "full"}:
            raise ValueError("zod_version must be 'mini' or 'full'")
        if not self.zod_root.exists():
            raise FileNotFoundError(f"ZOD dataset root does not exist: {self.zod_root}")
        if not 0 < self.top_match_percent <= 100:
            raise ValueError("top_match_percent must be in (0, 100]")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.min_lidar_matches < 4:
            raise ValueError("min_lidar_matches must be at least 4")

    @property
    def frame_output_dir(self) -> Path:
        return self.output_dir / self.frame_id
