# ZOD–Mapillary Pose Correction

A minimal reference implementation of the OpenCVL pipeline for correcting a
nearby Mapillary camera pose from one georeferenced Zenseact Open Dataset (ZOD)
frame.

The only required pipeline inputs are:

1. a ZOD single-frame ID; and
2. the path to a locally mounted ZOD dataset.

A Mapillary API token is supplied through the environment. The pipeline finds
nearby Mapillary images, builds MASt3R correspondences to the ZOD anchor,
associates those correspondences with ZOD LiDAR, and estimates the Mapillary
camera pose with COLMAP absolute-pose estimation and refinement.

You could in-theory replace the ZOD data with another datasource and use this pipeline for areas not covered by ZOD. 

## Method

```text
ZOD frame ID + dataset root
        │
        ├── front image, calibration, OXTS pose, compensated LiDAR
        │
        └── nearby Mapillary images and reported metadata
                         │
                         ▼
             MASt3R reciprocal matching
                         │
             fundamental-matrix filtering
                         │
              ZOD LiDAR-depth association
                         │
             COLMAP absolute-pose refinement
                         │
                         ▼
       accepted corrected poses + correspondences + metadata
```

Only high-confidence results are exposed as corrected poses. By default, a
candidate must have at least 200 LiDAR-backed correspondences and 100 COLMAP
inliers. Rejected candidates retain their diagnostics but have
`corrected_pose: null`.

## Installation

The tested setup uses Python 3.11, PyTorch 2.5.1, and CUDA 12.1. A CUDA-capable
GPU is strongly recommended; CPU execution is supported by the command but is
very slow for MASt3R.

```bash
cd zod-mapillary-pose-correction

conda env create -f environment.yml
conda activate opencvl-pose

./scripts/install_mast3r.sh
```

The installer clones MASt3R recursively at the pinned revision
`f5209afc300cec36239a7ac992263f36847bbba0`, installs its requirements, and
leaves it under `third_party/mast3r`. This repository itself is not installed
as a Python package. MASt3R and its model weights are not vendored here.

Create a Mapillary client token, then either export it or place it in a local
`.env` file:

```bash
cp .env.example .env
# Edit .env and replace the placeholder.
```

The `.env` file, downloaded MASt3R source, outputs, and model checkpoints are
ignored by Git.

## Run one frame

```bash
python run_pose_correction.py \
  --frame-id 032237 \
  --zod-root ~/zod_mount
```

The entry point is the visible root-level file `run_pose_correction.py`. An
equivalent module invocation is also available while working in the repository:

```bash
python -m opencvl_pose --frame-id 032237 --zod-root ~/zod_mount
```

Useful options:

```text
--output-dir PATH             Output root (default: outputs)
--radius METRES              Mapillary search radius (default: 5)
--max-candidates N           Maximum nearby images to evaluate (default: 8)
--top-match-percent PERCENT  Best MASt3R matches retained (default: 85)
--min-lidar-matches N        LiDAR-backed quality gate (default: 400)
--min-colmap-inliers N       COLMAP quality gate (default: 250)
--device DEVICE              auto, cpu, cuda, or cuda:N
```

The small wrapper in `examples/run_frame.sh` accepts the same two core inputs:

```bash
./examples/run_frame.sh 032237 ~/zod_mount
```

## Outputs

```text
outputs/032237/
├── summary.json
├── zod_anchor.png
├── zod_lidar_camera.npy
└── candidates/
    └── MAPILLARY_IMAGE_ID/
        ├── result.json
        ├── mapillary.jpg
        ├── anchor_crop.png
        ├── mapillary_crop.png
        ├── correspondences.npz
        └── verified_matches.png
```

`summary.json` contains the raw Mapillary pose, the OpenSfM pose, quality
statistics, and—only for accepted candidates—the corrected transforms:

- `mapillary_camera_from_zod_camera`: relative camera transform estimated from
  LiDAR-backed correspondences.
- `mapillary_camera_from_utm`: corrected global camera transform.

The transform naming follows `target_from_source`: multiplying a homogeneous
point in the source frame produces that point in the target frame.

`correspondences.npz` stores the retained MASt3R matches, fundamental-matrix
inliers, LiDAR-associated pixels, and ZOD-camera-frame 3D points.

## Python API

```python
from pathlib import Path

from opencvl_pose import PipelineConfig, run_pipeline

config = PipelineConfig(
    frame_id="032237",
    zod_root=Path("~/zod_mount"),
    output_dir=Path("outputs"),
)
summary = run_pipeline(config)
print(summary["selection"]["best_candidate_id"])
```

## Repository layout

```text
opencvl_pose/
├── cli.py             Command-line interface
├── config.py          Validated runtime configuration
├── geometry.py        Image/LiDAR geometry utilities
├── mapillary_data.py  Nearby-image metadata and download client
├── matching.py        MASt3R matching and COLMAP pose estimation
├── pipeline.py        End-to-end orchestration and output contract
├── visualization.py   Compact verified-match rendering
└── zod_data.py        ZOD image, LiDAR, calibration, and OXTS loading
```

## Licenses and responsible release

The wrapper code in this repository is MIT-licensed. MASt3R is CC BY-NC-SA 4.0
and its checkpoints have additional restrictions. ZOD and Mapillary imagery
also retain their own terms. Read [NOTICE.md](../NOTICE.md) before redistribution
or use.