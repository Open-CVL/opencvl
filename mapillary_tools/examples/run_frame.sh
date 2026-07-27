#!/usr/bin/env bash
set -euo pipefail

FRAME_ID="${1:?Usage: $0 ZOD_FRAME_ID /path/to/zod_dataset}"
ZOD_ROOT="${2:?Usage: $0 ZOD_FRAME_ID /path/to/zod_dataset}"
REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "${REPOSITORY_ROOT}/run_pose_correction.py" \
  --frame-id "${FRAME_ID}" \
  --zod-root "${ZOD_ROOT}" \
  --output-dir outputs
