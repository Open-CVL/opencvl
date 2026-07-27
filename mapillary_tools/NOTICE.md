# Third-party software and data

This repository's wrapper code is released under the MIT License. The complete
pipeline also relies on software, model weights, APIs, and datasets with their
own terms.

- **MASt3R** is distributed by NAVER under CC BY-NC-SA 4.0. Its pretrained
  checkpoints have additional notices. This project does not vendor MASt3R or
  its weights. Review the upstream `LICENSE`, `NOTICE`, and
  `CHECKPOINTS_NOTICE` before use: https://github.com/naver/mast3r
- **Zenseact Open Dataset (ZOD)** access and use remain subject to the ZOD
  dataset terms: https://zod.zenseact.com/
- **Mapillary** imagery and API access remain subject to Mapillary's platform
  terms. Users must supply their own access token.
- **COLMAP/pycolmap**, the ZOD Python SDK, PyTorch, OpenCV, and all other
  dependencies retain their respective licenses.

The MIT license in this repository does not override any third-party license
or data-use restriction. In particular, the MASt3R dependency makes the
reference pipeline suitable for non-commercial research unless separate
permissions are obtained.

