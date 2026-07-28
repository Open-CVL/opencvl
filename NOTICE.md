# Third-party software and data

This repository's wrapper code is released under the MIT License. The complete
pipeline also relies on software, model weights, APIs, and datasets with their
own terms.

- **Zenseact Open Dataset (ZOD)** access and use remain subject to the ZOD
  dataset terms: https://zod.zenseact.com/
- **Mapillary** imagery and API access remain subject to Mapillary's platform
  terms. Users must supply their own access token. Imagery from this platform 
  is available under CC BY SA. See here 
  [Mapillary license](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data)

- **Aerial imagery** from all sources in OpenCVL is open-source. See respective 
licenses from the correponding sources for each country.

- **MASt3R** is distributed by NAVER under CC BY-NC-SA 4.0. Its pretrained
  checkpoints have additional notices. This project does not vendor MASt3R or
  its weights. Review the upstream `LICENSE`, `NOTICE`, and
  `CHECKPOINTS_NOTICE` before use: https://github.com/naver/mast3r
- **COLMAP/pycolmap**, the ZOD Python SDK, PyTorch, OpenCV, and all other
  dependencies retain their respective licenses.

The MIT license in this repository does not override any third-party license
or data-use restriction. In particular, the MASt3R dependency makes the
Mapillary pose correction pipeline suitable for non-commercial research unless separate
permissions are obtained.

