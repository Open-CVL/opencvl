# OpenCVL Devkit
This codebase supports our ECCV'26 Spotlight [paper](https://arxiv.org/abs/2608.25274): OpenCVL: An Open, Diverse, and Large-Scale
Dataset for Fine-Grained Cross-View Localization. See the [project webpage](https://open-cvl.github.io/) for an overview.

OpenCVL Devkit has two key objectives:

- It provides Python tools for preparing, splitting, watermarking,
and visualizing the OpenCVL cross-view localization dataset. See the corresponding
`opencvl_tools/`. 

- Tools related to Mapillary downloading and correction are provided in `mapillary_tools/`. 
See the corresponding [README](mapillary_tools/README.md).

## OpenCVL overview

<p align="center">
  <a href="media/opencvl-overview-1080p.mp4">
    <img src="media/opencvl-overview-poster.jpg" alt="OpenCVL project overview video" width="900">
  </a>
</p>

<p align="center">
  <a href="media/opencvl-overview-1080p.mp4"><strong>Watch the OpenCVL overview video</strong></a>
</p>

## Features

This repository provides tools for:

- Registering, downloading, and prepare archives of the OpenCVL dataset (`opencvl_tools/`)
- Aerial imagery downloading codes (`opencvl_tools/`)
- Mapillary imagery downloading toolkit (`mapillary_tools/`)
- Implementation of our novel pose correction pipeline for Mapillary images (`mapillary_tools/`)
- Visualization tools (`opencvl_tools/` and `mapillary_tools/`)

## Requirements for OpenCVL tools

- Python 3.10 or newer
- Pillow for image loading, watermarking, and visualization

```bash
python3 -m pip install "Pillow>=9.2"
```

## Register, download, and prepare archives of the OpenCVL dataset

Register for dataset access at <https://open-cvl.github.io/access.html>, then
download all `.tar` archives and `SHA256SUMS` from the page shown after
registration. Place the downloaded files together in one directory.

Check that the complete archive set is present, verify every file, and extract
the dataset:

```bash
python3 scripts/prepare_dataset.py /path/to/downloaded_archives \
  --output /path/to/OpenCVL
```

## Create official splits of OpenCVL

Generate every benchmark manifest from the distributed `labels.json` files:

```bash
python3 scripts/create_splits.py /path/to/OpenCVL \
  --output /path/to/OpenCVL_splits
```

| File | Official split | OpenCVL v1 samples |
| --- | --- | ---: |
| `train.jsonl` | ZOD train + Mapillary cities | 579,752 |
| `val.jsonl` | ZOD validation | 14,756 |
| `test_cross_area.jsonl` | ZOD cross-area test | 18,504 |
| `test_snow.jsonl` | ZOD snowy test | 3,015 |
| `test_in_the_wild.jsonl` | Mapillary in-the-wild test | 1,361 |

## Add source watermarks to OpenCVL images [Optional but recommended]

If you display OpenCVL images in figures, slides, websites, or demos, we
recommend attributing the image source. This script creates attributed image
copies without modifying the original dataset.

```bash
python3 scripts/add_watermarks.py /path/to/OpenCVL \
  --ground-image 000023_uniform_2022-05-18T07:05:25.907107Z.png \
  --output /path/to/OpenCVL_watermarked
```

## Plot GT pose

The OpenCVL pose convention is:

```text
ground_x = aerial_width  / 2 + dx
ground_y = aerial_height / 2 - dy
```

Heading is in degrees, with `0` pointing north/up.

```bash
python3 scripts/plot_sample.py /path/to/OpenCVL \
  --ground-image 000023_uniform_2022-05-18T07:05:25.907107Z.png \
  --output ground_aerial_gt.png
```

## Download your own Mapillary imagery

We have provided codes to retrieve Mapillary imagery and associated metadata
that we found useful for this research.
Please see the [Python code](mapillary_tools/mapillary_downloader.py) 
and [Jupyter Notebook](mapillary_tools/mapillary_downloader_nb.ipynb) 
for this in the accompanying `mapillary_tools\`.


## Correct your own Mapillary imagery
The Python implementation of our Mapillary pose correction pipeline is provided in the 
accompanying `mapillary_tools/`. Please see the corresponding detailed [README](mapillary_tools/README.md) on that.

[Watch the full demo video](https://drive.google.com/file/d/1rjW6OQ2pGdxlhhzJEJ_TkIlP_bkHywxG/view?usp=sharing) 
of the Mapillary pose correction pipeline in action. 
A quick visualization is given below:

![Demo](mapillary_tools/pose_improvement_visualization.gif)


## Acknowledgements

OpenCVL is developed by researchers from:

- École Polytechnique Fédérale de Lausanne (EPFL)
- Delft University of Technology (TU Delft)
- Southern University of Science and Technology (SUSTech)
- Zenseact

---

## License

All data sources in OpenCVL and this toolkit are generally open-source and 
have permissible licenses. Please find overview in the accompanying [LICENSE](LICENSE) and [NOTICE](NOTICE.md).

## Contact
Zimin Xia (zimin.xia at epfl dot ch)
Mubariz Zaffar (m.zaffar at tudelft dot nl)
Julian F. P. Kooij (j.f.p.kooij at tudelft dot nl)

For questions related to aerial imagery and dataset usage
please contact Zimin at first, for the mapillary pose correction
Mubariz at first, and for other generic questions, please contact
any of us.

## Citation

If you use OpenCVL in your research, please cite:

```
@inproceedings{opencvl2026,
  title = {OpenCVL: An Open, Diverse, and Large-Scale Dataset for Fine-Grained Cross-View Localization},
  author = {Xia, Zimin and Zaffar, Mubariz and Fu, Junsheng and Alahi, Alexandre and Kooij, Julian F. P.},
  booktitle = {European Conference on Computer Vision},
  year = {2026}
}
```
