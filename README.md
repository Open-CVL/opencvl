# OpenCVL Devkit

OpenCVL Devkit provides Python tools for preparing, splitting, watermarking,
and visualizing the OpenCVL cross-view localization dataset.

## Requirements

- Python 3.10 or newer
- Pillow for image loading, watermarking, and visualization

```bash
python3 -m pip install "Pillow>=9.2"
```

## Register, download, and prepare archives

Register for dataset access at <https://open-cvl.github.io/access.html>, then
download all `.tar` archives and `SHA256SUMS` from the page shown after
registration. Place the downloaded files together in one directory.

Check that the complete archive set is present, verify every file, and extract
the dataset:

```bash
python3 scripts/prepare_dataset.py /path/to/downloaded_archives \
  --output /path/to/OpenCVL
```

## Create official splits

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

## Add source watermarks

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
