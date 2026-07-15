# OpenCVL Developer Tools

Developer tools and utilities for working with **OpenCVL: An Open, Diverse, and Large-Scale Dataset for Fine-Grained Cross-View Localization**.

OpenCVL is a large-scale dataset designed for cross-view localization research, providing aligned ground-level and aerial imagery across diverse geographic regions. This repository contains tools for accessing, preparing, processing, and evaluating OpenCVL data.

Project website:

https://open-cvl.github.io/

---

## Overview

Cross-view localization aims to determine the geographic location of a ground-level image by matching it with aerial or satellite imagery.

OpenCVL provides a large-scale benchmark for this task with:

- 617,388 ground–aerial image pairs
- Coverage across 4 countries
- Coverage across 41 cities
- More than 7,000 km² of geographic area

The dataset includes diverse environments, viewpoints, and capture conditions to support research in fine-grained localization.

---

## Features

This repository provides tools for:

- Mapillary imagery downloading
- Our novel pose correction pipeline for Mapillary images 
- Aerial imagery downloading
- Visualization tools
- Ground/aerial image pair handling
- Data preprocessing

---

## Dataset Information

### Geographic Coverage

| Country | Image Pairs |
|----------|-------------|
| Sweden | 327,647 |
| Netherlands | 93,062 |
| Poland | 147,173 |
| Norway | 49,506 |

---

## Installation

### Requirements

See accompanying requirements.txt for the respective features.



## Getting Started

### Download Dataset

```bash

```

### Prepare Dataset

```bash

```

### Run Example

```bash

```

---

## Repository Structure

```text
OpenCVL/
├── datasets/          # Dataset loading and processing
├── tools/             # Developer utilities
├── scripts/           # Training and evaluation scripts
├── configs/           # Configuration files
├── examples/          # Example usage
└── README.md
```

---

## Data Format

Each dataset sample contains:

```text
sample/
├── ground_image
├── aerial_image
├── metadata
└── location_information
```

Additional format details:

```

```

---

## Development Tools

### Dataset Downloader

Description:

```

```

Usage:

```bash

```

---

### Dataset Converter

Description:

```

```

Usage:

```bash

```

---

### Visualization Tools

Description:

```

```

Usage:

```bash

```

---

## Training

Example training command:

```bash

```

---

## Evaluation

Example evaluation command:

```bash

```

Evaluation metrics:

```

```

---

## Documentation

Documentation:

```

```

---

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

---

## Contributing

Contributions are welcome.

Please submit issues or pull requests for:

- Bug fixes
- Documentation improvements
- New utilities
- Dataset tooling improvements

Contribution guidelines:

```

```

---

## License

```

```

---

## Acknowledgements

OpenCVL is developed by researchers from:

- École Polytechnique Fédérale de Lausanne (EPFL)
- Delft University of Technology (TU Delft)
- Southern University of Science and Technology (SUSTech)
- Zenseact

---

## Contact

Zimin Xia (zimin.xia at epfl dot ch)
Mubariz Zaffar (m.zaffar at tudelft dot nl)

```
