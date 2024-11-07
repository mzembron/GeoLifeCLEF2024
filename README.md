# GeoLifeCLEF 2024 Contest

This repository contains the source code for the multimodal model used in the [GeoLifeCLEF 2024](https://www.kaggle.com/competitions/geolifeclef-2024) contest.

# Solution

The implemented solution ([src/clef_model.py](src/clef_model.py)) consists of fusion of visual transformer and two ResNet18 architectures

Each modality is processed by separate model:
 - Visual transformer processes RGB and NIR images.
 - ResNets extract essential information from time series.
 - Standalone features are fed directly to the head classifier.

![solution diagram](docs/photos/solution_diagram.png)



## Kaggle score

|                     | Score   |
|---------------------|---------|
| Public leaderboard  | 0.32723 |
| Private leaderboard | 0.32825 |