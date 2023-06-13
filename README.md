# PARIS: Part-level Reconstruction and Motion Analysis for Articulated Objects

Authors

Accepted by xxxx

[Project page]() | [Paper]() | [Data]()

## Setup
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies. 

Create an environment from the `environment.yml` file.
```
conda env create -f environment.yml
```

Install the torch bindings for [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
## Data Preparation
