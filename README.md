# PARIS: Part-level Reconstruction and Motion Analysis for Articulated Objects

Authors

Under review

[Project page]() | [Paper]() | [Data](https://aspis.cmpt.sfu.ca/projects/paris/dataset.zip)

## Setup
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies. 

Create an environment from the `environment.yml` file.
```
conda env create -f environment.yml
```

Then install the torch bindings for [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn):
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
## Data
We release both synthetic and real data shown in the paper [here](https://aspis.cmpt.sfu.ca/projects/paris/datasets.zip). After downloaded, folders `data` and `load` should be put directly under the project directory.

Our synthetic data is preprocessed from the [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset. If you want to try out more examples, you can refer to `preprocess.py` to generate the two input states by articulating one part and save the meshes.

## Citation