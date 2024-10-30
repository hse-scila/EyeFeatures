## Package Description

[EyeFeatures](https://github.com/hse-scila/EyeFeatures) is an open-source Python package for analyzing eye movement
data in any visual task. Its capabilities encompass preprocessing, visualization,
statistical analysis, feature engineering and machine learning. Its unique feature
is its architecture and versatility. Accepting data in .csv format containing gaze
position coordinates, the package allows filtration of raw data to remove noise and
detecting fixations and saccades with different algorithms. Having fixations any
standard descriptive statistical eye movement features (such as totalFD, meanFD etc.)
can be computed, including AOI-wise features. AOIs can be predefined or assigned
automatically. More complex features, such as chaos measures, topological features,
density maps, scanpath similarities for various distance metrics can be computed as well.
The package allows to account for the panel structure of the data, calculating shift
features relative to group averages. The visualization module allows output a variety
of visualization options, including static and dynamic scanpath plots. The architecture of the package allows seamless embedding of its
preprocessing and feature extraction classes in Sklearn pipelines. Moreover, it provides
datasets and models for deep learning with Pytorch.

## Installation

1. It is recommended to install package in separate python environment. In conda you can create it with `conda create -n <name_of_environment>`
2. To activate environment write `conda activate <name_of_environment>`. In order to make it visible in jupyter write `pip install ipykernel` and  `python -m ipykernel install --user --name <name_of_environment> --display-name "<name_of_environment>`

By default eyefeatures is installed without `deep` module:

3. To install eyefeatures write `pip install eyefeatures`.
4. Write command `cd EyeFeatures`.
5. Write command `pip install poetry`.

If you want to install it with `deep` module:

3. Write command `git clone https://github.com/hse-scila/EyeFeatures` (in windows you need to do it in anaconda prompt).
4. Write command `cd EyeFeatures`.
5. Write command `pip install poetry`.
6. Write `poetry install --with deep`.

## Documentation

Documentation for the latest version can be found [here](https://eyefeatures-docs.readthedocs.io/en/latest/). Documentation contains description of all classes, functions and their parameters.

## Tutorials

You can find notebooks with tutorials devoted to differnet parts of the library in this reposiry in [tutorial](https://github.com/hse-scila/EyeFeatures/tree/main/tutorials) folder.

## Coming soon

Extensive table with references to all methods is coming soon
