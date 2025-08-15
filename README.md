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
of visualization options, including static and dynamic scanpath plots. The architecture of
the package allows seamless embedding of its
preprocessing and feature extraction classes in Sklearn pipelines. Moreover, it provides
datasets and models for deep learning with Pytorch.

## Documentation

Documentation for the latest version can be found [here](https://eyefeatures-docs.readthedocs.io/en/latest/).

## Tutorials

Here are package tutorials devoted to different parts of the library:

| Tutorial                                                                                                                  | Notebook                                                                                                                                                                                       |
|---------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Simple Features Tutorial](https://github.com/hse-scila/EyeFeatures/tree/main/tutorials/features_tutorial.ipynb)          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/features_tutorial.ipynb)       |
| [Complex Features Tutorial](https://github.com/hse-scila/EyeFeatures/tree/main/tutorials/complex_tutorial.ipynb)          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/complex_tutorial.ipynb)        |
| [Gazes Preprocessing Tutorial](https://github.com/hse-scila/EyeFeatures/tree/main/tutorials/preprocessing_tutorial.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/preprocessing_tutorial.ipynb)  |
| [AOI Definition Tutorial](https://github.com/hse-scila/EyeFeatures/tree/main/tutorials/AOI_definition_tutorial.ipynb)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/AOI_definition_tutorial.ipynb) |
| [Visualization Tutorial](https://github.com/hse-scila/EyeFeatures/tree/main/tutorials/visualization_tutorial.ipynb)       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/visualization_tutorial.ipynb)  |
| [Deep Learning Tutorial](https://github.com/hse-scila/EyeFeatures/tree/main/tutorials/DL_tutorial.ipynb)                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hse-scila/EyeFeatures/blob/main/tutorials/DL_tutorial.ipynb)             |

## Coming soon

Extensive table with references to all methods is coming soon.
