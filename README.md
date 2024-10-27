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
of visualization options, including static and dynamic scanpath plots, customized heatmaps
and histograms. The architecture of the package allows seamless embedding of its
preprocessing and feature extraction classes in Sklearn pipelines. Moreover, it provides
datasets and models for deep learning with Pytorch. Since the work is in progress all
functionality will be implemented by the time of the report.

## Usage

For now, package is still in development and no pre-release version is uploaded to PyPI. However, you can still use it
by cloning the repository on your local machine:

1. Open a command prompt/terminal and move to an empty folder.
2. Write command `git clone https://github.com/hse-scila/EyeFeatures` (in windows you need to do it in anaconda prompt).
3. Write command `cd EyeFeatures`.
4. If you want to use all modules except for the `deep` module, then write `pip3 install -r base-requirements.txt`.
If you want to use the `deep` module, write `pip3 install -r requirements.txt`.
5. If you want to explore the library in Jupyter Notebook, then commands `pip3 install jupyter` and `jupyter notebook`
will help you.
