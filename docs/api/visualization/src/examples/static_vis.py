import pandas as pd

from eyefeatures.visualization.static_visualization import scanpath_visualization

data = pd.read_csv("<your_data>.csv")  # read your dataframe
x = "fixation_x"
y = "fixation_y"

scanpath_visualization(data, x, y, return_ndarray=False, with_axes=True, path_width=1)
