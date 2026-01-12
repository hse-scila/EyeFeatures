import pandas as pd

from eyefeatures.visualization.dynamic_visualization import scanpath_animation

# read data and define column names
data = pd.read_csv("<your_data>.csv")  # read your dataframe
x = "fixation_x"
y = "fixation_y"
aoi = "fixation_AOI"
duration = "fixation_duration"

# extract a single object from data
record = data[data["OBJECT_ID"] == 0]  # example

# get scanpath animation of OBJECT_ID=0
scanpath_animation(
    record,
    x=x,
    y=y,
    add_regression=True,
    rule=(2,),
    animation_duration=500,
    save_gif="scanpath.gif",
)
