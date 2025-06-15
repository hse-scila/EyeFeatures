import pandas as pd
from eyefeatures.preprocessing.fixation_extraction import IDT
from eyefeatures.preprocessing.smoothing import WienerFilter, SavGolFilter
from sklearn.pipeline import Pipeline

# load your pd.DataFrame with gazes
data = pd.read_csv('<your_data>.csv')  # read your dataframe

# specify columns in your dataframe
x = 'gaze_x'              # column with x-coordinate of gazes
y = 'gaze_y'              # column with y-coordinate of gazes
t = 'timestamp'           # column with timestamp
pk = ['subject', 'film']  # list of columns being primary key

# initialize fixation extraction algorithm
fixation_extractor = IDT(x=x, y=y, t=t, pk=pk, min_duration=1e-5, max_dispersion=1e-3)

# create pipeline instance
pipe = Pipeline(steps=[
    ("w_filter", WienerFilter(x=x, y=y, t=t, pk=pk, K='auto')),          # Wiener
    ("sg_filter", SavGolFilter(x=x, y=y, t=t, pk=pk, window_length=10)), # Savitzkiy-Golay
    ("fixation_extractor", fixation_extractor)                           # IDT
])

# run preprocessing & IDT algorithm to get fixations from gazes
fixations_smooth = pipe.fit_transform(data)
