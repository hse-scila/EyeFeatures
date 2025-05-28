from eyefeatures.preprocessing.fixation_extraction import IDT
from eyefeatures.preprocessing.smoothing import WienerFilter, SavGolFilter
from sklearn.pipeline import Pipeline

# load your pd.DataFrame with gazes
data = ...

# specify columns in your dataframe
x = 'gaze_x'              # column with x-coordinate of gazes
y = 'gaze_y'              # column with y-coordinate of gazes
t = 'timestamp'           # column with timestamp
pk = ['subject', 'film']  # list of columns being primary key

# initialize preprocessor
preprocessor = IDT(x=x, y=y, t=t, pk=pk, min_duration=1e-5, max_dispersion=1e-3)

# create pipeline instance
pipe = Pipeline(steps=[
    ("wf_filter", WienerFilter(x=x, y=y, t=t, pk=pk, K='auto')),          # Wiener filter
    ("sg_filter", SavGolFilter(x=x, y=y, t=t, pk=pk, window_length=10)),  # Savitzkiy-Golay filter
    ("preprocessor", preprocessor)                                        # IDT algorithm
])

# run preprocessing & IDT algorithm to get fixations from gazes
fixations_smooth = pipe.fit_transform(data)
print(fixations_smooth)
