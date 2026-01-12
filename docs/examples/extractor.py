import pandas as pd

import eyefeatures.features.measures as eye_measures
import eyefeatures.features.scanpath_dist as eye_dist
import eyefeatures.features.stats as eye_stats
from eyefeatures.features.extractor import Extractor

# load your pd.DataFrame with gazes
data = pd.read_csv("<your_data>.csv")  # read your dataframe

# define all required features & columns in single class
extractor = Extractor(
    features=[  # list of features
        eye_dist.SimpleDistances(
            methods=["euc", "eye", "man"],
            expected_paths_method="fwp",
        ),
        eye_measures.GriddedDistributionEntropy(),
        eye_stats.SaccadeFeatures(
            features_stats={"length": ["min", "max"], "acceleration": ["min", "mean"]}
        ),
    ],
    x="fixation_x",  # column with x-coordinate of fixations
    y="fixation_y",  # column with y-coordinate of fixations
    t="timestamp",  # column with timestamp
    duration="duration",  # column with duration in ms
    dispersion="dispersion",  # column with dispersion
    path_pk=["group"],  # list of columns by which to get paths
    pk=["SUBJ", "group"],  # list of columns being primary key
    return_df=True,  # return as pd.DataFrame
)

extractor.fit_transform(data)
