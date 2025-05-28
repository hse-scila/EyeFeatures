import eyefeatures.features.measures as eye_measures
import eyefeatures.features.stats as eye_stats
import eyefeatures.features.scanpath_dist as eye_dist
from eyefeatures.features.extractor import Extractor

# load your pd.DataFrame with gazes
data = ...

# define all required features & columns in single class
extractor = Extractor(
    features=[                                       # list of features
        eye_dist.SimpleDistances(
            methods=["euc", "eye", "man"],
            expected_paths_method="fwp",
        ),
        eye_measures.GriddedDistributionEntropy(),
        eye_stats.SaccadeFeatures(
            features_stats={
                'length': ['min', 'max'],
                'acceleration': ['min', 'mean']
            }
        )
    ],
    x='norm_pos_x',                                  # column with x-coordinate of fixations
    y='norm_pos_y',                                  # column with y-coordinate of fixations
    t='timestamp',                                   # column with timestamp
    duration='duration',                             # column with duration in ms
    dispersion='dispersion',                         # column with dispersion
    path_pk=['group'],                               # list of columns by which to get paths
    pk=['SUBJ', 'group'],                            # list of columns being primary key
    return_df=True                                   # return as pd.DataFrame
)

extractor.fit_transform(data)
