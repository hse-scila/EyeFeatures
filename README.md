# EyeFeatures Overview

## 1. Preprocessing

#### EyeFeature provides methods to preprocess a raw gaze dataset into a sequence of fixations. For demonstration purposes we use IVT:


```python
from preprocessing.fixation_extraction import IVT

x = 'norm_pos_x'
y = 'norm_pos_y'
t = 'gaze_timestamp'

ivt = IVT(x=x, y=y, t=t, pk=['Participant', 'tekst'], threshold=0.10)
data_ivt = ivt.transform(df_gaze)
data_ivt
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Participant</th>
      <th>tekst</th>
      <th>norm_pos_x</th>
      <th>norm_pos_y</th>
      <th>start_time</th>
      <th>end_time</th>
      <th>distance_min</th>
      <th>distance_max</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.242056</td>
      <td>0.510704</td>
      <td>317242.694809</td>
      <td>317242.715197</td>
      <td>0.000988</td>
      <td>0.012066</td>
      <td>0.020388</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.243933</td>
      <td>0.507985</td>
      <td>317242.728767</td>
      <td>317242.766803</td>
      <td>0.000971</td>
      <td>0.010593</td>
      <td>0.038036</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.244890</td>
      <td>0.507302</td>
      <td>317242.779258</td>
      <td>317242.779258</td>
      <td>0.002909</td>
      <td>0.002909</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.243674</td>
      <td>0.473291</td>
      <td>317242.805314</td>
      <td>317243.004508</td>
      <td>0.000679</td>
      <td>0.054160</td>
      <td>0.199194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.246791</td>
      <td>0.435542</td>
      <td>317243.017989</td>
      <td>317243.024352</td>
      <td>0.000697</td>
      <td>0.001653</td>
      <td>0.006363</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>193707</th>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.407710</td>
      <td>0.521161</td>
      <td>289275.675247</td>
      <td>289275.675247</td>
      <td>0.003031</td>
      <td>0.003031</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>193708</th>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.409729</td>
      <td>0.518900</td>
      <td>289275.689806</td>
      <td>289275.689806</td>
      <td>0.001765</td>
      <td>0.001765</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>193709</th>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.408336</td>
      <td>0.519858</td>
      <td>289275.703919</td>
      <td>289275.703919</td>
      <td>0.003316</td>
      <td>0.003316</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>193710</th>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.410257</td>
      <td>0.516873</td>
      <td>289275.715644</td>
      <td>289275.715644</td>
      <td>0.001421</td>
      <td>0.001421</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>193711</th>
      <td>10.0</td>
      <td>37.0</td>
      <td>0.447454</td>
      <td>0.495215</td>
      <td>289275.737404</td>
      <td>289275.759911</td>
      <td>0.012606</td>
      <td>0.049920</td>
      <td>0.022508</td>
    </tr>
  </tbody>
</table>
<p>193712 rows × 9 columns</p>
</div>



#### Visual Angle Dispersion (VAD) extraction is yet to be implemented. In further steps we use externally preprocessed dataset of fixations.

## 2. Visualization
#### The visualization module allows users to plot the graphic images of scanpaths over the original background.


```python
from visualization.static_visualization import scanpath_visualization
from PIL import Image

dur = 'duration'
dis = 'dispersion'
t = 'start_timestamp'

picture = "./background.jpg"
width, height = None, None
with Image.open(picture) as im:
    width, height = im.size


data = df_fix[(df_fix.Participant == 1) & (df_fix.tekst == 26)]
scanpath_visualization(
    data_=data, x=x, y=y, duration=dur, dispersion=dis, 
    time_stamps=t, img_path=picture, points_enumeration=True, 
    regression_color="red", micro_sac_color="yellow",  is_vectors=True
)
```


    
![png](output_12_0.png)
    



## 3. Feature Extraction
#### The features module has a variety of different features, ranging from simple aggregations to more complex scanpath algorithms.

#### User can either 
- test each feature independently by importing it from the relevant module.
- make use of Extractor class which is fully integrated into standard sklearn pipelines

### 3.1 Independent usage

#### Extracting scanpaths features using MultiMatch algorithm.


```python
from features.scanpath_dist import MultiMatchDist

dist = MultiMatchDist(x=x, y=y, duration=dur, path_pk=['tekst'], pk=['Participant', 'tekst'], return_df=True)
data = df_fix[df_fix.Participant < 6]
dist.fit_transform(data)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mm_shape</th>
      <th>mm_angle</th>
      <th>mm_len</th>
      <th>mm_pos</th>
      <th>mm_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.984409</td>
      <td>0.798314</td>
      <td>0.979722</td>
      <td>0.829876</td>
      <td>0.696303</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.988953</td>
      <td>0.912963</td>
      <td>0.985781</td>
      <td>0.846121</td>
      <td>0.736894</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.988904</td>
      <td>0.729273</td>
      <td>0.985388</td>
      <td>0.762853</td>
      <td>0.511047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.988411</td>
      <td>0.842714</td>
      <td>0.987568</td>
      <td>0.792416</td>
      <td>0.733453</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.995065</td>
      <td>0.975281</td>
      <td>0.996452</td>
      <td>0.808010</td>
      <td>0.938098</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>172</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.977153</td>
    </tr>
    <tr>
      <th>173</th>
      <td>0.990601</td>
      <td>0.856112</td>
      <td>0.989796</td>
      <td>0.950561</td>
      <td>0.648517</td>
    </tr>
    <tr>
      <th>174</th>
      <td>0.987199</td>
      <td>0.842388</td>
      <td>0.986055</td>
      <td>0.904987</td>
      <td>0.626429</td>
    </tr>
    <tr>
      <th>175</th>
      <td>0.995229</td>
      <td>0.933000</td>
      <td>0.997590</td>
      <td>0.976402</td>
      <td>0.734341</td>
    </tr>
    <tr>
      <th>176</th>
      <td>0.991163</td>
      <td>0.833923</td>
      <td>0.990731</td>
      <td>0.912807</td>
      <td>0.629052</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 5 columns</p>
</div>



#### Feature classes support <i>shift features</i> calculated using provided list of stats on fit which are used on transform.


```python
from features.stats import SaccadeFeatures

sac_feats_stats = {
    'length': ['min', 'max'],
    'speed': ['mean', 'kurtosis'],
    'acceleration': ['mean']
}
sac_feats_stats_shift = {'length': ['max'],
                         'acceleration': ['mean']}

sf = SaccadeFeatures(x=x, y=y, t=t,
                     pk=['Participant', 'tekst'],
                     features_stats=sac_feats_stats,
                     shift_features=sac_feats_stats_shift,
                     shift_pk=['tekst'])

sf.fit_transform(data).head()
```
<table border="1" class="dataframe">
<tr>
  <th></th>
  <th>sac_length_min</th>
  <th>sac_length_max</th>
  <th>sac_length_max_shift</th>
  <th>sac_acceleration_mean</th>
  <th>sac_acceleration_mean_shift</th>
  <th>sac_speed_mean</th>
  <th>sac_speed_kurtosis</th>
</tr>
<tr>
  <td>1_1</td>
  <td>0.002657</td>
  <td>0.433486</td>
  <td>-0.008934</td>
  <td>1.522857</td>
  <td>0.316149</td>
  <td>0.488956</td>
  <td>5.498018</td>
</tr>
<tr>
  <td>1_2</td>
  <td>0.003757</td>
  <td>0.321837</td>
  <td>-0.127119</td>
  <td>1.507806</td>
  <td>0.322016</td>
  <td>0.451773</td>
  <td>8.356764</td>
</tr>
<tr>
  <td>1_3</td>
  <td>0.003663</td>
  <td>0.365776</td>
  <td>-0.207636</td>
  <td>1.292694</td>
  <td>0.083609</td>
  <td>0.390753</td>
  <td>7.639049</td>
</tr>
<tr>
  <td>1_4</td>
  <td>0.000212</td>
  <td>0.342315</td>
  <td>-0.362239</td>
  <td>0.999054</td>
  <td>-0.038003</td>
  <td>0.333495</td>
  <td>9.593384</td>
</tr>
<tr>
  <td>1_5</td>
  <td>0.002705</td>
  <td>0.375434</td>
  <td>-0.094030</td>
  <td>1.187418</td>
  <td>0.089377</td>
  <td>0.360223</td>
  <td>9.717641</td>
</tr>
</table>
<p>5 rows × 7 columns</p>
</div>

### 3.2 Extractor usage


```python
from features.stats import SaccadeFeatures, FixationFeatures, RegressionFeatures, MicroSaccades
from features.measures import HurstExponent
from features.extractor import Extractor

from features.scanpath_dist import ScanMatchDist, EyeAnalysisDist
from features.scanpath_dist import EucDist, HauDist, DTWDist, MannanDist, DFDist, TDEDist

msv = 4.7  # MS_VELOCITY_THRESHOLD
msa = 1.2  # MS_AMPLITUDE_THRESHOLD

sac_feats_stats = {
    'length': ['min', 'max'],
    'speed': ['mean', 'kurtosis'],
    'acceleration': ['mean']
}
sac_feats_stats_shift = {'length': ['max'],
                         'acceleration': ['mean']}

fix_feats_stats = {'duration': ['sum'], 'vad': ['mean']}

features = [
    SaccadeFeatures(features_stats=sac_feats_stats,
                    shift_features=sac_feats_stats_shift,
                    shift_pk=['tekst']),
    FixationFeatures(features_stats=fix_feats_stats,
                     shift_pk=['Participant']),
    RegressionFeatures(features_stats=sac_feats_stats,
                       shift_features=sac_feats_stats_shift,
                       shift_pk=['tekst'],
                       rule=[1, 3]),
    MicroSaccades(features_stats=sac_feats_stats,
                  shift_features=sac_feats_stats_shift,
                  shift_pk=['Participant'],
                  min_dispersion=msa,
                  max_speed=msv),
    HurstExponent(var='duration', n_iters=10, fill_strategy='last'),
    ScanMatchDist(sub_mat=np.random.randint(0, 11, size=(20, 20)))
]

extractor = Extractor(features=features, x=x, y=y, t=t, duration=dur, 
                      dispersion=dis, pk=['tekst', 'Participant'], 
                      path_pk=['tekst'], return_df=True)

extractor.fit_transform(df_fix).head()
```

<!DOCTYPE html>
<html>
  <head>
    <title></title>
    <meta charset="UTF-8">
  </head>
<body>
<table border="1" style="border-collapse:collapse">
<tr>
  <th></th>
  <th>sac_length_min</th>
  <th>sac_length_max</th>
  <th>sac_length_max_shift</th>
  <th>sac_acceleration_mean</th>
  <th>sac_acceleration_mean_shift</th>
  <th>sac_speed_mean</th>
  <th>sac_speed_kurtosis</th>
  <th>fix_duration_sum</th>
  <th>fix_vad_mean</th>
  <th>reg_length_min</th>
  <th>reg_length_max</th>
  <th>reg_length_max_shift</th>
  <th>reg_acceleration_mean</th>
  <th>reg_acceleration_mean_shift</th>
  <th>reg_speed_mean</th>
  <th>reg_speed_kurtosis</th>
  <th>microsac_length_min</th>
  <th>microsac_length_max</th>
  <th>microsac_length_max_shift</th>
  <th>microsac_acceleration_mean</th>
  <th>microsac_acceleration_mean_shift</th>
  <th>microsac_speed_mean</th>
  <th>microsac_speed_kurtosis</th>
  <th>he_duration</th>
  <th>scan_match_dist</th>
</tr>
<tr>
  <td>1_1</td>
  <td>0.002657</td>
  <td>0.433486</td>
  <td>-0.008934</td>
  <td>100.310810</td>
  <td>7.586064</td>
  <td>3.354343</td>
  <td>5.209396</td>
  <td>29010.3960</td>
  <td>1.110338</td>
  <td>0.002657</td>
  <td>0.420153</td>
  <td>0.000000</td>
  <td>95.801067</td>
  <td>6.659842</td>
  <td>3.091449</td>
  <td>3.139267</td>
  <td>0.002657</td>
  <td>0.433486</td>
  <td>-0.063943</td>
  <td>100.310810</td>
  <td>7.220269</td>
  <td>3.354343</td>
  <td>5.209396</td>
  <td>0.012489</td>
  <td>144.0</td>
</tr>
<tr>
  <td>1_2</td>
  <td>0.002650</td>
  <td>0.442420</td>
  <td>0.000000</td>
  <td>85.101139</td>
  <td>-7.623606</td>
  <td>2.640718</td>
  <td>2.718729</td>
  <td>36462.2155</td>
  <td>1.214629</td>
  <td>0.005549</td>
  <td>0.214754</td>
  <td>-0.205400</td>
  <td>66.429542</td>
  <td>-22.711683</td>
  <td>2.177004</td>
  <td>-0.012081</td>
  <td>0.002650</td>
  <td>0.442420</td>
  <td>-0.055584</td>
  <td>85.101139</td>
  <td>-19.585891</td>
  <td>2.640718</td>
  <td>2.718729</td>
  <td>0.012318</td>
  <td>89.0</td>
</tr>
<tr>
  <td>2_1</td>
  <td>0.003757</td>
  <td>0.321837</td>
  <td>-0.127119</td>
  <td>82.774871</td>
  <td>0.180885</td>
  <td>2.790059</td>
  <td>1.062294</td>
  <td>23942.4375</td>
  <td>1.171651</td>
  <td>0.004125</td>
  <td>0.321837</td>
  <td>0.000000</td>
  <td>74.298355</td>
  <td>-0.826921</td>
  <td>2.869121</td>
  <td>1.625288</td>
  <td>0.003757</td>
  <td>0.321837</td>
  <td>-0.175592</td>
  <td>82.774871</td>
  <td>-10.315669</td>
  <td>2.790059</td>
  <td>1.062294</td>
  <td>0.017036</td>
  <td>76.0</td>
</tr>
<tr>
  <td>2_2</td>
  <td>0.003741</td>
  <td>0.448955</td>
  <td>0.000000</td>
  <td>83.321212</td>
  <td>0.727227</td>
  <td>2.585629</td>
  <td>4.223900</td>
  <td>18857.0335</td>
  <td>1.171989</td>
  <td>0.004011</td>
  <td>0.019028</td>
  <td>-0.302809</td>
  <td>91.250241</td>
  <td>16.124965</td>
  <td>1.478032</td>
  <td>1.285976</td>
  <td>0.003741</td>
  <td>0.448955</td>
  <td>-0.049048</td>
  <td>83.321212</td>
  <td>-21.365817</td>
  <td>2.585629</td>
  <td>4.223900</td>
  <td>0.019069</td>
  <td>93.0</td>
</tr>
<tr>
  <td>3_1</td>
  <td>0.003663</td>
  <td>0.365776</td>
  <td>-0.207636</td>
  <td>86.208709</td>
  <td>2.364303</td>
  <td>2.859377</td>
  <td>3.861619</td>
  <td>24894.4815</td>
  <td>1.167868</td>
  <td>0.003804</td>
  <td>0.314083</td>
  <td>0.000000</td>
  <td>74.972402</td>
  <td>-1.945541</td>
  <td>2.732318</td>
  <td>1.471709</td>
  <td>0.003663</td>
  <td>0.365776</td>
  <td>-0.131653</td>
  <td>86.208709</td>
  <td>-6.881831</td>
  <td>2.859377</td>
  <td>3.861619</td>
  <td>0.010373</td>
  <td>64.0</td>
</tr>
</table>
<p>5 rows × 26 columns</p>
</body>
</html>

### 3.3 Extracting features for each area of interest
Note: if the text does not have an area of interest, then corresponding value is `NaN`.

Same as previous example, just add `aoi` parameter to `Extractor` or concrete feature instance.

```python
extractor = Extractor(features=features, x=x, y=y, t=t, duration=dur, dispersion=dis,
                      aoi='AOI',
                      pk=['Participant', 'tekst'],
                      return_df=True)

extractor.fit_transform(data).head()
```

<!DOCTYPE html>
<html>
  <head>
    <title></title>
    <meta charset="UTF-8">
  </head>
<body>
<table border="1" style="border-collapse:collapse">
<tr>
  <th></th>
  <th>sac_length_aoi_2_min</th>
  <th>sac_length_aoi_2_max</th>
  <th>sac_acceleration_aoi_2_mean</th>
  <th>sac_speed_aoi_2_mean</th>
  <th>sac_speed_aoi_2_kurtosis</th>
  <th>...</th>
  <th>microsac_length_aoi_1_min</th>
  <th>microsac_length_aoi_1_max</th>
  <th>microsac_acceleration_aoi_1_mean</th>
  <th>microsac_speed_aoi_1_mean</th>
  <th>microsac_speed_aoi_1_kurtosis</th>
</tr>
<tr>
  <td>1_1</td>
  <td>0.011581</td>
  <td>0.420153</td>
  <td>88.408114</td>
  <td>3.316192</td>
  <td>2.582125</td>
  <td>...</td>
  <td>0.016593</td>
  <td>0.272665</td>
  <td>68.252408</td>
  <td>3.213086</td>
  <td>-0.278472</td>
</tr>
<tr>
  <td>1_2</td>
  <td>0.006408</td>
  <td>0.321837</td>
  <td>85.793710</td>
  <td>3.007439</td>
  <td>-0.464949</td>
  <td>...</td>
  <td>0.014979</td>
  <td>0.261686</td>
  <td>54.794634</td>
  <td>2.103520</td>
  <td>-0.316122</td>
</tr>
<tr>
  <td>1_3</td>
  <td>0.005065</td>
  <td>0.365776</td>
  <td>94.441713</td>
  <td>2.959531</td>
  <td>6.020307</td>
  <td>...</td>
  <td>0.003663</td>
  <td>0.254050</td>
  <td>76.291863</td>
  <td>2.827255</td>
  <td>0.763153</td>
</tr>
<tr>
  <td>1_4</td>
  <td>0.001980</td>
  <td>0.277172</td>
  <td>121.601653</td>
  <td>3.005369</td>
  <td>0.249804</td>
  <td>...</td>
  <td>0.041129</td>
  <td>0.272092</td>
  <td>53.823491</td>
  <td>2.801384</td>
  <td>2.951833</td>
</tr>
<tr>
  <td>1_5</td>
  <td>0.002705</td>
  <td>0.300800</td>
  <td>69.126721</td>
  <td>2.426858</td>
  <td>1.677840</td>
  <td>...</td>
  <td>0.020778</td>
  <td>0.104575</td>
  <td>68.725060</td>
  <td>2.409873</td>
  <td>0.137052</td>
</tr>
</table>
</body>
</html>


### 3.4 Integrating into `sklearn` pipeline


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('extractor', extractor),
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression())])
```


```python
X_train = df_fix[df_fix.Participant < 9].drop(columns="ACC")
X_test = df_fix[df_fix.Participant >= 9].drop(columns="ACC")

y_train = df_target[df_target.Participant < 9].ACC # binary class
y_test = df_target[df_target.Participant >= 9].ACC

pipe.fit(X_train, y_train).score(X_test, y_test)
```




    Accuracy on test: 0.8351351351351351

### 3.5 Heatmaps

Example usage of similarity matrix.

```python
from seaborn import heatmap
from features.scanpath_dist import hau_dist
from features.scanpath_complex import get_sim_matrix

def similarity_metric(path1, path2, c=5):
    return 1 / (1 + c * hau_dist(path1, path2))

x = 'norm_pos_x'
y = 'norm_pos_y'
t = 'start_timestamp'

scanpaths = []
texts = data.tekst.unique()
for t in texts:
    df_p = data[data.tekst == t]
    x_path, y_path = df_p[x], df_p[y]
    scanpaths.append(np.vstack([x_path, y_path]))

sim_matrix = get_sim_matrix(scanpaths, similarity_metric)
heatmap(sim_matrix);
```

![png](similarity_matrix.png)


## 4. Deep learning

```python
from deep.models import SimpleCNNclassifier
from deep.datasets import HeatMapDatasetLightning

# Having fixations dataframe with columns [x,y,Participant, tekst]
# and dataframe with class labels [label, Participant, tekst] we can train CNN for classification
#k - width and height of heatmap
data = HeatMapDatasetLightning(X, y, label_name='ACC', pk=['Participant','tekst'],
                             k=50, test_size=0.5, batch_size=2)
model = SimpleCNNclassifier(n_classes=2, shape = (50,50))
trainer = pl.Trainer(max_epochs=6)
trainer.fit(model=model, datamodule=data)
```
