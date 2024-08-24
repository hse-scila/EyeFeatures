# EyeFeatures: introduction to features with scanpaths and fixations


```python
import os
from tqdm import tqdm

import requests
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter("ignore")
```

## Getting simple dataset to work with. 


```python
def get_paris_dataset():
    '''
    Download and load the Paris experiment dataset from Zenodo.
    The dataset contains scanpaths data from 15 participants reading approximately 180 texts.
    The dataset is normalized and split into X (fixations data), Y (target), and other features.
    Deatiled description of variables and task can be found at: https://zenodo.org/records/4655840
    '''
    if not os.path.exists("data/em-y35-fasttext.csv"):
        url = "https://zenodo.org/records/4655840/files/em-y35-fasttext.csv?download=1"
        response = requests.get(url, stream=True)

        os.makedirs("data", exist_ok=True)
        with open("data/em-y35-fasttext.csv", "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=1024)):
                handle.write(data)

    df = pd.read_csv("data/em-y35-fasttext.csv")
    df.X = df.X / df.X.max()
    df.Y = df.Y / df.Y.max()
    df = df.rename(columns={'FDUR': 'duration', 'X': 'norm_pos_x', 'Y': 'norm_pos_y'})
    df['dispersion'] = df['duration']
    df['timestamp'] = df.duration.cumsum()  # timestamps of fixations
    df['timestamp'] /= 1e3                    # milliseconds

    return df.drop(columns=['Unnamed: 0'])
```


```python
data = get_paris_dataset()
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SUBJ</th>
      <th>SUBJ_NAME</th>
      <th>TEXT_NO</th>
      <th>TEXT</th>
      <th>ANSWER</th>
      <th>FIX_NUM</th>
      <th>FIX_LATENCY</th>
      <th>norm_pos_x</th>
      <th>norm_pos_y</th>
      <th>duration</th>
      <th>...</th>
      <th>WFREQ_RANK_FASTTEXT_2016</th>
      <th>COS_INST_FASTTEXT_2018</th>
      <th>COS_CUM_FASTTEXT_2018</th>
      <th>WFREQ_RANK_FASTTEXT_2018</th>
      <th>WFREQ_RANK_FASTTEXT_1618</th>
      <th>COS_INST_FASTTEXT_1618</th>
      <th>COS_CUM_FASTTEXT_1618</th>
      <th>TEXT_TYPE_2</th>
      <th>dispersion</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>s01</td>
      <td>1</td>
      <td>chasse_oiseaux-a1</td>
      <td>1</td>
      <td>1</td>
      <td>202</td>
      <td>0.376268</td>
      <td>0.384969</td>
      <td>96</td>
      <td>...</td>
      <td>8205.0</td>
      <td>0.186901</td>
      <td>0.186901</td>
      <td>5590.0</td>
      <td>6897.5</td>
      <td>0.185782</td>
      <td>0.185782</td>
      <td>a</td>
      <td>96</td>
      <td>0.096</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>s01</td>
      <td>1</td>
      <td>chasse_oiseaux-a1</td>
      <td>1</td>
      <td>2</td>
      <td>321</td>
      <td>0.437754</td>
      <td>0.383532</td>
      <td>129</td>
      <td>...</td>
      <td>8205.0</td>
      <td>0.186901</td>
      <td>0.186901</td>
      <td>5590.0</td>
      <td>6897.5</td>
      <td>0.185782</td>
      <td>0.185782</td>
      <td>a</td>
      <td>129</td>
      <td>0.225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>s01</td>
      <td>1</td>
      <td>chasse_oiseaux-a1</td>
      <td>1</td>
      <td>3</td>
      <td>477</td>
      <td>0.546146</td>
      <td>0.382957</td>
      <td>280</td>
      <td>...</td>
      <td>12071.0</td>
      <td>0.221362</td>
      <td>0.228615</td>
      <td>18406.0</td>
      <td>15238.5</td>
      <td>0.214195</td>
      <td>0.225632</td>
      <td>a</td>
      <td>280</td>
      <td>0.505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>s01</td>
      <td>1</td>
      <td>chasse_oiseaux-a1</td>
      <td>1</td>
      <td>4</td>
      <td>792</td>
      <td>0.706643</td>
      <td>0.399626</td>
      <td>278</td>
      <td>...</td>
      <td>1217.0</td>
      <td>0.256207</td>
      <td>0.254959</td>
      <td>2094.0</td>
      <td>1655.5</td>
      <td>0.213694</td>
      <td>0.247522</td>
      <td>a</td>
      <td>278</td>
      <td>0.783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>s01</td>
      <td>1</td>
      <td>chasse_oiseaux-a1</td>
      <td>1</td>
      <td>5</td>
      <td>1085</td>
      <td>0.724645</td>
      <td>0.397615</td>
      <td>266</td>
      <td>...</td>
      <td>1217.0</td>
      <td>0.256207</td>
      <td>0.268313</td>
      <td>2094.0</td>
      <td>1655.5</td>
      <td>0.213694</td>
      <td>0.254901</td>
      <td>a</td>
      <td>266</td>
      <td>1.049</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>39559</th>
      <td>15</td>
      <td>s21</td>
      <td>57</td>
      <td>conflit_israelo_palestinien-f2</td>
      <td>1</td>
      <td>10</td>
      <td>2268</td>
      <td>0.420385</td>
      <td>0.796091</td>
      <td>142</td>
      <td>...</td>
      <td>200185.0</td>
      <td>0.522610</td>
      <td>0.331133</td>
      <td>3706.0</td>
      <td>101945.5</td>
      <td>0.426449</td>
      <td>0.445799</td>
      <td>f+</td>
      <td>142</td>
      <td>7279.520</td>
    </tr>
    <tr>
      <th>39560</th>
      <td>15</td>
      <td>s21</td>
      <td>57</td>
      <td>conflit_israelo_palestinien-f2</td>
      <td>1</td>
      <td>11</td>
      <td>2442</td>
      <td>0.536004</td>
      <td>0.806581</td>
      <td>171</td>
      <td>...</td>
      <td>8832.0</td>
      <td>0.251470</td>
      <td>0.340826</td>
      <td>11554.0</td>
      <td>10193.0</td>
      <td>0.263324</td>
      <td>0.455220</td>
      <td>f+</td>
      <td>171</td>
      <td>7279.691</td>
    </tr>
    <tr>
      <th>39561</th>
      <td>15</td>
      <td>s21</td>
      <td>57</td>
      <td>conflit_israelo_palestinien-f2</td>
      <td>1</td>
      <td>12</td>
      <td>2638</td>
      <td>0.526749</td>
      <td>0.882885</td>
      <td>152</td>
      <td>...</td>
      <td>15043.0</td>
      <td>0.127237</td>
      <td>0.345502</td>
      <td>11359.0</td>
      <td>13201.0</td>
      <td>0.167958</td>
      <td>0.457360</td>
      <td>f+</td>
      <td>152</td>
      <td>7279.843</td>
    </tr>
    <tr>
      <th>39562</th>
      <td>15</td>
      <td>s21</td>
      <td>57</td>
      <td>conflit_israelo_palestinien-f2</td>
      <td>1</td>
      <td>13</td>
      <td>2827</td>
      <td>0.757860</td>
      <td>0.875126</td>
      <td>276</td>
      <td>...</td>
      <td>1245.0</td>
      <td>0.741338</td>
      <td>0.395152</td>
      <td>2263.0</td>
      <td>1754.0</td>
      <td>0.720622</td>
      <td>0.511603</td>
      <td>f+</td>
      <td>276</td>
      <td>7280.119</td>
    </tr>
    <tr>
      <th>39563</th>
      <td>15</td>
      <td>s21</td>
      <td>57</td>
      <td>conflit_israelo_palestinien-f2</td>
      <td>1</td>
      <td>14</td>
      <td>3126</td>
      <td>0.701952</td>
      <td>0.888346</td>
      <td>139</td>
      <td>...</td>
      <td>1245.0</td>
      <td>0.741338</td>
      <td>0.439672</td>
      <td>2263.0</td>
      <td>1754.0</td>
      <td>0.720622</td>
      <td>0.555105</td>
      <td>f+</td>
      <td>139</td>
      <td>7280.258</td>
    </tr>
  </tbody>
</table>
<p>39564 rows × 39 columns</p>
</div>



##### In order to extract features using the EyeFeatures methods, we only need the following columns: coordinates of fixations on the screen (that is x, y coordinates) and columns that identify the unique objects in the dataset. You can preprocess a dataset of raw gazes into the required format using a preprocessing module.


```python
data = data[['SUBJ', 'norm_pos_x', 'norm_pos_y', 'timestamp', 'duration', 'dispersion', 'ANSWER']]
data['group'] = 1                           # dummy column for grouping purposes (we operate with single group)
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SUBJ</th>
      <th>norm_pos_x</th>
      <th>norm_pos_y</th>
      <th>timestamp</th>
      <th>duration</th>
      <th>dispersion</th>
      <th>ANSWER</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.376268</td>
      <td>0.384969</td>
      <td>0.096</td>
      <td>96</td>
      <td>96</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.437754</td>
      <td>0.383532</td>
      <td>0.225</td>
      <td>129</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.546146</td>
      <td>0.382957</td>
      <td>0.505</td>
      <td>280</td>
      <td>280</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.706643</td>
      <td>0.399626</td>
      <td>0.783</td>
      <td>278</td>
      <td>278</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.724645</td>
      <td>0.397615</td>
      <td>1.049</td>
      <td>266</td>
      <td>266</td>
      <td>1</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>39559</th>
      <td>15</td>
      <td>0.420385</td>
      <td>0.796091</td>
      <td>7279.520</td>
      <td>142</td>
      <td>142</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39560</th>
      <td>15</td>
      <td>0.536004</td>
      <td>0.806581</td>
      <td>7279.691</td>
      <td>171</td>
      <td>171</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39561</th>
      <td>15</td>
      <td>0.526749</td>
      <td>0.882885</td>
      <td>7279.843</td>
      <td>152</td>
      <td>152</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39562</th>
      <td>15</td>
      <td>0.757860</td>
      <td>0.875126</td>
      <td>7280.119</td>
      <td>276</td>
      <td>276</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39563</th>
      <td>15</td>
      <td>0.701952</td>
      <td>0.888346</td>
      <td>7280.258</td>
      <td>139</td>
      <td>139</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>39564 rows × 8 columns</p>
</div>



## Scanpath Distances

##### Let us describe the core idea of extrator-classes in this module. Each class calculates the "expected paths" for each path-group which are further used in distance functions. That is the resulting features for each group are simply the distances between two scanpaths: expected and given one.


```python
import eyetracking.features.scanpath_dist as eye_dist
```

##### See the example of calculating some basic distances (Euclidean, Eye and Mannan). Note that the primary key (`path_pk`) is set to 'group' so there is a separate expected path for each unique group. The primary key (`pk`) is also set to 'SUBJ' and 'group' which defines the way to distinguish between unique paths.


```python
transformer = eye_dist.SimpleDistances(
    x='norm_pos_x',
    y='norm_pos_y',
    path_pk=['group'],
    pk=['SUBJ', 'group'],
    methods=["euc", "eye", "man"],
    expected_paths_method="fwp",
    return_df=True
)

transformer.fit_transform(data)
```

    100%|██████████| 15/15 [00:00<00:00, 174.15it/s]
    100%|██████████| 15/15 [00:03<00:00,  4.56it/s]
    100%|██████████| 15/15 [00:03<00:00,  4.60it/s]





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>euc_dist</th>
      <th>eye_dist</th>
      <th>man_dist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>200.623121</td>
      <td>0.058312</td>
      <td>0.015358</td>
    </tr>
    <tr>
      <th>2_1</th>
      <td>122.518147</td>
      <td>0.064941</td>
      <td>0.017263</td>
    </tr>
    <tr>
      <th>3_1</th>
      <td>1162.275204</td>
      <td>0.071679</td>
      <td>0.017920</td>
    </tr>
    <tr>
      <th>4_1</th>
      <td>412.366120</td>
      <td>0.058348</td>
      <td>0.015121</td>
    </tr>
    <tr>
      <th>5_1</th>
      <td>261.495549</td>
      <td>0.066644</td>
      <td>0.017448</td>
    </tr>
    <tr>
      <th>6_1</th>
      <td>72.612316</td>
      <td>0.051765</td>
      <td>0.014200</td>
    </tr>
    <tr>
      <th>7_1</th>
      <td>112.304684</td>
      <td>0.067186</td>
      <td>0.017750</td>
    </tr>
    <tr>
      <th>8_1</th>
      <td>198.174110</td>
      <td>0.068108</td>
      <td>0.018178</td>
    </tr>
    <tr>
      <th>9_1</th>
      <td>255.661709</td>
      <td>0.061571</td>
      <td>0.016137</td>
    </tr>
    <tr>
      <th>10_1</th>
      <td>143.557332</td>
      <td>0.050267</td>
      <td>0.013282</td>
    </tr>
    <tr>
      <th>11_1</th>
      <td>525.852980</td>
      <td>0.063494</td>
      <td>0.016458</td>
    </tr>
    <tr>
      <th>12_1</th>
      <td>82.064542</td>
      <td>0.061822</td>
      <td>0.016428</td>
    </tr>
    <tr>
      <th>13_1</th>
      <td>120.106331</td>
      <td>0.051667</td>
      <td>0.014180</td>
    </tr>
    <tr>
      <th>14_1</th>
      <td>39.239737</td>
      <td>0.045353</td>
      <td>0.012841</td>
    </tr>
    <tr>
      <th>15_1</th>
      <td>86.587902</td>
      <td>0.046960</td>
      <td>0.013304</td>
    </tr>
  </tbody>
</table>
</div>



##### Note that the expected paths are recalculated for each new distance class when the `fit` method is called, which takes up most of the runtime. To speed up the process, you can get the expected paths from the previous class and use them with the new class. It is important that the primary keys for the new class match those of the previous class from which you obtained the expected paths.

##### The same principle can be applied to the filling path (`fill_path`), which is used when no expected path is found for a particular group when `transform` is called. It is calculated as the mean of all known expected paths (basically the expected path over the expected paths).


```python
expected_paths = transformer.expected_paths
expected_paths
```




    {'1':          x_est     y_est
     0     0.362373  0.408229
     1     0.449865  0.404311
     2     0.547887  0.407606
     3     0.623107  0.411228
     4     0.616489  0.429458
     ...        ...       ...
     4497  0.037297  0.037889
     4498  0.047262  0.038253
     4499  0.053338  0.037946
     4500  0.027975  0.042276
     4501  0.041751  0.043311
     
     [4502 rows x 2 columns]}




```python
fill_path = transformer.fill_path
fill_path
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x_est</th>
      <th>y_est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.362373</td>
      <td>0.408229</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.449865</td>
      <td>0.404311</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.547887</td>
      <td>0.407606</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.623107</td>
      <td>0.411228</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.616489</td>
      <td>0.429458</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4497</th>
      <td>0.037297</td>
      <td>0.037889</td>
    </tr>
    <tr>
      <th>4498</th>
      <td>0.047262</td>
      <td>0.038253</td>
    </tr>
    <tr>
      <th>4499</th>
      <td>0.053338</td>
      <td>0.037946</td>
    </tr>
    <tr>
      <th>4500</th>
      <td>0.027975</td>
      <td>0.042276</td>
    </tr>
    <tr>
      <th>4501</th>
      <td>0.041751</td>
      <td>0.043311</td>
    </tr>
  </tbody>
</table>
<p>4502 rows × 2 columns</p>
</div>



##### Let's calculate some complex distance such like MultiMatch which outputs as many as 5 scanpath features. Besides x and y coordinates one has to pass fixations duration column as well. The rest is pretty much the same except we pass precalculated expected and filling paths to optimze the inference.

##### ScanMatch encodes the spatial and temporal characteristics of each scanpath into a sequence of symbols. The visual space is divided into a grid, and each grid cell is assigned a unique symbol. As the eye moves through different regions, the sequence of visited grid cells generates a symbolic representation of the scanpath. Two hyperparameters are temporal bin for quantifying fixations (`t_bin`) and substitute cost matrix (`sub_mat`) which must be of shape 20x20 for this method.  


```python
scanmatch = eye_dist.ScanMatchDist(
    t_bin=30,
    sub_mat=np.random.random((20, 20)),
    x='norm_pos_x',
    y='norm_pos_y',
    duration='duration',
    path_pk=['group'],
    pk=['SUBJ', 'group'],
    expected_paths=expected_paths,
    fill_path=fill_path,
    return_df=True
)

scanmatch.fit_transform(data)
```

    100%|██████████| 15/15 [00:03<00:00,  4.55it/s]





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>scan_match_dist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>0.933843</td>
    </tr>
    <tr>
      <th>2_1</th>
      <td>0.352993</td>
    </tr>
    <tr>
      <th>3_1</th>
      <td>0.620893</td>
    </tr>
    <tr>
      <th>4_1</th>
      <td>0.691871</td>
    </tr>
    <tr>
      <th>5_1</th>
      <td>0.926866</td>
    </tr>
    <tr>
      <th>6_1</th>
      <td>0.891209</td>
    </tr>
    <tr>
      <th>7_1</th>
      <td>0.239042</td>
    </tr>
    <tr>
      <th>8_1</th>
      <td>0.584023</td>
    </tr>
    <tr>
      <th>9_1</th>
      <td>0.399081</td>
    </tr>
    <tr>
      <th>10_1</th>
      <td>0.694188</td>
    </tr>
    <tr>
      <th>11_1</th>
      <td>0.775750</td>
    </tr>
    <tr>
      <th>12_1</th>
      <td>0.239042</td>
    </tr>
    <tr>
      <th>13_1</th>
      <td>0.791370</td>
    </tr>
    <tr>
      <th>14_1</th>
      <td>0.738963</td>
    </tr>
    <tr>
      <th>15_1</th>
      <td>0.928938</td>
    </tr>
  </tbody>
</table>
</div>



##### There are also methods in `scanpath_complex` module that return features not in the form of numbers that can be used for inference. This is usually some kind of structure (for instance, a matrix) that can be used for analyzing data or further feature extracting.


```python
import eyetracking.features.scanpath_complex as eye_complex
```

##### Let's see one of the possible usecases. 

##### We get the list of scanpaths of form `(x_coord, y_coord)` in order to calculate the pairwise distance matrix. One can use his own custom metric or those implemented in `scanpath_dist`.


```python
list_of_scanpaths = [scanpath[['norm_pos_x', 'norm_pos_y']].reset_index(drop=True) for _, scanpath in data.groupby('SUBJ')]
len(list_of_scanpaths)
```




    15




```python
euc_matrix = eye_complex.get_dist_matrix(list_of_scanpaths, dist_metric=eye_dist.calc_euc_dist)
euc_matrix
```

    100%|██████████| 15/15 [00:00<00:00, 3816.01it/s]





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>q</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
    <tr>
      <th>p</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>216.891445</td>
      <td>241.739752</td>
      <td>236.518306</td>
      <td>253.582253</td>
      <td>157.548907</td>
      <td>195.508599</td>
      <td>237.403918</td>
      <td>245.282704</td>
      <td>224.949497</td>
      <td>263.351295</td>
      <td>166.859744</td>
      <td>206.992784</td>
      <td>80.401605</td>
      <td>167.280796</td>
    </tr>
    <tr>
      <th>1</th>
      <td>216.891445</td>
      <td>0.000000</td>
      <td>212.105848</td>
      <td>213.882556</td>
      <td>226.860971</td>
      <td>158.614094</td>
      <td>202.045338</td>
      <td>224.791482</td>
      <td>227.851853</td>
      <td>212.250657</td>
      <td>232.041838</td>
      <td>177.053103</td>
      <td>218.198329</td>
      <td>83.403611</td>
      <td>176.055411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>241.739752</td>
      <td>212.105848</td>
      <td>0.000000</td>
      <td>294.448064</td>
      <td>273.577526</td>
      <td>148.392904</td>
      <td>189.853201</td>
      <td>235.524909</td>
      <td>281.726542</td>
      <td>239.653290</td>
      <td>316.759147</td>
      <td>174.367230</td>
      <td>206.031155</td>
      <td>84.577833</td>
      <td>162.951707</td>
    </tr>
    <tr>
      <th>3</th>
      <td>236.518306</td>
      <td>213.882556</td>
      <td>294.448064</td>
      <td>0.000000</td>
      <td>272.753285</td>
      <td>154.945192</td>
      <td>192.192167</td>
      <td>242.065233</td>
      <td>266.397787</td>
      <td>226.716470</td>
      <td>298.556201</td>
      <td>169.659725</td>
      <td>209.946643</td>
      <td>76.707918</td>
      <td>173.897121</td>
    </tr>
    <tr>
      <th>4</th>
      <td>253.582253</td>
      <td>226.860971</td>
      <td>273.577526</td>
      <td>272.753285</td>
      <td>0.000000</td>
      <td>165.565762</td>
      <td>208.633328</td>
      <td>253.776842</td>
      <td>280.771001</td>
      <td>231.457743</td>
      <td>290.333419</td>
      <td>184.852322</td>
      <td>220.774162</td>
      <td>84.520301</td>
      <td>172.443492</td>
    </tr>
    <tr>
      <th>5</th>
      <td>157.548907</td>
      <td>158.614094</td>
      <td>148.392904</td>
      <td>154.945192</td>
      <td>165.565762</td>
      <td>0.000000</td>
      <td>144.433993</td>
      <td>162.285554</td>
      <td>158.155702</td>
      <td>151.596783</td>
      <td>167.261651</td>
      <td>150.152013</td>
      <td>157.527088</td>
      <td>79.499392</td>
      <td>160.638707</td>
    </tr>
    <tr>
      <th>6</th>
      <td>195.508599</td>
      <td>202.045338</td>
      <td>189.853201</td>
      <td>192.192167</td>
      <td>208.633328</td>
      <td>144.433993</td>
      <td>0.000000</td>
      <td>203.690246</td>
      <td>206.412659</td>
      <td>193.931330</td>
      <td>208.825800</td>
      <td>159.639914</td>
      <td>199.382172</td>
      <td>81.846284</td>
      <td>158.035375</td>
    </tr>
    <tr>
      <th>7</th>
      <td>237.403918</td>
      <td>224.791482</td>
      <td>235.524909</td>
      <td>242.065233</td>
      <td>253.776842</td>
      <td>162.285554</td>
      <td>203.690246</td>
      <td>0.000000</td>
      <td>264.774150</td>
      <td>239.876712</td>
      <td>252.577683</td>
      <td>177.639797</td>
      <td>225.430046</td>
      <td>92.829167</td>
      <td>168.774300</td>
    </tr>
    <tr>
      <th>8</th>
      <td>245.282704</td>
      <td>227.851853</td>
      <td>281.726542</td>
      <td>266.397787</td>
      <td>280.771001</td>
      <td>158.155702</td>
      <td>206.412659</td>
      <td>264.774150</td>
      <td>0.000000</td>
      <td>232.927866</td>
      <td>282.382338</td>
      <td>180.432626</td>
      <td>229.559072</td>
      <td>78.292232</td>
      <td>178.308103</td>
    </tr>
    <tr>
      <th>9</th>
      <td>224.949497</td>
      <td>212.250657</td>
      <td>239.653290</td>
      <td>226.716470</td>
      <td>231.457743</td>
      <td>151.596783</td>
      <td>193.931330</td>
      <td>239.876712</td>
      <td>232.927866</td>
      <td>0.000000</td>
      <td>247.045637</td>
      <td>169.114434</td>
      <td>208.023160</td>
      <td>77.574107</td>
      <td>173.817330</td>
    </tr>
    <tr>
      <th>10</th>
      <td>263.351295</td>
      <td>232.041838</td>
      <td>316.759147</td>
      <td>298.556201</td>
      <td>290.333419</td>
      <td>167.261651</td>
      <td>208.825800</td>
      <td>252.577683</td>
      <td>282.382338</td>
      <td>247.045637</td>
      <td>0.000000</td>
      <td>192.220963</td>
      <td>220.629479</td>
      <td>91.980945</td>
      <td>184.849111</td>
    </tr>
    <tr>
      <th>11</th>
      <td>166.859744</td>
      <td>177.053103</td>
      <td>174.367230</td>
      <td>169.659725</td>
      <td>184.852322</td>
      <td>150.152013</td>
      <td>159.639914</td>
      <td>177.639797</td>
      <td>180.432626</td>
      <td>169.114434</td>
      <td>192.220963</td>
      <td>0.000000</td>
      <td>180.191023</td>
      <td>80.406245</td>
      <td>158.973206</td>
    </tr>
    <tr>
      <th>12</th>
      <td>206.992784</td>
      <td>218.198329</td>
      <td>206.031155</td>
      <td>209.946643</td>
      <td>220.774162</td>
      <td>157.527088</td>
      <td>199.382172</td>
      <td>225.430046</td>
      <td>229.559072</td>
      <td>208.023160</td>
      <td>220.629479</td>
      <td>180.191023</td>
      <td>0.000000</td>
      <td>87.219790</td>
      <td>176.826045</td>
    </tr>
    <tr>
      <th>13</th>
      <td>80.401605</td>
      <td>83.403611</td>
      <td>84.577833</td>
      <td>76.707918</td>
      <td>84.520301</td>
      <td>79.499392</td>
      <td>81.846284</td>
      <td>92.829167</td>
      <td>78.292232</td>
      <td>77.574107</td>
      <td>91.980945</td>
      <td>80.406245</td>
      <td>87.219790</td>
      <td>0.000000</td>
      <td>89.854903</td>
    </tr>
    <tr>
      <th>14</th>
      <td>167.280796</td>
      <td>176.055411</td>
      <td>162.951707</td>
      <td>173.897121</td>
      <td>172.443492</td>
      <td>160.638707</td>
      <td>158.035375</td>
      <td>168.774300</td>
      <td>178.308103</td>
      <td>173.817330</td>
      <td>184.849111</td>
      <td>158.973206</td>
      <td>176.826045</td>
      <td>89.854903</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
eye_matrix = eye_complex.get_dist_matrix(list_of_scanpaths, dist_metric=eye_dist.calc_eye_dist)
eye_matrix
```

    100%|██████████| 15/15 [00:08<00:00,  1.85it/s]





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>q</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
    <tr>
      <th>p</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000093</td>
      <td>0.000095</td>
      <td>0.000090</td>
      <td>0.000188</td>
      <td>0.000130</td>
      <td>0.000138</td>
      <td>0.000114</td>
      <td>0.000103</td>
      <td>0.000106</td>
      <td>0.000088</td>
      <td>0.000071</td>
      <td>0.000106</td>
      <td>0.000325</td>
      <td>0.000140</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000093</td>
      <td>0.000000</td>
      <td>0.000095</td>
      <td>0.000056</td>
      <td>0.000086</td>
      <td>0.000149</td>
      <td>0.000180</td>
      <td>0.000109</td>
      <td>0.000098</td>
      <td>0.000130</td>
      <td>0.000116</td>
      <td>0.000101</td>
      <td>0.000174</td>
      <td>0.000417</td>
      <td>0.000224</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000095</td>
      <td>0.000095</td>
      <td>0.000000</td>
      <td>0.000076</td>
      <td>0.000094</td>
      <td>0.000112</td>
      <td>0.000100</td>
      <td>0.000065</td>
      <td>0.000117</td>
      <td>0.000122</td>
      <td>0.000106</td>
      <td>0.000089</td>
      <td>0.000105</td>
      <td>0.000330</td>
      <td>0.000117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000090</td>
      <td>0.000056</td>
      <td>0.000076</td>
      <td>0.000000</td>
      <td>0.000074</td>
      <td>0.000096</td>
      <td>0.000123</td>
      <td>0.000110</td>
      <td>0.000102</td>
      <td>0.000129</td>
      <td>0.000121</td>
      <td>0.000070</td>
      <td>0.000166</td>
      <td>0.000376</td>
      <td>0.000144</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000188</td>
      <td>0.000086</td>
      <td>0.000094</td>
      <td>0.000074</td>
      <td>0.000000</td>
      <td>0.000126</td>
      <td>0.000239</td>
      <td>0.000126</td>
      <td>0.000116</td>
      <td>0.000144</td>
      <td>0.000131</td>
      <td>0.000114</td>
      <td>0.000172</td>
      <td>0.000383</td>
      <td>0.000210</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000130</td>
      <td>0.000149</td>
      <td>0.000112</td>
      <td>0.000096</td>
      <td>0.000126</td>
      <td>0.000000</td>
      <td>0.000129</td>
      <td>0.000153</td>
      <td>0.000098</td>
      <td>0.000108</td>
      <td>0.000092</td>
      <td>0.000130</td>
      <td>0.000141</td>
      <td>0.000345</td>
      <td>0.000188</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000138</td>
      <td>0.000180</td>
      <td>0.000100</td>
      <td>0.000123</td>
      <td>0.000239</td>
      <td>0.000129</td>
      <td>0.000000</td>
      <td>0.000133</td>
      <td>0.000156</td>
      <td>0.000167</td>
      <td>0.000140</td>
      <td>0.000150</td>
      <td>0.000208</td>
      <td>0.000431</td>
      <td>0.000217</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.000114</td>
      <td>0.000109</td>
      <td>0.000065</td>
      <td>0.000110</td>
      <td>0.000126</td>
      <td>0.000153</td>
      <td>0.000133</td>
      <td>0.000000</td>
      <td>0.000131</td>
      <td>0.000118</td>
      <td>0.000092</td>
      <td>0.000101</td>
      <td>0.000115</td>
      <td>0.000391</td>
      <td>0.000155</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000103</td>
      <td>0.000098</td>
      <td>0.000117</td>
      <td>0.000102</td>
      <td>0.000116</td>
      <td>0.000098</td>
      <td>0.000156</td>
      <td>0.000131</td>
      <td>0.000000</td>
      <td>0.000086</td>
      <td>0.000086</td>
      <td>0.000088</td>
      <td>0.000103</td>
      <td>0.000294</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.000106</td>
      <td>0.000130</td>
      <td>0.000122</td>
      <td>0.000129</td>
      <td>0.000144</td>
      <td>0.000108</td>
      <td>0.000167</td>
      <td>0.000118</td>
      <td>0.000086</td>
      <td>0.000000</td>
      <td>0.000092</td>
      <td>0.000089</td>
      <td>0.000099</td>
      <td>0.000245</td>
      <td>0.000130</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000088</td>
      <td>0.000116</td>
      <td>0.000106</td>
      <td>0.000121</td>
      <td>0.000131</td>
      <td>0.000092</td>
      <td>0.000140</td>
      <td>0.000092</td>
      <td>0.000086</td>
      <td>0.000092</td>
      <td>0.000000</td>
      <td>0.000095</td>
      <td>0.000080</td>
      <td>0.000267</td>
      <td>0.000111</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.000071</td>
      <td>0.000101</td>
      <td>0.000089</td>
      <td>0.000070</td>
      <td>0.000114</td>
      <td>0.000130</td>
      <td>0.000150</td>
      <td>0.000101</td>
      <td>0.000088</td>
      <td>0.000089</td>
      <td>0.000095</td>
      <td>0.000000</td>
      <td>0.000113</td>
      <td>0.000367</td>
      <td>0.000151</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.000106</td>
      <td>0.000174</td>
      <td>0.000105</td>
      <td>0.000166</td>
      <td>0.000172</td>
      <td>0.000141</td>
      <td>0.000208</td>
      <td>0.000115</td>
      <td>0.000103</td>
      <td>0.000099</td>
      <td>0.000080</td>
      <td>0.000113</td>
      <td>0.000000</td>
      <td>0.000317</td>
      <td>0.000157</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.000325</td>
      <td>0.000417</td>
      <td>0.000330</td>
      <td>0.000376</td>
      <td>0.000383</td>
      <td>0.000345</td>
      <td>0.000431</td>
      <td>0.000391</td>
      <td>0.000294</td>
      <td>0.000245</td>
      <td>0.000267</td>
      <td>0.000367</td>
      <td>0.000317</td>
      <td>0.000000</td>
      <td>0.000380</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.000140</td>
      <td>0.000224</td>
      <td>0.000117</td>
      <td>0.000144</td>
      <td>0.000210</td>
      <td>0.000188</td>
      <td>0.000217</td>
      <td>0.000155</td>
      <td>0.000174</td>
      <td>0.000130</td>
      <td>0.000111</td>
      <td>0.000151</td>
      <td>0.000157</td>
      <td>0.000380</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### Now we can calculate the compromise matrix using these pairwise distances matrices. 

##### The compromise matrix serves as a summary that captures the most common structure or pattern shared across all the individual distance matrices. It is used in applications where you need to summarize information from multiple sources (in our case these are different metrics), providing a single matrix that best represents the consensus of all input matrices.


```python
eye_complex.get_compromise_matrix([euc_matrix, eye_matrix])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>73.176261</td>
      <td>-6.152585</td>
      <td>-8.346088</td>
      <td>-7.278371</td>
      <td>-11.161435</td>
      <td>-0.856152</td>
      <td>-4.193279</td>
      <td>-8.705100</td>
      <td>-8.383300</td>
      <td>-7.896376</td>
      <td>-11.576780</td>
      <td>0.682575</td>
      <td>-3.485281</td>
      <td>4.086666</td>
      <td>0.089247</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-6.152585</td>
      <td>67.884046</td>
      <td>-0.515029</td>
      <td>-1.921520</td>
      <td>-4.360107</td>
      <td>-3.878867</td>
      <td>-9.150487</td>
      <td>-6.892036</td>
      <td>-4.866669</td>
      <td>-6.052774</td>
      <td>-3.153333</td>
      <td>-5.567439</td>
      <td>-10.093171</td>
      <td>0.379157</td>
      <td>-5.659185</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-8.346088</td>
      <td>-0.515029</td>
      <td>81.067447</td>
      <td>-23.814035</td>
      <td>-14.285205</td>
      <td>6.326584</td>
      <td>1.751813</td>
      <td>-4.095159</td>
      <td>-17.322554</td>
      <td>-9.149364</td>
      <td>-26.513720</td>
      <td>1.973866</td>
      <td>0.800300</td>
      <td>6.555738</td>
      <td>5.565413</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-7.278371</td>
      <td>-1.921520</td>
      <td>-23.814035</td>
      <td>79.510760</td>
      <td>-14.772128</td>
      <td>3.231662</td>
      <td>0.146512</td>
      <td>-7.185873</td>
      <td>-12.681359</td>
      <td>-5.353854</td>
      <td>-20.856356</td>
      <td>2.859883</td>
      <td>-1.362400</td>
      <td>8.559813</td>
      <td>0.917271</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-11.161435</td>
      <td>-4.360107</td>
      <td>-14.285205</td>
      <td>-14.772128</td>
      <td>83.810733</td>
      <td>1.626700</td>
      <td>-3.516370</td>
      <td>-9.176571</td>
      <td>-15.613076</td>
      <td>-4.880165</td>
      <td>-15.799181</td>
      <td>-0.361540</td>
      <td>-3.040521</td>
      <td>7.947703</td>
      <td>3.581170</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.856152</td>
      <td>-3.878867</td>
      <td>6.326584</td>
      <td>3.231662</td>
      <td>1.626700</td>
      <td>36.515428</td>
      <td>-4.466092</td>
      <td>-0.477178</td>
      <td>4.090333</td>
      <td>-0.292692</td>
      <td>4.065622</td>
      <td>-11.740787</td>
      <td>-4.326944</td>
      <td>-13.924776</td>
      <td>-15.892853</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-4.193279</td>
      <td>-9.150487</td>
      <td>1.751813</td>
      <td>0.146512</td>
      <td>-3.516370</td>
      <td>-4.466092</td>
      <td>56.682736</td>
      <td>-5.032286</td>
      <td>-2.887445</td>
      <td>-5.176582</td>
      <td>-0.545887</td>
      <td>-5.011619</td>
      <td>-9.041321</td>
      <td>-4.670904</td>
      <td>-4.888792</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-8.705100</td>
      <td>-6.892036</td>
      <td>-4.095159</td>
      <td>-7.185873</td>
      <td>-9.176571</td>
      <td>-0.477178</td>
      <td>-5.032286</td>
      <td>77.283540</td>
      <td>-13.220937</td>
      <td>-11.120308</td>
      <td>-5.714095</td>
      <td>-1.075119</td>
      <td>-7.950200</td>
      <td>1.746477</td>
      <td>1.614849</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-8.383300</td>
      <td>-4.866669</td>
      <td>-17.322554</td>
      <td>-12.681359</td>
      <td>-15.613076</td>
      <td>4.090333</td>
      <td>-2.887445</td>
      <td>-13.220937</td>
      <td>83.498276</td>
      <td>-5.556140</td>
      <td>-13.144262</td>
      <td>1.044839</td>
      <td>-6.302660</td>
      <td>9.993461</td>
      <td>1.351501</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-7.896376</td>
      <td>-6.052774</td>
      <td>-9.149364</td>
      <td>-5.353854</td>
      <td>-4.880165</td>
      <td>-0.292692</td>
      <td>-5.176582</td>
      <td>-11.120308</td>
      <td>-5.556140</td>
      <td>70.094378</td>
      <td>-7.352802</td>
      <td>-1.655526</td>
      <td>-5.390513</td>
      <td>3.545425</td>
      <td>-3.762704</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-11.576780</td>
      <td>-3.153333</td>
      <td>-26.513720</td>
      <td>-20.856356</td>
      <td>-15.799181</td>
      <td>4.065622</td>
      <td>-0.545887</td>
      <td>-5.714095</td>
      <td>-13.144262</td>
      <td>-7.352802</td>
      <td>89.887728</td>
      <td>0.071756</td>
      <td>0.049162</td>
      <td>8.348505</td>
      <td>2.233654</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.682575</td>
      <td>-5.567439</td>
      <td>1.973866</td>
      <td>2.859883</td>
      <td>-0.361540</td>
      <td>-11.740787</td>
      <td>-5.011619</td>
      <td>-1.075119</td>
      <td>1.044839</td>
      <td>-1.655526</td>
      <td>0.071756</td>
      <td>46.176597</td>
      <td>-7.509261</td>
      <td>-9.414821</td>
      <td>-10.473411</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-3.485281</td>
      <td>-10.093171</td>
      <td>0.800300</td>
      <td>-1.362400</td>
      <td>-3.040521</td>
      <td>-4.326944</td>
      <td>-9.041321</td>
      <td>-7.950200</td>
      <td>-6.302660</td>
      <td>-5.390513</td>
      <td>0.049162</td>
      <td>-7.509261</td>
      <td>66.219254</td>
      <td>-1.802426</td>
      <td>-6.764017</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4.086666</td>
      <td>0.379157</td>
      <td>6.555738</td>
      <td>8.559813</td>
      <td>7.947703</td>
      <td>-13.924776</td>
      <td>-4.670904</td>
      <td>1.746477</td>
      <td>9.993461</td>
      <td>3.545425</td>
      <td>8.348505</td>
      <td>-9.414821</td>
      <td>-1.802426</td>
      <td>-8.150178</td>
      <td>-13.199870</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.089247</td>
      <td>-5.659185</td>
      <td>5.565413</td>
      <td>0.917271</td>
      <td>3.581170</td>
      <td>-15.892853</td>
      <td>-4.888792</td>
      <td>1.614849</td>
      <td>1.351501</td>
      <td>-3.762704</td>
      <td>2.233654</td>
      <td>-10.473411</td>
      <td>-6.764017</td>
      <td>-13.199870</td>
      <td>45.287719</td>
    </tr>
  </tbody>
</table>
</div>



##### One can also calculate a similarity matrix of the scanpaths. See the example of using it with a custom metric. 


```python
def sim(p, q) -> float:
    return 1 / eye_dist.calc_euc_dist(p, q)

sim_matrix = eye_complex.get_sim_matrix(list_of_scanpaths, sim_metric=sim)
pd.DataFrame(sim_matrix)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.502305</td>
      <td>0.502068</td>
      <td>0.502114</td>
      <td>0.501972</td>
      <td>0.503174</td>
      <td>0.502557</td>
      <td>0.502106</td>
      <td>0.502038</td>
      <td>0.502223</td>
      <td>0.501899</td>
      <td>0.502997</td>
      <td>0.502416</td>
      <td>0.506219</td>
      <td>0.502989</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.502305</td>
      <td>1.000000</td>
      <td>0.502357</td>
      <td>0.502338</td>
      <td>0.502204</td>
      <td>0.503152</td>
      <td>0.502475</td>
      <td>0.502224</td>
      <td>0.502194</td>
      <td>0.502356</td>
      <td>0.502155</td>
      <td>0.502824</td>
      <td>0.502291</td>
      <td>0.505995</td>
      <td>0.502840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.502068</td>
      <td>0.502357</td>
      <td>1.000000</td>
      <td>0.501698</td>
      <td>0.501828</td>
      <td>0.503369</td>
      <td>0.502634</td>
      <td>0.502123</td>
      <td>0.501775</td>
      <td>0.502086</td>
      <td>0.501578</td>
      <td>0.502868</td>
      <td>0.502427</td>
      <td>0.505912</td>
      <td>0.503068</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.502114</td>
      <td>0.502338</td>
      <td>0.501698</td>
      <td>1.000000</td>
      <td>0.501833</td>
      <td>0.503227</td>
      <td>0.502602</td>
      <td>0.502066</td>
      <td>0.501877</td>
      <td>0.502205</td>
      <td>0.501675</td>
      <td>0.502947</td>
      <td>0.502382</td>
      <td>0.506518</td>
      <td>0.502875</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.501972</td>
      <td>0.502204</td>
      <td>0.501828</td>
      <td>0.501833</td>
      <td>1.000000</td>
      <td>0.503020</td>
      <td>0.502397</td>
      <td>0.501970</td>
      <td>0.501781</td>
      <td>0.502160</td>
      <td>0.501722</td>
      <td>0.502705</td>
      <td>0.502265</td>
      <td>0.505916</td>
      <td>0.502900</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.503174</td>
      <td>0.503152</td>
      <td>0.503369</td>
      <td>0.503227</td>
      <td>0.503020</td>
      <td>1.000000</td>
      <td>0.503462</td>
      <td>0.503081</td>
      <td>0.503161</td>
      <td>0.503298</td>
      <td>0.502989</td>
      <td>0.503330</td>
      <td>0.503174</td>
      <td>0.506289</td>
      <td>0.503113</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.502557</td>
      <td>0.502475</td>
      <td>0.502634</td>
      <td>0.502602</td>
      <td>0.502397</td>
      <td>0.503462</td>
      <td>1.000000</td>
      <td>0.502455</td>
      <td>0.502422</td>
      <td>0.502578</td>
      <td>0.502394</td>
      <td>0.503132</td>
      <td>0.502508</td>
      <td>0.506109</td>
      <td>0.503164</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.502106</td>
      <td>0.502224</td>
      <td>0.502123</td>
      <td>0.502066</td>
      <td>0.501970</td>
      <td>0.503081</td>
      <td>0.502455</td>
      <td>1.000000</td>
      <td>0.501888</td>
      <td>0.502084</td>
      <td>0.501980</td>
      <td>0.502815</td>
      <td>0.502218</td>
      <td>0.505386</td>
      <td>0.502963</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.502038</td>
      <td>0.502194</td>
      <td>0.501775</td>
      <td>0.501877</td>
      <td>0.501781</td>
      <td>0.503161</td>
      <td>0.502422</td>
      <td>0.501888</td>
      <td>1.000000</td>
      <td>0.502147</td>
      <td>0.501771</td>
      <td>0.502771</td>
      <td>0.502178</td>
      <td>0.506386</td>
      <td>0.502804</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.502223</td>
      <td>0.502356</td>
      <td>0.502086</td>
      <td>0.502205</td>
      <td>0.502160</td>
      <td>0.503298</td>
      <td>0.502578</td>
      <td>0.502084</td>
      <td>0.502147</td>
      <td>1.000000</td>
      <td>0.502024</td>
      <td>0.502957</td>
      <td>0.502404</td>
      <td>0.506445</td>
      <td>0.502877</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.501899</td>
      <td>0.502155</td>
      <td>0.501578</td>
      <td>0.501675</td>
      <td>0.501722</td>
      <td>0.502989</td>
      <td>0.502394</td>
      <td>0.501980</td>
      <td>0.501771</td>
      <td>0.502024</td>
      <td>1.000000</td>
      <td>0.502601</td>
      <td>0.502266</td>
      <td>0.505436</td>
      <td>0.502705</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.502997</td>
      <td>0.502824</td>
      <td>0.502868</td>
      <td>0.502947</td>
      <td>0.502705</td>
      <td>0.503330</td>
      <td>0.503132</td>
      <td>0.502815</td>
      <td>0.502771</td>
      <td>0.502957</td>
      <td>0.502601</td>
      <td>1.000000</td>
      <td>0.502775</td>
      <td>0.506218</td>
      <td>0.503145</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.502416</td>
      <td>0.502291</td>
      <td>0.502427</td>
      <td>0.502382</td>
      <td>0.502265</td>
      <td>0.503174</td>
      <td>0.502508</td>
      <td>0.502218</td>
      <td>0.502178</td>
      <td>0.502404</td>
      <td>0.502266</td>
      <td>0.502775</td>
      <td>1.000000</td>
      <td>0.505733</td>
      <td>0.502828</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.506219</td>
      <td>0.505995</td>
      <td>0.505912</td>
      <td>0.506518</td>
      <td>0.505916</td>
      <td>0.506289</td>
      <td>0.506109</td>
      <td>0.505386</td>
      <td>0.506386</td>
      <td>0.506445</td>
      <td>0.505436</td>
      <td>0.506218</td>
      <td>0.505733</td>
      <td>1.000000</td>
      <td>0.505565</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.502989</td>
      <td>0.502840</td>
      <td>0.503068</td>
      <td>0.502875</td>
      <td>0.502900</td>
      <td>0.503113</td>
      <td>0.503164</td>
      <td>0.502963</td>
      <td>0.502804</td>
      <td>0.502877</td>
      <td>0.502705</td>
      <td>0.503145</td>
      <td>0.502828</td>
      <td>0.505565</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### There are 4 methods implemented in `scanpath_complex` for similarity matrix reordering. For example, we can use `dimensionality_reduction_order` for the matrix calculated above. This function applies a dimensionality reduction technique, such as Multi-Dimensional Scaling (MDS), to the input similarity matrix. The goal is to project the items into a lower-dimensional space (typically 1D) where the order of items reflects their dissimilarities as closely as possible. The indices of items are then reordered according to their positions in this lower-dimensional space, resulting in an ordering that preserves the structure of the original similarities.


```python
mds_reordered_matrix = eye_complex.dimensionality_reduction_order(sim_matrix)
pd.DataFrame(mds_reordered_matrix)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.501775</td>
      <td>0.501888</td>
      <td>0.502147</td>
      <td>0.501771</td>
      <td>0.502038</td>
      <td>0.502194</td>
      <td>0.503161</td>
      <td>0.506386</td>
      <td>0.501781</td>
      <td>0.502422</td>
      <td>0.502804</td>
      <td>0.501877</td>
      <td>0.502178</td>
      <td>0.502771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.501775</td>
      <td>1.000000</td>
      <td>0.502123</td>
      <td>0.502086</td>
      <td>0.501578</td>
      <td>0.502068</td>
      <td>0.502357</td>
      <td>0.503369</td>
      <td>0.505912</td>
      <td>0.501828</td>
      <td>0.502634</td>
      <td>0.503068</td>
      <td>0.501698</td>
      <td>0.502427</td>
      <td>0.502868</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.501888</td>
      <td>0.502123</td>
      <td>1.000000</td>
      <td>0.502084</td>
      <td>0.501980</td>
      <td>0.502106</td>
      <td>0.502224</td>
      <td>0.503081</td>
      <td>0.505386</td>
      <td>0.501970</td>
      <td>0.502455</td>
      <td>0.502963</td>
      <td>0.502066</td>
      <td>0.502218</td>
      <td>0.502815</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.502147</td>
      <td>0.502086</td>
      <td>0.502084</td>
      <td>1.000000</td>
      <td>0.502024</td>
      <td>0.502223</td>
      <td>0.502356</td>
      <td>0.503298</td>
      <td>0.506445</td>
      <td>0.502160</td>
      <td>0.502578</td>
      <td>0.502877</td>
      <td>0.502205</td>
      <td>0.502404</td>
      <td>0.502957</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.501771</td>
      <td>0.501578</td>
      <td>0.501980</td>
      <td>0.502024</td>
      <td>1.000000</td>
      <td>0.501899</td>
      <td>0.502155</td>
      <td>0.502989</td>
      <td>0.505436</td>
      <td>0.501722</td>
      <td>0.502394</td>
      <td>0.502705</td>
      <td>0.501675</td>
      <td>0.502266</td>
      <td>0.502601</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.502038</td>
      <td>0.502068</td>
      <td>0.502106</td>
      <td>0.502223</td>
      <td>0.501899</td>
      <td>1.000000</td>
      <td>0.502305</td>
      <td>0.503174</td>
      <td>0.506219</td>
      <td>0.501972</td>
      <td>0.502557</td>
      <td>0.502989</td>
      <td>0.502114</td>
      <td>0.502416</td>
      <td>0.502997</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.502194</td>
      <td>0.502357</td>
      <td>0.502224</td>
      <td>0.502356</td>
      <td>0.502155</td>
      <td>0.502305</td>
      <td>1.000000</td>
      <td>0.503152</td>
      <td>0.505995</td>
      <td>0.502204</td>
      <td>0.502475</td>
      <td>0.502840</td>
      <td>0.502338</td>
      <td>0.502291</td>
      <td>0.502824</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.503161</td>
      <td>0.503369</td>
      <td>0.503081</td>
      <td>0.503298</td>
      <td>0.502989</td>
      <td>0.503174</td>
      <td>0.503152</td>
      <td>1.000000</td>
      <td>0.506289</td>
      <td>0.503020</td>
      <td>0.503462</td>
      <td>0.503113</td>
      <td>0.503227</td>
      <td>0.503174</td>
      <td>0.503330</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.506386</td>
      <td>0.505912</td>
      <td>0.505386</td>
      <td>0.506445</td>
      <td>0.505436</td>
      <td>0.506219</td>
      <td>0.505995</td>
      <td>0.506289</td>
      <td>1.000000</td>
      <td>0.505916</td>
      <td>0.506109</td>
      <td>0.505565</td>
      <td>0.506518</td>
      <td>0.505733</td>
      <td>0.506218</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.501781</td>
      <td>0.501828</td>
      <td>0.501970</td>
      <td>0.502160</td>
      <td>0.501722</td>
      <td>0.501972</td>
      <td>0.502204</td>
      <td>0.503020</td>
      <td>0.505916</td>
      <td>1.000000</td>
      <td>0.502397</td>
      <td>0.502900</td>
      <td>0.501833</td>
      <td>0.502265</td>
      <td>0.502705</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.502422</td>
      <td>0.502634</td>
      <td>0.502455</td>
      <td>0.502578</td>
      <td>0.502394</td>
      <td>0.502557</td>
      <td>0.502475</td>
      <td>0.503462</td>
      <td>0.506109</td>
      <td>0.502397</td>
      <td>1.000000</td>
      <td>0.503164</td>
      <td>0.502602</td>
      <td>0.502508</td>
      <td>0.503132</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.502804</td>
      <td>0.503068</td>
      <td>0.502963</td>
      <td>0.502877</td>
      <td>0.502705</td>
      <td>0.502989</td>
      <td>0.502840</td>
      <td>0.503113</td>
      <td>0.505565</td>
      <td>0.502900</td>
      <td>0.503164</td>
      <td>1.000000</td>
      <td>0.502875</td>
      <td>0.502828</td>
      <td>0.503145</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.501877</td>
      <td>0.501698</td>
      <td>0.502066</td>
      <td>0.502205</td>
      <td>0.501675</td>
      <td>0.502114</td>
      <td>0.502338</td>
      <td>0.503227</td>
      <td>0.506518</td>
      <td>0.501833</td>
      <td>0.502602</td>
      <td>0.502875</td>
      <td>1.000000</td>
      <td>0.502382</td>
      <td>0.502947</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.502178</td>
      <td>0.502427</td>
      <td>0.502218</td>
      <td>0.502404</td>
      <td>0.502266</td>
      <td>0.502416</td>
      <td>0.502291</td>
      <td>0.503174</td>
      <td>0.505733</td>
      <td>0.502265</td>
      <td>0.502508</td>
      <td>0.502828</td>
      <td>0.502382</td>
      <td>1.000000</td>
      <td>0.502775</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.502771</td>
      <td>0.502868</td>
      <td>0.502815</td>
      <td>0.502957</td>
      <td>0.502601</td>
      <td>0.502997</td>
      <td>0.502824</td>
      <td>0.503330</td>
      <td>0.506218</td>
      <td>0.502705</td>
      <td>0.503132</td>
      <td>0.503145</td>
      <td>0.502947</td>
      <td>0.502775</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Scanpath Measures

##### This module offers classes and methods which calculate various measures of scanpaths. 


```python
import eyetracking.features.measures as eye_measures
```

##### Let's calculate some basic measures in order to demonstrate the usecase process.

##### The HurstExponent class estimates the Hurst exponent of a time series using Rescaled Range (R/S) analysis. The Hurst exponent is a measure of the long-term memory of time series data, indicating whether the data is trending, mean-reverting, or behaving in a random walk. Parameter `n_iters` regulates the number of iterations for the R/S analysis, while `fill_strategy` is the strategy to adjust data to the power of 2.


```python
hurst_exponent = eye_measures.HurstExponent(
    var=['norm_pos_x', 'norm_pos_y'],
    n_iters=10,
    fill_strategy='reduce',
    pk=['SUBJ', 'group'],
    return_df=True
)

hurst_exponent.fit_transform(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>he_['norm_pos_x', 'norm_pos_y']</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001590</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001716</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000698</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.001475</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.001473</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.002331</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.001375</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.001384</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.001553</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.001521</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.001682</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.001679</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.001560</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.005400</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.002474</td>
    </tr>
  </tbody>
</table>
</div>



##### We can also calculate the entropy of a 2D spatial distribution. This class firstly divides the space into a grid and then calculates the entropy based on the resulting distribution of points within the grid which measures the randomness or unpredictability of the spatial distribution.


```python
gridded_entropy = eye_measures.GriddedDistributionEntropy(
    x='norm_pos_x',
    y='norm_pos_y',
    pk=['SUBJ', 'group'],
    return_df=True
)

gridded_entropy.fit_transform(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grid_entropy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>3.960535</td>
    </tr>
    <tr>
      <th>2_1</th>
      <td>3.829708</td>
    </tr>
    <tr>
      <th>3_1</th>
      <td>4.002632</td>
    </tr>
    <tr>
      <th>4_1</th>
      <td>3.997430</td>
    </tr>
    <tr>
      <th>5_1</th>
      <td>3.972908</td>
    </tr>
    <tr>
      <th>6_1</th>
      <td>4.002764</td>
    </tr>
    <tr>
      <th>7_1</th>
      <td>4.139917</td>
    </tr>
    <tr>
      <th>8_1</th>
      <td>4.158252</td>
    </tr>
    <tr>
      <th>9_1</th>
      <td>3.799890</td>
    </tr>
    <tr>
      <th>10_1</th>
      <td>4.022404</td>
    </tr>
    <tr>
      <th>11_1</th>
      <td>4.164256</td>
    </tr>
    <tr>
      <th>12_1</th>
      <td>4.023588</td>
    </tr>
    <tr>
      <th>13_1</th>
      <td>3.972001</td>
    </tr>
    <tr>
      <th>14_1</th>
      <td>3.930190</td>
    </tr>
    <tr>
      <th>15_1</th>
      <td>4.148421</td>
    </tr>
  </tbody>
</table>
</div>



##### There are more complex features as well. 


##### One of them is RQA (Recurrence Quantification Analysis) for time-series or spatial data. The metrics calculated include Recurrence (REC), Determinism (DET), Laminarity (LAM), and Center of Recurrence Mass (CORM). These measures help to quantify the complexity and structure of the recurrence patterns within the data. In this example we use a default euclidean metric as `metric`. Parameters `rho` and `min_length` correspond for RQA matrix threshold radius and threshold length of its diagonal. In `measures` we specify the required features to calculate.

#####


```python
rqa_measures = eye_measures.RQAMeasures(
    metric=lambda p, q: np.linalg.norm(p - q),
    rho=0.10,
    min_length=1,
    measures=["rec", "corm"],
    x='norm_pos_x',
    y='norm_pos_y',
    pk=['SUBJ', 'group'],
    return_df=True
)

rqa_measures.fit_transform(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rec</th>
      <th>corm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>11.811201</td>
      <td>33.640767</td>
    </tr>
    <tr>
      <th>2_1</th>
      <td>12.874153</td>
      <td>34.330960</td>
    </tr>
    <tr>
      <th>3_1</th>
      <td>11.805156</td>
      <td>33.379707</td>
    </tr>
    <tr>
      <th>4_1</th>
      <td>12.986185</td>
      <td>33.767070</td>
    </tr>
    <tr>
      <th>5_1</th>
      <td>12.815115</td>
      <td>34.050548</td>
    </tr>
    <tr>
      <th>6_1</th>
      <td>12.493029</td>
      <td>34.067843</td>
    </tr>
    <tr>
      <th>7_1</th>
      <td>13.414963</td>
      <td>33.543729</td>
    </tr>
    <tr>
      <th>8_1</th>
      <td>10.816080</td>
      <td>33.598875</td>
    </tr>
    <tr>
      <th>9_1</th>
      <td>13.173418</td>
      <td>33.557892</td>
    </tr>
    <tr>
      <th>10_1</th>
      <td>13.326421</td>
      <td>33.787375</td>
    </tr>
    <tr>
      <th>11_1</th>
      <td>10.237605</td>
      <td>33.482746</td>
    </tr>
    <tr>
      <th>12_1</th>
      <td>12.285995</td>
      <td>33.848199</td>
    </tr>
    <tr>
      <th>13_1</th>
      <td>11.025638</td>
      <td>33.578878</td>
    </tr>
    <tr>
      <th>14_1</th>
      <td>12.762660</td>
      <td>33.339035</td>
    </tr>
    <tr>
      <th>15_1</th>
      <td>10.291070</td>
      <td>33.559307</td>
    </tr>
  </tbody>
</table>
</div>



##### SaccadeUnlikelihood class calculates the cumulative Negative Log-Likelihood (NLL) of saccades in a scanpath based on a saccade transition model. The model differentiates between progressive and regressive saccades, each characterized by its own asymmetric Gaussian distribution. The default distributions parameters used in this case are derived from Potsdam Sentence Corpus.  


```python
saccade_unl = eye_measures.SaccadeUnlikelihood(
    mu_p=1,
    sigma_p1=0.5,
    sigma_p2=1,
    mu_r=1,
    sigma_r1=0.3,
    sigma_r2=0.7,
    psi=0.9,
    x='norm_pos_x',
    y='norm_pos_y',
    pk=['SUBJ', 'group'],
    return_df=True
)

saccade_unl.fit_transform(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>saccade_nll</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>5855.969515</td>
    </tr>
    <tr>
      <th>2_1</th>
      <td>5162.866221</td>
    </tr>
    <tr>
      <th>3_1</th>
      <td>9246.306051</td>
    </tr>
    <tr>
      <th>4_1</th>
      <td>7201.777450</td>
    </tr>
    <tr>
      <th>5_1</th>
      <td>6378.480145</td>
    </tr>
    <tr>
      <th>6_1</th>
      <td>3425.953969</td>
    </tr>
    <tr>
      <th>7_1</th>
      <td>4950.269975</td>
    </tr>
    <tr>
      <th>8_1</th>
      <td>5591.412520</td>
    </tr>
    <tr>
      <th>9_1</th>
      <td>6492.712042</td>
    </tr>
    <tr>
      <th>10_1</th>
      <td>5770.656154</td>
    </tr>
    <tr>
      <th>11_1</th>
      <td>7163.822825</td>
    </tr>
    <tr>
      <th>12_1</th>
      <td>4257.228198</td>
    </tr>
    <tr>
      <th>13_1</th>
      <td>4480.162030</td>
    </tr>
    <tr>
      <th>14_1</th>
      <td>1950.917958</td>
    </tr>
    <tr>
      <th>15_1</th>
      <td>3704.882425</td>
    </tr>
  </tbody>
</table>
</div>



## Statistical Features

##### Attributes of saccades, fixations, as well as microsaccades and regressions such as max length, mean acceleration, and other are available in `stats` module of `eyetracking.features`. You can calculate statistics using any aggregation function supported by `pandas`.


```python
import eyetracking.features.stats as eye_stats
```

##### Prepare desired statistics about saccades:


```python
sac_feats_stats = {
    'length': ['min', 'max'],
    'speed': ['mean', 'kurtosis'],
    'acceleration': ['mean']
}
```

##### Also, one would like to see the similarity of object and its group. We have people ('SUBJ') which are divided into groups ('group', here we have a single group, but there could be many groups, for example, age-based). Thus, we can calculate shift features, which are a difference of object's feature value and its group's mean value.

##### Prepare saccade statistics which we want to calculate shift features for:


```python
sac_feats_stats_shift = {'length': ['max'],
                         'acceleration': ['mean']}
```

##### Define transformer for saccades:


```python
sf = eye_stats.SaccadeFeatures(x='norm_pos_x',
                               y='norm_pos_y',
                               t='timestamp',
                               pk=['SUBJ', 'group'],
                               features_stats=sac_feats_stats,
                               shift_features=sac_feats_stats_shift,
                               shift_pk=['group'])
```


```python
sf.fit_transform(data)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sac_length_min</th>
      <th>sac_length_max</th>
      <th>sac_length_max_shift</th>
      <th>sac_acceleration_mean</th>
      <th>sac_acceleration_mean_shift</th>
      <th>sac_speed_mean</th>
      <th>sac_speed_kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>0.006033</td>
      <td>0.735187</td>
      <td>0.000000</td>
      <td>3.200016</td>
      <td>-0.412047</td>
      <td>1.036293</td>
      <td>4.289506</td>
    </tr>
    <tr>
      <th>2_1</th>
      <td>0.004946</td>
      <td>0.695758</td>
      <td>-0.039428</td>
      <td>4.411436</td>
      <td>0.799373</td>
      <td>1.188990</td>
      <td>3.949788</td>
    </tr>
    <tr>
      <th>3_1</th>
      <td>0.002494</td>
      <td>0.691093</td>
      <td>-0.044094</td>
      <td>3.351796</td>
      <td>-0.260267</td>
      <td>1.073377</td>
      <td>2.275953</td>
    </tr>
    <tr>
      <th>4_1</th>
      <td>0.004396</td>
      <td>0.637963</td>
      <td>-0.097223</td>
      <td>3.246972</td>
      <td>-0.365090</td>
      <td>0.999678</td>
      <td>4.342265</td>
    </tr>
    <tr>
      <th>5_1</th>
      <td>0.004814</td>
      <td>0.693513</td>
      <td>-0.041673</td>
      <td>3.874696</td>
      <td>0.262634</td>
      <td>1.111982</td>
      <td>4.788790</td>
    </tr>
    <tr>
      <th>6_1</th>
      <td>0.014949</td>
      <td>0.659511</td>
      <td>-0.075676</td>
      <td>4.518683</td>
      <td>0.906621</td>
      <td>1.423984</td>
      <td>2.408947</td>
    </tr>
    <tr>
      <th>7_1</th>
      <td>0.003880</td>
      <td>0.579964</td>
      <td>-0.155222</td>
      <td>4.047229</td>
      <td>0.435167</td>
      <td>1.190862</td>
      <td>2.696679</td>
    </tr>
    <tr>
      <th>8_1</th>
      <td>0.012126</td>
      <td>0.661617</td>
      <td>-0.073569</td>
      <td>4.614863</td>
      <td>1.002801</td>
      <td>1.307412</td>
      <td>1.579665</td>
    </tr>
    <tr>
      <th>9_1</th>
      <td>0.006906</td>
      <td>0.656098</td>
      <td>-0.079089</td>
      <td>2.537365</td>
      <td>-1.074698</td>
      <td>0.872490</td>
      <td>6.404846</td>
    </tr>
    <tr>
      <th>10_1</th>
      <td>0.002115</td>
      <td>0.656778</td>
      <td>-0.078408</td>
      <td>1.736126</td>
      <td>-1.875936</td>
      <td>0.707411</td>
      <td>6.400686</td>
    </tr>
    <tr>
      <th>11_1</th>
      <td>0.008301</td>
      <td>0.673268</td>
      <td>-0.061919</td>
      <td>3.719054</td>
      <td>0.106992</td>
      <td>1.186794</td>
      <td>2.141173</td>
    </tr>
    <tr>
      <th>12_1</th>
      <td>0.004982</td>
      <td>0.648675</td>
      <td>-0.086511</td>
      <td>3.103430</td>
      <td>-0.508633</td>
      <td>1.051155</td>
      <td>6.032897</td>
    </tr>
    <tr>
      <th>13_1</th>
      <td>0.014569</td>
      <td>0.673546</td>
      <td>-0.061641</td>
      <td>5.627082</td>
      <td>2.015020</td>
      <td>1.587480</td>
      <td>1.892242</td>
    </tr>
    <tr>
      <th>14_1</th>
      <td>0.013864</td>
      <td>0.670388</td>
      <td>-0.064799</td>
      <td>2.511618</td>
      <td>-1.100445</td>
      <td>0.942285</td>
      <td>5.053431</td>
    </tr>
    <tr>
      <th>15_1</th>
      <td>0.017111</td>
      <td>0.730887</td>
      <td>-0.004300</td>
      <td>3.800426</td>
      <td>0.188364</td>
      <td>1.232593</td>
      <td>3.360688</td>
    </tr>
  </tbody>
</table>
</div>



## Extractor Class

##### Finally, we can combine several extractor classes into one `Extractor` class to calculate all the features at once.


```python
from eyetracking.features.extractor import Extractor

extractor = Extractor(
    features=[
        eye_dist.SimpleDistances(
            methods=["euc", "eye", "man"],
            expected_paths_method="fwp",
        ),
        eye_measures.GriddedDistributionEntropy(),
        eye_measures.SaccadeUnlikelihood(),
        eye_stats.SaccadeFeatures(
            features_stats=sac_feats_stats,
            shift_features=sac_feats_stats_shift,
            shift_pk=['group']
        )
    ],
    x='norm_pos_x',
    y='norm_pos_y',
    t='timestamp',
    duration='duration',
    dispersion='dispersion',
    path_pk=['group'],
    pk=['SUBJ', 'group'],
    return_df=True
)

extractor.fit_transform(data)
```

    100%|██████████| 15/15 [00:00<00:00, 2763.90it/s]
    100%|██████████| 15/15 [00:02<00:00,  5.15it/s]
    100%|██████████| 15/15 [00:02<00:00,  5.09it/s]





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>euc_dist</th>
      <th>eye_dist</th>
      <th>man_dist</th>
      <th>grid_entropy</th>
      <th>saccade_nll</th>
      <th>sac_length_min</th>
      <th>sac_length_max</th>
      <th>sac_length_max_shift</th>
      <th>sac_acceleration_mean</th>
      <th>sac_acceleration_mean_shift</th>
      <th>sac_speed_mean</th>
      <th>sac_speed_kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>200.623121</td>
      <td>0.058312</td>
      <td>0.015358</td>
      <td>3.960535</td>
      <td>5855.969515</td>
      <td>0.006033</td>
      <td>0.735187</td>
      <td>0.000000</td>
      <td>1.736375e+16</td>
      <td>-4.088451e+16</td>
      <td>1.937620e+16</td>
      <td>1409.816333</td>
    </tr>
    <tr>
      <th>2_1</th>
      <td>122.518147</td>
      <td>0.064941</td>
      <td>0.017263</td>
      <td>3.829708</td>
      <td>5162.866221</td>
      <td>0.004946</td>
      <td>0.695758</td>
      <td>-0.039428</td>
      <td>5.750563e+16</td>
      <td>-7.426283e+14</td>
      <td>8.341979e+16</td>
      <td>623.452215</td>
    </tr>
    <tr>
      <th>3_1</th>
      <td>1162.275204</td>
      <td>0.071679</td>
      <td>0.017920</td>
      <td>4.002632</td>
      <td>9246.306051</td>
      <td>0.002494</td>
      <td>0.691093</td>
      <td>-0.044094</td>
      <td>4.692676e+16</td>
      <td>-1.132150e+16</td>
      <td>7.542848e+16</td>
      <td>679.472530</td>
    </tr>
    <tr>
      <th>4_1</th>
      <td>412.366120</td>
      <td>0.058348</td>
      <td>0.015121</td>
      <td>3.997430</td>
      <td>7201.777450</td>
      <td>0.004396</td>
      <td>0.637963</td>
      <td>-0.097223</td>
      <td>8.420396e+16</td>
      <td>2.595570e+16</td>
      <td>1.207275e+17</td>
      <td>375.534401</td>
    </tr>
    <tr>
      <th>5_1</th>
      <td>261.495549</td>
      <td>0.066644</td>
      <td>0.017448</td>
      <td>3.972908</td>
      <td>6378.480145</td>
      <td>0.004814</td>
      <td>0.693513</td>
      <td>-0.041673</td>
      <td>3.597491e+16</td>
      <td>-2.227334e+16</td>
      <td>6.661781e+16</td>
      <td>1494.364756</td>
    </tr>
    <tr>
      <th>6_1</th>
      <td>72.612316</td>
      <td>0.051765</td>
      <td>0.014200</td>
      <td>4.002764</td>
      <td>3425.953969</td>
      <td>0.014949</td>
      <td>0.659511</td>
      <td>-0.075676</td>
      <td>6.432294e+16</td>
      <td>6.074690e+15</td>
      <td>7.196951e+16</td>
      <td>567.907161</td>
    </tr>
    <tr>
      <th>7_1</th>
      <td>112.304684</td>
      <td>0.067186</td>
      <td>0.017750</td>
      <td>4.139917</td>
      <td>4950.269975</td>
      <td>0.003880</td>
      <td>0.579964</td>
      <td>-0.155222</td>
      <td>8.772772e+16</td>
      <td>2.947947e+16</td>
      <td>1.261946e+17</td>
      <td>392.914133</td>
    </tr>
    <tr>
      <th>8_1</th>
      <td>198.174110</td>
      <td>0.068108</td>
      <td>0.018178</td>
      <td>4.158252</td>
      <td>5591.412520</td>
      <td>0.012126</td>
      <td>0.661617</td>
      <td>-0.073569</td>
      <td>8.430990e+16</td>
      <td>2.606164e+16</td>
      <td>1.068958e+17</td>
      <td>376.451656</td>
    </tr>
    <tr>
      <th>9_1</th>
      <td>255.661709</td>
      <td>0.061571</td>
      <td>0.016137</td>
      <td>3.799890</td>
      <td>6492.712042</td>
      <td>0.006906</td>
      <td>0.656098</td>
      <td>-0.079089</td>
      <td>9.964382e+15</td>
      <td>-4.828387e+16</td>
      <td>1.150047e+16</td>
      <td>1618.912271</td>
    </tr>
    <tr>
      <th>10_1</th>
      <td>143.557332</td>
      <td>0.050267</td>
      <td>0.013282</td>
      <td>4.022404</td>
      <td>5770.656154</td>
      <td>0.002115</td>
      <td>0.656778</td>
      <td>-0.078408</td>
      <td>3.178135e+16</td>
      <td>-2.646690e+16</td>
      <td>4.584361e+16</td>
      <td>1447.565585</td>
    </tr>
    <tr>
      <th>11_1</th>
      <td>525.852980</td>
      <td>0.063494</td>
      <td>0.016458</td>
      <td>4.164256</td>
      <td>7163.822825</td>
      <td>0.008301</td>
      <td>0.673268</td>
      <td>-0.061919</td>
      <td>4.828441e+16</td>
      <td>-9.963844e+15</td>
      <td>7.518146e+16</td>
      <td>723.486787</td>
    </tr>
    <tr>
      <th>12_1</th>
      <td>82.064542</td>
      <td>0.061822</td>
      <td>0.016428</td>
      <td>4.023588</td>
      <td>4257.228198</td>
      <td>0.004982</td>
      <td>0.648675</td>
      <td>-0.086511</td>
      <td>4.361359e+16</td>
      <td>-1.463466e+16</td>
      <td>6.764585e+16</td>
      <td>768.066955</td>
    </tr>
    <tr>
      <th>13_1</th>
      <td>120.106331</td>
      <td>0.051667</td>
      <td>0.014180</td>
      <td>3.972001</td>
      <td>4480.162030</td>
      <td>0.014569</td>
      <td>0.673546</td>
      <td>-0.061641</td>
      <td>1.857274e+17</td>
      <td>1.274792e+17</td>
      <td>2.804642e+17</td>
      <td>177.747624</td>
    </tr>
    <tr>
      <th>14_1</th>
      <td>39.239737</td>
      <td>0.045353</td>
      <td>0.012841</td>
      <td>3.930190</td>
      <td>1950.917958</td>
      <td>0.013864</td>
      <td>0.670388</td>
      <td>-0.064799</td>
      <td>8.424760e+15</td>
      <td>-4.982349e+16</td>
      <td>1.684952e+16</td>
      <td>535.324400</td>
    </tr>
    <tr>
      <th>15_1</th>
      <td>86.587902</td>
      <td>0.046960</td>
      <td>0.013304</td>
      <td>4.148421</td>
      <td>3704.882425</td>
      <td>0.017111</td>
      <td>0.730887</td>
      <td>-0.004300</td>
      <td>7.318467e+16</td>
      <td>1.493641e+16</td>
      <td>1.034776e+17</td>
      <td>554.657857</td>
    </tr>
  </tbody>
</table>
</div>



##### Extractor class can be easily integrated into the `sklearn Pipeline` as it uses the sklearn API fully.

##### In this example, the pipeline calls the extractor to calculate the desired features first and then passes them to the model. Note that the extractor can save additional features from the input `DataFrame` before passing them to the model, if needed.


```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pipeline = Pipeline([
    ('extractor', extractor),
    ('classifier', LogisticRegression())
])
```


```python
target = data.drop_duplicates(subset=['SUBJ'])['ANSWER'].reset_index(drop=True)

predictions = pipeline.fit(data, target).predict(data)
```

    100%|██████████| 15/15 [00:00<00:00, 2965.01it/s]
    100%|██████████| 15/15 [00:03<00:00,  4.99it/s]
    100%|██████████| 15/15 [00:02<00:00,  5.16it/s]
    100%|██████████| 15/15 [00:00<00:00, 2789.14it/s]
    100%|██████████| 15/15 [00:02<00:00,  5.08it/s]
    100%|██████████| 15/15 [00:03<00:00,  4.98it/s]



```python
accuracy_score(target, predictions)
```




    1.0


