```python
import pandas as pd
import numpy as np

from os.path import join
from stats import SaccadeFeatures, FixationFeatures, RegressionFeatures, MicroSaccades
from measures import HurstExponent, ShannonEntropy
from extractor import Extractor

DATA_PATH = join('..', 'test_data')
```


```python
data = pd.concat([pd.read_excel(join(DATA_PATH, 'itog_fix_1.xlsx')),
                  pd.read_excel(join(DATA_PATH, 'itog_fix_2.xlsx'))],
                 axis=0)
```


```python
data['AOI'] = 'aoi_1'

for i in data.index[::5]:
    data.loc[i, 'AOI'] = 'aoi_2'
for i in data.index[::6]:
    data.loc[i, 'AOI'] = 'aoi_2'
for i in data.index[::7]:
    data.loc[i, 'AOI'] = 'aoi_2'
```


```python
data['AOI'].value_counts()
```




    AOI
    aoi_2    5304
    aoi_1    3233
    Name: count, dtype: int64




```python
x = 'norm_pos_x'
y = 'norm_pos_y'
t = 'start_timestamp'
dur = 'duration'
dis = 'dispersion'


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


fix_feats_stats = {'duration': ['sum'], 'vad': ['mean']}

fx = FixationFeatures(duration=dur, dispersion=dis,
                     pk=['Participant', 'tekst'],
                     aoi='AOI',
                     features_stats=fix_feats_stats,
                     shift_pk=['Participant'])

rg = RegressionFeatures(x=x, y=y, t=t, duration=dur,
                        pk=['Participant', 'tekst'],
                        aoi='AOI',
                        features_stats=sac_feats_stats,
                        shift_features=sac_feats_stats_shift,
                        shift_pk=['tekst'],
                        rule=(1, 3))

ms = MicroSaccades(x=x, y=y, t=t, dispersion=dis,
                   pk=['Participant', 'tekst'],
                   aoi='AOI',
                   features_stats=sac_feats_stats,
                   shift_features=sac_feats_stats_shift,
                   shift_pk=['tekst'],
                   min_dispersion=0.01,
                   max_speed=0.2)

features = [
    SaccadeFeatures(features_stats=sac_feats_stats,
                    shift_features=sac_feats_stats_shift,
                    shift_pk=['tekst']),
    FixationFeatures(features_stats=fix_feats_stats,
                     shift_pk=['Participant']),
    RegressionFeatures(features_stats=sac_feats_stats,
                       shift_features=sac_feats_stats_shift,
                       shift_pk=['tekst'],
                       rule=(1, 3)),
    MicroSaccades(features_stats=sac_feats_stats,
                  shift_features=sac_feats_stats_shift,
                  shift_pk=['Participant'],
                  min_dispersion=0.001,
                  max_speed=0.5)
]

extractor_ = Extractor(features=features, x=x, y=y, t=t, duration=dur, dispersion=dis,
                      aoi='AOI',
                      pk=['Participant', 'tekst'],
                      return_df=True)
```


```python
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
                       rule=(1, 3)),
    HurstExponent(), ShannonEntropy()
]

extractor = Extractor(features=features, x=x, y=y, t=t, duration=dur, dispersion=dis,
                      aoi='AOI',
                      pk=['Participant', 'tekst'],
                      return_df=True)
```


```python
sf.fit_transform(data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>0.002657</td>
      <td>0.433486</td>
      <td>-0.008934</td>
      <td>1.522857</td>
      <td>0.316149</td>
      <td>0.488956</td>
      <td>5.498018</td>
    </tr>
    <tr>
      <th>1_2</th>
      <td>0.003757</td>
      <td>0.321837</td>
      <td>-0.127119</td>
      <td>1.507806</td>
      <td>0.322016</td>
      <td>0.451773</td>
      <td>8.356764</td>
    </tr>
    <tr>
      <th>1_3</th>
      <td>0.003663</td>
      <td>0.365776</td>
      <td>-0.207636</td>
      <td>1.292694</td>
      <td>0.083609</td>
      <td>0.390753</td>
      <td>7.639049</td>
    </tr>
    <tr>
      <th>1_4</th>
      <td>0.000212</td>
      <td>0.342315</td>
      <td>-0.362239</td>
      <td>0.999054</td>
      <td>-0.038003</td>
      <td>0.333495</td>
      <td>9.593384</td>
    </tr>
    <tr>
      <th>1_5</th>
      <td>0.002705</td>
      <td>0.375434</td>
      <td>-0.094030</td>
      <td>1.187418</td>
      <td>0.089377</td>
      <td>0.360223</td>
      <td>9.717641</td>
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
    </tr>
    <tr>
      <th>2_33</th>
      <td>0.001920</td>
      <td>0.406550</td>
      <td>0.000000</td>
      <td>0.945929</td>
      <td>-0.050745</td>
      <td>0.264515</td>
      <td>10.235914</td>
    </tr>
    <tr>
      <th>2_34</th>
      <td>0.007928</td>
      <td>0.354386</td>
      <td>-0.260736</td>
      <td>0.883503</td>
      <td>-0.158937</td>
      <td>0.298396</td>
      <td>9.732734</td>
    </tr>
    <tr>
      <th>2_35</th>
      <td>0.002170</td>
      <td>1.269198</td>
      <td>0.000000</td>
      <td>2.527952</td>
      <td>0.883969</td>
      <td>0.623824</td>
      <td>45.476927</td>
    </tr>
    <tr>
      <th>2_36</th>
      <td>0.002196</td>
      <td>1.124204</td>
      <td>0.000000</td>
      <td>1.650884</td>
      <td>0.266994</td>
      <td>0.443600</td>
      <td>10.414252</td>
    </tr>
    <tr>
      <th>2_37</th>
      <td>0.002151</td>
      <td>0.772169</td>
      <td>0.000000</td>
      <td>1.606125</td>
      <td>0.193406</td>
      <td>0.457776</td>
      <td>25.881450</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 7 columns</p>
</div>




```python
fx.fit_transform(data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fix_duration_aoi_2_sum</th>
      <th>fix_vad_aoi_2_mean</th>
      <th>fix_duration_aoi_1_sum</th>
      <th>fix_vad_aoi_1_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>10760.106500</td>
      <td>1.084216</td>
      <td>2504.5290</td>
      <td>1.173406</td>
    </tr>
    <tr>
      <th>1_2</th>
      <td>9308.563500</td>
      <td>1.099930</td>
      <td>1652.0625</td>
      <td>1.427237</td>
    </tr>
    <tr>
      <th>1_3</th>
      <td>9218.071000</td>
      <td>1.219697</td>
      <td>1645.0175</td>
      <td>1.014305</td>
    </tr>
    <tr>
      <th>1_4</th>
      <td>7324.355999</td>
      <td>1.125739</td>
      <td>952.0735</td>
      <td>1.285798</td>
    </tr>
    <tr>
      <th>1_5</th>
      <td>9060.284000</td>
      <td>1.148466</td>
      <td>1942.1885</td>
      <td>1.344456</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2_33</th>
      <td>13896.132000</td>
      <td>1.376585</td>
      <td>2428.7540</td>
      <td>1.332026</td>
    </tr>
    <tr>
      <th>2_34</th>
      <td>5525.225500</td>
      <td>1.291868</td>
      <td>891.6540</td>
      <td>1.522308</td>
    </tr>
    <tr>
      <th>2_35</th>
      <td>9431.293500</td>
      <td>1.414826</td>
      <td>1157.4960</td>
      <td>1.392368</td>
    </tr>
    <tr>
      <th>2_36</th>
      <td>5885.597000</td>
      <td>1.333476</td>
      <td>614.9980</td>
      <td>1.361147</td>
    </tr>
    <tr>
      <th>2_37</th>
      <td>7887.573500</td>
      <td>1.454140</td>
      <td>1260.4125</td>
      <td>1.366280</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 4 columns</p>
</div>




```python
rg.fit_transform(data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reg_length_aoi_2_min</th>
      <th>reg_length_aoi_2_max</th>
      <th>reg_acceleration_aoi_2_mean</th>
      <th>reg_speed_aoi_2_mean</th>
      <th>reg_speed_aoi_2_kurtosis</th>
      <th>reg_length_aoi_1_min</th>
      <th>reg_length_aoi_1_max</th>
      <th>reg_acceleration_aoi_1_mean</th>
      <th>reg_speed_aoi_1_mean</th>
      <th>reg_speed_aoi_1_kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>0.011841</td>
      <td>0.420153</td>
      <td>101.520275</td>
      <td>3.480621</td>
      <td>4.196115</td>
      <td>0.037026</td>
      <td>0.272665</td>
      <td>54.131503</td>
      <td>3.396826</td>
      <td>-1.593181</td>
    </tr>
    <tr>
      <th>1_2</th>
      <td>0.006408</td>
      <td>0.321837</td>
      <td>70.596140</td>
      <td>3.276942</td>
      <td>-0.922803</td>
      <td>0.030813</td>
      <td>0.261686</td>
      <td>35.216219</td>
      <td>2.167320</td>
      <td>-0.934722</td>
    </tr>
    <tr>
      <th>1_3</th>
      <td>0.005249</td>
      <td>0.265067</td>
      <td>73.916643</td>
      <td>2.629076</td>
      <td>1.171818</td>
      <td>0.016539</td>
      <td>0.254050</td>
      <td>89.722992</td>
      <td>3.206941</td>
      <td>3.553356</td>
    </tr>
    <tr>
      <th>1_4</th>
      <td>0.009554</td>
      <td>0.277172</td>
      <td>107.220463</td>
      <td>3.406318</td>
      <td>-0.759758</td>
      <td>0.041129</td>
      <td>0.049056</td>
      <td>57.301368</td>
      <td>2.288800</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1_5</th>
      <td>0.002705</td>
      <td>0.300800</td>
      <td>76.290851</td>
      <td>2.548291</td>
      <td>1.412943</td>
      <td>0.042946</td>
      <td>0.104575</td>
      <td>57.119943</td>
      <td>2.593522</td>
      <td>0.004752</td>
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
    </tr>
    <tr>
      <th>2_33</th>
      <td>0.007210</td>
      <td>0.202824</td>
      <td>97.018484</td>
      <td>2.316939</td>
      <td>1.628924</td>
      <td>0.025199</td>
      <td>0.221629</td>
      <td>148.173164</td>
      <td>3.673611</td>
      <td>-0.042142</td>
    </tr>
    <tr>
      <th>2_34</th>
      <td>0.013271</td>
      <td>0.303437</td>
      <td>161.015426</td>
      <td>3.315178</td>
      <td>0.574263</td>
      <td>0.023785</td>
      <td>0.114924</td>
      <td>132.171743</td>
      <td>3.094738</td>
      <td>-1.646757</td>
    </tr>
    <tr>
      <th>2_35</th>
      <td>0.007504</td>
      <td>1.269198</td>
      <td>186.293178</td>
      <td>5.238658</td>
      <td>6.259332</td>
      <td>0.003259</td>
      <td>0.559806</td>
      <td>54.424245</td>
      <td>2.653471</td>
      <td>-1.497979</td>
    </tr>
    <tr>
      <th>2_36</th>
      <td>0.009415</td>
      <td>0.504216</td>
      <td>132.919882</td>
      <td>3.304351</td>
      <td>9.638348</td>
      <td>0.017941</td>
      <td>1.124204</td>
      <td>142.120547</td>
      <td>6.391043</td>
      <td>6.352378</td>
    </tr>
    <tr>
      <th>2_37</th>
      <td>0.004536</td>
      <td>0.772169</td>
      <td>92.990986</td>
      <td>3.433222</td>
      <td>4.143707</td>
      <td>0.039056</td>
      <td>0.159758</td>
      <td>56.846989</td>
      <td>2.832624</td>
      <td>2.093700</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 10 columns</p>
</div>




```python
ms.fit_transform(data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>microsac_length_aoi_2_min</th>
      <th>microsac_length_aoi_2_max</th>
      <th>microsac_acceleration_aoi_2_mean</th>
      <th>microsac_speed_aoi_2_mean</th>
      <th>microsac_speed_aoi_2_kurtosis</th>
      <th>microsac_length_aoi_1_min</th>
      <th>microsac_length_aoi_1_max</th>
      <th>microsac_acceleration_aoi_1_mean</th>
      <th>microsac_speed_aoi_1_mean</th>
      <th>microsac_speed_aoi_1_kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>0.011581</td>
      <td>0.190765</td>
      <td>1.167471</td>
      <td>0.345077</td>
      <td>1.952940</td>
      <td>0.016593</td>
      <td>0.198150</td>
      <td>0.660408</td>
      <td>0.287073</td>
      <td>-0.298782</td>
    </tr>
    <tr>
      <th>1_2</th>
      <td>0.006408</td>
      <td>0.176232</td>
      <td>1.003533</td>
      <td>0.318633</td>
      <td>1.040025</td>
      <td>0.014979</td>
      <td>0.117818</td>
      <td>1.040075</td>
      <td>0.310151</td>
      <td>0.457796</td>
    </tr>
    <tr>
      <th>1_3</th>
      <td>0.005065</td>
      <td>0.185019</td>
      <td>0.762113</td>
      <td>0.245316</td>
      <td>14.807231</td>
      <td>0.003663</td>
      <td>0.063193</td>
      <td>0.744146</td>
      <td>0.194864</td>
      <td>1.458462</td>
    </tr>
    <tr>
      <th>1_4</th>
      <td>0.001980</td>
      <td>0.110651</td>
      <td>0.813152</td>
      <td>0.241609</td>
      <td>7.668473</td>
      <td>0.041129</td>
      <td>0.070823</td>
      <td>1.077196</td>
      <td>0.313787</td>
      <td>-1.606570</td>
    </tr>
    <tr>
      <th>1_5</th>
      <td>0.002705</td>
      <td>0.192296</td>
      <td>0.942590</td>
      <td>0.301805</td>
      <td>3.149362</td>
      <td>0.020778</td>
      <td>0.104575</td>
      <td>1.326198</td>
      <td>0.359371</td>
      <td>-0.409101</td>
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
    </tr>
    <tr>
      <th>2_33</th>
      <td>0.003962</td>
      <td>0.181661</td>
      <td>1.088828</td>
      <td>0.270166</td>
      <td>6.836550</td>
      <td>0.016619</td>
      <td>0.129722</td>
      <td>0.902597</td>
      <td>0.220889</td>
      <td>-1.668084</td>
    </tr>
    <tr>
      <th>2_34</th>
      <td>0.011230</td>
      <td>0.063878</td>
      <td>0.750266</td>
      <td>0.186453</td>
      <td>3.302904</td>
      <td>0.023785</td>
      <td>0.114924</td>
      <td>0.296183</td>
      <td>0.161013</td>
      <td>1.929328</td>
    </tr>
    <tr>
      <th>2_35</th>
      <td>0.007504</td>
      <td>0.166901</td>
      <td>1.533474</td>
      <td>0.327634</td>
      <td>5.700887</td>
      <td>0.003259</td>
      <td>0.180431</td>
      <td>1.281691</td>
      <td>0.399833</td>
      <td>-1.697831</td>
    </tr>
    <tr>
      <th>2_36</th>
      <td>0.004319</td>
      <td>0.154740</td>
      <td>1.109342</td>
      <td>0.287022</td>
      <td>8.486159</td>
      <td>0.017941</td>
      <td>0.166901</td>
      <td>0.951186</td>
      <td>0.355342</td>
      <td>-0.564187</td>
    </tr>
    <tr>
      <th>2_37</th>
      <td>0.002803</td>
      <td>0.161252</td>
      <td>1.013128</td>
      <td>0.274852</td>
      <td>7.224261</td>
      <td>0.002151</td>
      <td>0.159758</td>
      <td>1.252403</td>
      <td>0.354663</td>
      <td>7.014556</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 10 columns</p>
</div>




```python
extractor.fit_transform(data).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sac_length_aoi_2_min</th>
      <th>sac_length_aoi_2_max</th>
      <th>sac_acceleration_aoi_2_mean</th>
      <th>sac_speed_aoi_2_mean</th>
      <th>sac_speed_aoi_2_kurtosis</th>
      <th>sac_length_aoi_1_min</th>
      <th>sac_length_aoi_1_max</th>
      <th>sac_acceleration_aoi_1_mean</th>
      <th>sac_speed_aoi_1_mean</th>
      <th>sac_speed_aoi_1_kurtosis</th>
      <th>...</th>
      <th>microsac_length_aoi_2_min</th>
      <th>microsac_length_aoi_2_max</th>
      <th>microsac_acceleration_aoi_2_mean</th>
      <th>microsac_speed_aoi_2_mean</th>
      <th>microsac_speed_aoi_2_kurtosis</th>
      <th>microsac_length_aoi_1_min</th>
      <th>microsac_length_aoi_1_max</th>
      <th>microsac_acceleration_aoi_1_mean</th>
      <th>microsac_speed_aoi_1_mean</th>
      <th>microsac_speed_aoi_1_kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>0.011581</td>
      <td>0.420153</td>
      <td>88.408114</td>
      <td>3.316192</td>
      <td>2.582125</td>
      <td>0.016593</td>
      <td>0.272665</td>
      <td>68.252408</td>
      <td>3.213086</td>
      <td>-0.278472</td>
      <td>...</td>
      <td>0.011581</td>
      <td>0.420153</td>
      <td>88.408114</td>
      <td>3.316192</td>
      <td>2.582125</td>
      <td>0.016593</td>
      <td>0.272665</td>
      <td>68.252408</td>
      <td>3.213086</td>
      <td>-0.278472</td>
    </tr>
    <tr>
      <th>1_2</th>
      <td>0.006408</td>
      <td>0.321837</td>
      <td>85.793710</td>
      <td>3.007439</td>
      <td>-0.464949</td>
      <td>0.014979</td>
      <td>0.261686</td>
      <td>54.794634</td>
      <td>2.103520</td>
      <td>-0.316122</td>
      <td>...</td>
      <td>0.006408</td>
      <td>0.321837</td>
      <td>85.793710</td>
      <td>3.007439</td>
      <td>-0.464949</td>
      <td>0.014979</td>
      <td>0.261686</td>
      <td>54.794634</td>
      <td>2.103520</td>
      <td>-0.316122</td>
    </tr>
    <tr>
      <th>1_3</th>
      <td>0.005065</td>
      <td>0.365776</td>
      <td>94.441713</td>
      <td>2.959531</td>
      <td>6.020307</td>
      <td>0.003663</td>
      <td>0.254050</td>
      <td>76.291863</td>
      <td>2.827255</td>
      <td>0.763153</td>
      <td>...</td>
      <td>0.005065</td>
      <td>0.365776</td>
      <td>94.441713</td>
      <td>2.959531</td>
      <td>6.020307</td>
      <td>0.003663</td>
      <td>0.254050</td>
      <td>76.291863</td>
      <td>2.827255</td>
      <td>0.763153</td>
    </tr>
    <tr>
      <th>1_4</th>
      <td>0.001980</td>
      <td>0.277172</td>
      <td>121.601653</td>
      <td>3.005369</td>
      <td>0.249804</td>
      <td>0.041129</td>
      <td>0.272092</td>
      <td>53.823491</td>
      <td>2.801384</td>
      <td>2.951833</td>
      <td>...</td>
      <td>0.001980</td>
      <td>0.277172</td>
      <td>121.601653</td>
      <td>3.005369</td>
      <td>0.249804</td>
      <td>0.041129</td>
      <td>0.272092</td>
      <td>53.823491</td>
      <td>2.801384</td>
      <td>2.951833</td>
    </tr>
    <tr>
      <th>1_5</th>
      <td>0.002705</td>
      <td>0.300800</td>
      <td>69.126721</td>
      <td>2.426858</td>
      <td>1.677840</td>
      <td>0.020778</td>
      <td>0.104575</td>
      <td>68.725060</td>
      <td>2.409873</td>
      <td>0.137052</td>
      <td>...</td>
      <td>0.002705</td>
      <td>0.300800</td>
      <td>69.126721</td>
      <td>2.426858</td>
      <td>1.677840</td>
      <td>0.020778</td>
      <td>0.104575</td>
      <td>68.725060</td>
      <td>2.409873</td>
      <td>0.137052</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
rg_angles = RegressionFeatures(x=x, y=y, t=t, duration=dur,
                        pk=['Participant', 'tekst'],
                        aoi='AOI',
                        features_stats=sac_feats_stats,
                        shift_features=sac_feats_stats_shift,
                        shift_pk=['tekst'],
                        rule=(90, 180),
                        deviation=15)
```


```python
rg_angles.fit_transform(data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reg_length_aoi_2_min</th>
      <th>reg_length_aoi_2_max</th>
      <th>reg_acceleration_aoi_2_mean</th>
      <th>reg_speed_aoi_2_mean</th>
      <th>reg_speed_aoi_2_kurtosis</th>
      <th>reg_length_aoi_1_min</th>
      <th>reg_length_aoi_1_max</th>
      <th>reg_acceleration_aoi_1_mean</th>
      <th>reg_speed_aoi_1_mean</th>
      <th>reg_speed_aoi_1_kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1_1</th>
      <td>0.011581</td>
      <td>0.420153</td>
      <td>88.408114</td>
      <td>3.316192</td>
      <td>2.582125</td>
      <td>0.016593</td>
      <td>0.272665</td>
      <td>68.252408</td>
      <td>3.213086</td>
      <td>-0.278472</td>
    </tr>
    <tr>
      <th>1_2</th>
      <td>0.006408</td>
      <td>0.321837</td>
      <td>85.793710</td>
      <td>3.007439</td>
      <td>-0.464949</td>
      <td>0.014979</td>
      <td>0.261686</td>
      <td>54.794634</td>
      <td>2.103520</td>
      <td>-0.316122</td>
    </tr>
    <tr>
      <th>1_3</th>
      <td>0.005065</td>
      <td>0.365776</td>
      <td>94.441713</td>
      <td>2.959531</td>
      <td>6.020307</td>
      <td>0.003663</td>
      <td>0.254050</td>
      <td>76.291863</td>
      <td>2.827255</td>
      <td>0.763153</td>
    </tr>
    <tr>
      <th>1_4</th>
      <td>0.001980</td>
      <td>0.277172</td>
      <td>121.601653</td>
      <td>3.005369</td>
      <td>0.249804</td>
      <td>0.041129</td>
      <td>0.272092</td>
      <td>53.823491</td>
      <td>2.801384</td>
      <td>2.951833</td>
    </tr>
    <tr>
      <th>1_5</th>
      <td>0.002705</td>
      <td>0.300800</td>
      <td>69.126721</td>
      <td>2.426858</td>
      <td>1.677840</td>
      <td>0.020778</td>
      <td>0.104575</td>
      <td>68.725060</td>
      <td>2.409873</td>
      <td>0.137052</td>
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
    </tr>
    <tr>
      <th>2_33</th>
      <td>0.003962</td>
      <td>0.255299</td>
      <td>118.610232</td>
      <td>2.502032</td>
      <td>-0.540786</td>
      <td>0.016619</td>
      <td>0.221629</td>
      <td>146.104445</td>
      <td>2.998909</td>
      <td>1.133978</td>
    </tr>
    <tr>
      <th>2_34</th>
      <td>0.011230</td>
      <td>0.310389</td>
      <td>150.700497</td>
      <td>3.062124</td>
      <td>-0.755616</td>
      <td>0.023785</td>
      <td>0.114924</td>
      <td>132.171743</td>
      <td>3.094738</td>
      <td>-1.646757</td>
    </tr>
    <tr>
      <th>2_35</th>
      <td>0.007504</td>
      <td>1.269198</td>
      <td>184.508856</td>
      <td>4.672236</td>
      <td>9.241733</td>
      <td>0.003259</td>
      <td>0.559806</td>
      <td>64.591090</td>
      <td>2.472077</td>
      <td>-0.814155</td>
    </tr>
    <tr>
      <th>2_36</th>
      <td>0.004319</td>
      <td>0.504216</td>
      <td>144.844016</td>
      <td>3.262612</td>
      <td>7.188011</td>
      <td>0.017941</td>
      <td>1.124204</td>
      <td>142.120547</td>
      <td>6.391043</td>
      <td>6.352378</td>
    </tr>
    <tr>
      <th>2_37</th>
      <td>0.002803</td>
      <td>0.772169</td>
      <td>99.716798</td>
      <td>3.186339</td>
      <td>4.663551</td>
      <td>0.002151</td>
      <td>0.159758</td>
      <td>87.144789</td>
      <td>2.468882</td>
      <td>1.216778</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 10 columns</p>
</div>


