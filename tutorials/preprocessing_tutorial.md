# EyeFeatures: From Gazes to Fixations

Process of extracting fixations from raw gazes determines all future characteristics of fixations and saccades, thus being quite important. This tutorial covers `eyetracking.preprocessing` module which offers algorithms to control this extraction and to make it as smooth as possible.

Let's load simple dataset with raw gazes.


```python
import os
from os.path import join

import pandas as pd

import warnings
warnings.simplefilter("ignore")
```


```python
def get_movies_dataset():
    """
    Read Movies Dataset from https://www.inb.uni-luebeck.de/index.php?id=515, statimages.
    * gaze_x: x-coordinate of gaze.
    * gaze_y: y-coordinate of gaze.
    * timestamp: number of milliseconds from start of the recording.
    * film: name of movie.
    * frame: number of frame.
    * subject: person identifier.
    """
    df = pd.read_csv(join('data', 'movies_data.csv'), index_col=False)
    return df
```


```python
data = get_movies_dataset()
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>film</th>
      <th>frame</th>
      <th>timestamp</th>
      <th>gaze_x</th>
      <th>gaze_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V10</td>
      <td>holsten_gate</td>
      <td>0</td>
      <td>3.667</td>
      <td>0.499219</td>
      <td>0.501482</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V10</td>
      <td>holsten_gate</td>
      <td>0</td>
      <td>7.581</td>
      <td>0.499219</td>
      <td>0.502962</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V10</td>
      <td>holsten_gate</td>
      <td>0</td>
      <td>9.757</td>
      <td>0.500000</td>
      <td>0.501482</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V10</td>
      <td>holsten_gate</td>
      <td>0</td>
      <td>13.774</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V10</td>
      <td>holsten_gate</td>
      <td>0</td>
      <td>18.686</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>261588</th>
      <td>V04</td>
      <td>st_petri_market</td>
      <td>270</td>
      <td>1982.363</td>
      <td>0.450781</td>
      <td>0.704444</td>
    </tr>
    <tr>
      <th>261589</th>
      <td>V04</td>
      <td>st_petri_market</td>
      <td>270</td>
      <td>1986.301</td>
      <td>0.450781</td>
      <td>0.704444</td>
    </tr>
    <tr>
      <th>261590</th>
      <td>V04</td>
      <td>st_petri_market</td>
      <td>270</td>
      <td>1990.346</td>
      <td>0.450781</td>
      <td>0.704444</td>
    </tr>
    <tr>
      <th>261591</th>
      <td>V04</td>
      <td>st_petri_market</td>
      <td>270</td>
      <td>1994.284</td>
      <td>0.450000</td>
      <td>0.704444</td>
    </tr>
    <tr>
      <th>261592</th>
      <td>V04</td>
      <td>st_petri_market</td>
      <td>270</td>
      <td>1998.350</td>
      <td>0.450000</td>
      <td>0.704444</td>
    </tr>
  </tbody>
</table>
<p>261593 rows × 6 columns</p>
</div>




```python
x = 'gaze_x'
y = 'gaze_y'
t = 'timestamp'
pk = ['subject', 'film']
```

Let's pick some object and keep track of it. In case of this dataset, single object is triple (subject, film, frame). Since each object contains small amount of data, we can analyze scanpath of particular person watching a particular movie, i.e. ignoring frame and defining object as (subject, film).


```python
subject = 'V10'
film = 'holsten_gate'

def get_object(df: pd.DataFrame):
    filter_object = (df['subject'] == subject) & (df['film'] == film)
    return df[filter_object]
```

Let's use EyeFeatures toolkit to get a grasp of how our object's scanpath looks like.


```python
from eyetracking.visualization.static_visualization import scanpath_visualization
```


```python
scanpath_visualization(get_object(data), x, y, with_axes=True, path_width=1)
```


    
![png](images/prep_output_9_0.png)
    


### 1. Fixation Extraction

While squashing gazes, we would like to extract fixations and keep the path trajectory. Thus, the general approach is as follows: mark each gaze with $0$ (not a part of fixation) or $1$ (a part of fixation), then squash consecutive ones into single fixation.  Here we denote $n$ fixations as triplets $\{(x_i, y_i, t_i)\}_{i=1}^{n}$ - x coordinate, y coordinate, timestamp. `eyetracking.preprocessing.fixation_extraction` has three algorithms for fixations extraction:

1. IVT  - velocity threshold identification algorithm. Gazes that have velocity below threshold are considered to be fixations, since high velocities are attributes of saccades. If $a$ is an algorithm and $d$ some metric in $\mathbb{R}^2$, then for single fixation:

$$a(i) = \left[\frac{d((x_i, y_i), (x_{i + 1}, y_{i + 1}))}{t_{i + 1} - t_i} \leq T\right]$$

2. IDT  - dispersion threshold identification algorithm. This algorithm uses sliding window to find consecutive gazes with dispersion less than `max_dispersion` and duration more than `min_duration`. These heuristics ensure that extracted fixations have small variance and their duration is long enough.


3. IHMM - hidden Markov model identification algorithm. Algorithm finds a sequence of fixations that maximizes the log probability of observing given velocities of gazes under conditions of Hidden Markov Model. More formally, denote velocity of $i$-th gaze as $\displaystyle v_i = \frac{d((x_i, y_i), (x_{i + 1}, y_{i + 1}))}{t_{i + 1} - t_i}$ - this is observed process, while hidden process is sequence of zeros and ones $\{s_i\}_{i=1}^{n}$, as discussed previously, $1$ indicating fixation. We fix some prior distribution of velocities (normal is taken as empirical rule) and transition matrix, then, under assumption of Markov process, i.e. $P(s_i = b|v_{i - 1}, ..., v_1) = P(s_i = b|v_{i - 1}, ..., v_{i - k})$ for some $k \geq 1$, probability is maximized in greedy manner.

Let's use IDT since it is a common choice among eyetracking software products.


```python
from eyetracking.preprocessing.fixation_extraction import IDT
```

IDT has two parameters: `min_duration` of fixation and `max_dispersion` of gazes within fixation. They provide a great control over the desired results.


```python
preprocessor = IDT(x=x, y=y, t=t, pk=pk, min_duration=1e-5, max_dispersion=1e-3)

fixations = preprocessor.fit_transform(data)
fixations
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>film</th>
      <th>gaze_x</th>
      <th>gaze_y</th>
      <th>start_time</th>
      <th>end_time</th>
      <th>distance_min</th>
      <th>distance_max</th>
      <th>dispersion</th>
      <th>duration</th>
      <th>saccade_duration</th>
      <th>saccade_length</th>
      <th>saccade_angle</th>
      <th>saccade2_angle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>V01</td>
      <td>doves</td>
      <td>0.502003</td>
      <td>0.931320</td>
      <td>3.249</td>
      <td>1133.132</td>
      <td>0.000000</td>
      <td>0.001675</td>
      <td>0.000781</td>
      <td>1129.883</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>V02</td>
      <td>bridge_2</td>
      <td>0.500647</td>
      <td>0.501644</td>
      <td>4.449</td>
      <td>254.506</td>
      <td>0.000000</td>
      <td>0.002964</td>
      <td>0.000781</td>
      <td>250.057</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>V02</td>
      <td>bridge_2</td>
      <td>0.402734</td>
      <td>0.442963</td>
      <td>290.371</td>
      <td>294.465</td>
      <td>0.001563</td>
      <td>0.001675</td>
      <td>0.000781</td>
      <td>4.094</td>
      <td>35.865</td>
      <td>0.114150</td>
      <td>210.934825</td>
      <td>19.111463</td>
    </tr>
    <tr>
      <th>3</th>
      <td>V02</td>
      <td>bridge_2</td>
      <td>0.409743</td>
      <td>0.451329</td>
      <td>306.451</td>
      <td>438.509</td>
      <td>0.000000</td>
      <td>0.003064</td>
      <td>0.000781</td>
      <td>132.058</td>
      <td>11.986</td>
      <td>0.010913</td>
      <td>50.046288</td>
      <td>48.033449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>V02</td>
      <td>bridge_2</td>
      <td>0.203320</td>
      <td>0.444074</td>
      <td>502.479</td>
      <td>514.339</td>
      <td>0.000781</td>
      <td>0.004444</td>
      <td>0.000781</td>
      <td>11.860</td>
      <td>63.970</td>
      <td>0.206550</td>
      <td>182.012839</td>
      <td>40.667033</td>
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
    </tr>
    <tr>
      <th>6319</th>
      <td>V11</td>
      <td>street</td>
      <td>0.884953</td>
      <td>0.585911</td>
      <td>1293.793</td>
      <td>1689.775</td>
      <td>0.000000</td>
      <td>0.001675</td>
      <td>0.000781</td>
      <td>395.982</td>
      <td>8.035</td>
      <td>0.007695</td>
      <td>222.514354</td>
      <td>158.264124</td>
    </tr>
    <tr>
      <th>6320</th>
      <td>V11</td>
      <td>street</td>
      <td>0.881576</td>
      <td>0.584630</td>
      <td>1697.815</td>
      <td>1885.798</td>
      <td>0.000000</td>
      <td>0.002773</td>
      <td>0.000781</td>
      <td>187.983</td>
      <td>8.040</td>
      <td>0.003613</td>
      <td>200.778478</td>
      <td>189.649606</td>
    </tr>
    <tr>
      <th>6321</th>
      <td>V11</td>
      <td>street</td>
      <td>0.809375</td>
      <td>0.542222</td>
      <td>1909.753</td>
      <td>1913.711</td>
      <td>0.001481</td>
      <td>0.004513</td>
      <td>0.000000</td>
      <td>3.958</td>
      <td>23.955</td>
      <td>0.083734</td>
      <td>210.428084</td>
      <td>11.315721</td>
    </tr>
    <tr>
      <th>6322</th>
      <td>V11</td>
      <td>street</td>
      <td>0.816016</td>
      <td>0.548148</td>
      <td>1925.794</td>
      <td>1929.769</td>
      <td>0.000781</td>
      <td>0.001675</td>
      <td>0.000781</td>
      <td>3.975</td>
      <td>12.083</td>
      <td>0.008900</td>
      <td>41.743804</td>
      <td>1.723784</td>
    </tr>
    <tr>
      <th>6323</th>
      <td>V11</td>
      <td>street</td>
      <td>0.812109</td>
      <td>0.544868</td>
      <td>1941.730</td>
      <td>1993.637</td>
      <td>0.000000</td>
      <td>0.001675</td>
      <td>0.000781</td>
      <td>51.907</td>
      <td>11.961</td>
      <td>0.005101</td>
      <td>220.020020</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>6324 rows × 14 columns</p>
</div>



Preprocessing algorithms compute a variety of features alongside with fixation extraction. For example, lengths and durations of saccades between fixations, as well as angles between succeeding and preceding saccades (`saccade2_angle`).

Now, let's see how the path of our object has changed:


```python
scanpath_visualization(get_object(fixations), x, y, with_axes=True, path_width=1)
```


    
![png](images/prep_output_15_0.png)
    


### 2. Filtering gazes.

`eyetracking.preprocessing.smoothing` provides 4 filters to smooth gazes before the process of fixation extraction:
1. Savitzkiy-Golay filter - fits $k$-degree polynomial using $k$ last points in sequence using OLS.
2. FIR filter - weighted sum of $k$ previous values (convolution of signals).
3. IIR filter - similar to FIR filter but with two convolution signals.
4. Wiener filter. This filter assumes the following model of distortion:

$$g(x) = f(x) * h(x) + s(x)$$

where $f$ is true signal, $h$ is distortion signal, $*$ is convolution operation, $s$ is noise, and $g$ is distorted signal (observed signal). Wiener's approach considers input signal and noise as random variables and finds such estimator $\hat{f}$ which minimizes the variance of $\hat{f} - f$. It could be shown that in the underlined model the minimum is achieved (in frequency domain) at:

$$\hat{F}(x) = \frac{\overline{H(x)}}{|H(x)|^2 + K}G(x)$$

where
* $\hat{F}(x)$ - Fourier-image of $f$.
* $H(x)$ - Fourier-image of distorting function $h$.
* $\overline{\cdot}$ - complex inverse.
* $|\cdot|$ - complex modulus.
* $K$ - approximation constant.

Filters are often applied before fixation extraction. Let's do it using `sklearn`'s pipeline.


```python
from eyetracking.preprocessing.smoothing import WienerFilter, SavGolFilter
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ("wf_filter", WienerFilter(x=x, y=y, t=t, pk=pk, K='auto')),          # Wiener filter
    ("sg_filter", SavGolFilter(x=x, y=y, t=t, pk=pk, window_length=10)),  # Savitzkiy-Golay filter
    ("preprocessor", preprocessor)                                        # IDT algorithm
])

fixations_smooth = pipe.fit_transform(data)
```


```python
scanpath_visualization(get_object(fixations_smooth), x, y, with_axes=True, path_width=1)
```


    
![png](images/prep_output_18_0.png)
    


Number of fixations without filtering:


```python
len(get_object(fixations))
```




    103



Number of fixations with filtering:


```python
len(get_object(fixations_smooth))
```




    94



As you can see, usage of filters smoothed scanpath and resulted in smaller number of fixations. Also, coordinates of fixations are more "compact", looking at axes on last two visualizations.

### References

1. [Sample dataset](https://www.inb.uni-luebeck.de/index.php?id=515), static images.
2. [Salvucci & Goldberg (2000)](https://www.researchgate.net/publication/220811146_Identifying_fixations_and_saccades_in_eye-tracking_protocols). Identifying saccades and fixations in eye-tracking protocols. Served as the source of fixation extraction algorithms
3. Savitzkiy-Golay, FIR, IIR classes are wrappers of `scipy.signal` methods.
4. About [Wiener filter](https://en.wikipedia.org/wiki/Wiener_filter) on Wikipedia.
