import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def scanpath_visualization(data: pd.DataFrame, x: str, y: str, duration: str = None, dispersion: str = None,
                           img_path: str = None, fig_size: tuple = (10, 10), points_width: float = 75,
                           path_width: float = 2, points_color: str = None, path_color: str = 'green',
                           points_enumeration: bool = False, regression_color: str = None):
    plt.figure(figsize=fig_size)
    X, Y = data[x], data[y]
    dX, dY = data[x].diff(), data[y].diff()
    XY = pd.concat([dX, dY], axis=1)
    fixation_size = np.arange(points_width, X.shape[0])
    reg_only = XY[ ( XY.iloc[:, 0] < -1 ) | ( XY.iloc[:, 1] < -1 ) ]
    if duration is not None:
        dur = data[duration]
        dur /= dur.max()
        fixation_size = np.array(dur * points_width)
    if dispersion is not None:
        disp = data[dispersion]
        disp /= disp.max()
        fixation_size = np.array(disp * points_width)

    result = plt.subplot()
    result.imshow(plt.imread(img_path))
    plt.scatter(X, Y, s=fixation_size, color=points_color)
    if points_enumeration:
        enumeration = range(dX.shape[0])
        for i in enumeration:
            plt.annotate(i, xy=(X[i], Y[i] + (max(fixation_size) / 2)))

    plt.plot([X.iloc[:-1], X.shift(-1).iloc[:-1]], [Y.iloc[:-1], Y.shift(-1).iloc[:-1]], color=path_color,
             linewidth=path_width)
    if regression_color is not None:
        regX = [(X.iloc[i - 1], X.iloc[i]) for i in reg_only.index]
        regY = [(Y.iloc[i - 1], Y.iloc[i]) for i in reg_only.index]
        for i in range(len(regX)):
            plt.plot(regX[i], regY[i], color=regression_color, linewidth=path_width)
    return
