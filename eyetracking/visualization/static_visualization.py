import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scanpath_visualization(
    data_: pd.DataFrame,
    x: str,
    y: str,
    duration: str = None,
    dispersion: str = None,
    time_stamps: str = None,
    img_path: str = None,
    fig_size: tuple = (10, 10),
    points_width: float = 75,
    path_width: float = 2,
    points_color: str = None,
    path_color: str = "green",
    points_enumeration: bool = False,
    regression_color: str = None,
    micro_sac_color: str = None,
    is_vectors: bool = False,
    min_dispersion: float = 1.2,
    max_velocity: float = 4.7,
):
    plt.figure(figsize=fig_size)
    eps = 1e-8
    data = data_.copy()
    data.reset_index(inplace=True, drop=True)
    X, Y = data[x], data[y]
    dX, dY = data[x].diff(), data[y].diff()
    XY = pd.concat([dX, dY], axis=1)
    print(XY)
    fixation_size = np.arange(points_width, X.shape[0])
    reg_only = XY[(XY.iloc[:, 0] < -1) | (XY.iloc[:, 1] < -1)]

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
            plt.annotate(i, xy=(X[i], Y[i] - (fixation_size[i] / 10)))

    if not is_vectors:
        plt.plot(
            [X.iloc[:-1], X.shift(-1).iloc[:-1]],
            [Y.iloc[:-1], Y.shift(-1).iloc[:-1]],
            color=path_color,
            linewidth=path_width,
        )
        if regression_color is not None:
            regX = [(X.iloc[i - 1], X.iloc[i]) for i in reg_only.index]
            regY = [(Y.iloc[i - 1], Y.iloc[i]) for i in reg_only.index]
            for i in range(len(regX)):
                plt.plot(regX[i], regY[i], color=regression_color, linewidth=path_width)
    else:
        plt.quiver(
            X.iloc[:-1],
            Y.iloc[:-1],
            X.shift(-1).iloc[:-1] - X.iloc[:-1],
            Y.shift(-1).iloc[:-1] - Y.iloc[:-1],
            angles="xy",
            scale_units="xy",
            color=path_color,
            scale=1,
            width=path_width / 500,
            edgecolor="yellow",
            linewidth=path_width / 1000,
        )
        if regression_color is not None:
            regX = np.array([(X.iloc[i - 1], X.iloc[i]) for i in reg_only.index]).T
            regY = np.array([(Y.iloc[i - 1], Y.iloc[i]) for i in reg_only.index]).T
            plt.quiver(
                regX[0],
                regY[0],
                regX[1] - regX[0],
                regY[1] - regY[0],
                angles="xy",
                scale_units="xy",
                color=regression_color,
                scale=1,
                width=path_width / 500,
                edgecolor="yellow",
                linewidth=path_width / 1000,
            )

    if micro_sac_color is not None:  # TODO (doesn't work with normalized coordinates)
        assert (
            dispersion is not None
        ), "Error: provide 'dispersion' column before calling visualization"
        assert (
            time_stamps is not None
        ), "Error: provide 'time_stamps' column before calling visualization"
        assert (
            duration is not None
        ), "Error: provide 'duration' column before calling visualization"
        dr = np.sqrt(dX**2 + dY**2)
        dt = data[time_stamps] - (data[time_stamps] + data[duration] / 1000).shift(1)
        v = dr / (dt + eps)

        mic_sac = data[(data[dispersion] > min_dispersion) & (v < max_velocity)]
        mic_sacX = np.array([(X.iloc[i - 1], X.iloc[i]) for i in mic_sac.index])
        mic_sacY = np.array([(Y.iloc[i - 1], Y.iloc[i]) for i in mic_sac.index])
        for i in range(len(mic_sacX)):
            plt.plot(
                mic_sacX[i], mic_sacY[i], color=micro_sac_color, linewidth=path_width
            )

    return
