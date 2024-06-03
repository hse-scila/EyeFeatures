from typing import Dict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull


def _cmap_generation(n: int):
    return plt.cm.get_cmap("tab20")(n)


def scanpath_visualization(
    data_: pd.DataFrame,
    x: str,
    y: str,
    duration: str = None,
    dispersion: str = None,
    time_stamps: str = None,
    aoi: str = None,
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
    aoi_c: Dict[str, mpl.colors] = None,
    only_points: bool = False,
    seq_colormap: bool = False,
    show_hull: bool = False,
    show_legend: bool = False,
    path_to_img: str = None,
):
    plt.figure(figsize=fig_size)
    eps = 1e-8
    marks = ("o", "^", "s", "*", "p")
    legend = []
    data = data_.copy()
    if aoi is not None:
        data.dropna(subset=[aoi], inplace=True)

    data.reset_index(inplace=True, drop=True)
    X, Y = data[x], data[y]
    dX, dY = data[x].diff(), data[y].diff()
    XY = pd.concat([dX, dY], axis=1)
    fixation_size = np.arange(points_width, X.shape[0])
    reg_only = XY[(XY.iloc[:, 0] < 0) | (XY.iloc[:, 1] > 0)]

    if duration is not None:
        dur_intervals = np.linspace(0, data[duration].max(), 6)
        for i in range(1, len(dur_intervals)):
            legend.append(
                f"[{round(dur_intervals[i - 1], 2)}, {round(dur_intervals[i], 2)})"
            )

    if dispersion is not None:
        disp = data[dispersion]
        disp /= disp.max()
        fixation_size = np.array(disp * points_width)

    if img_path is not None:
        plt.imshow(plt.imread(img_path))

    if aoi is None:
        if duration is None:
            plt.scatter(X, Y, s=fixation_size, color=points_color)
        else:
            for i in range(1, len(dur_intervals)):
                points_dur = data[
                    (data[duration] >= dur_intervals[i - 1])
                    & (data[duration] < dur_intervals[i])
                ]
                plt.scatter(
                    x=points_dur[x],
                    y=points_dur[y],
                    label=legend[i - 1],
                    edgecolors="black",
                    marker=marks[i - 1],
                )
    else:
        once = True
        if aoi_c is None:
            get_aoi_cm = plt.cm.get_cmap(lut=len(data[aoi].drop_duplicates().values))
            aoi_c = dict()
            for i in range(len(data[aoi].unique())):
                aoi_c[data[aoi].unique()[i]] = get_aoi_cm(i)
        for area in data[aoi].drop_duplicates().values:
            points = data[data[aoi] == area]
            if duration is not None:
                for i in range(1, len(dur_intervals)):
                    points_dur = points[
                        (points[duration] >= dur_intervals[i - 1])
                        & (points[duration] < dur_intervals[i])
                    ]
                    if once:
                        plt.scatter(
                            x=points_dur[x],
                            y=points_dur[y],
                            color=aoi_c[area],
                            label=legend[i - 1],
                            edgecolors="black",
                            marker=marks[i - 1],
                        )
                    else:
                        plt.scatter(
                            x=points_dur[x],
                            y=points_dur[y],
                            color=aoi_c[area],
                            edgecolors="black",
                            marker=marks[i - 1],
                        )
            else:
                if not show_hull:
                    plt.scatter(
                        x=points[x],
                        y=points[y],
                        color=aoi_c[area],
                        label=area,
                        edgecolors="black",
                    )
                else:
                    plt.scatter(
                        x=points[x],
                        y=points[y],
                        color=aoi_c[area],
                        edgecolors="black",
                    )
            if show_hull:
                points_num = np.array([points[x].to_numpy(), points[y].to_numpy()])
                points_num = points_num.T

                hull = ConvexHull(points_num)
                for simplex in hull.simplices:
                    plt.plot(
                        points_num[simplex, 0],
                        points_num[simplex, 1],
                        "-",
                        color=aoi_c[area],
                    )
                plt.fill(
                    points_num[hull.vertices, 0],
                    points_num[hull.vertices, 1],
                    color=aoi_c[area],
                    alpha=0.5,
                    label=area,
                )
            once = False
    if show_legend:
        plt.legend()

    if points_enumeration:
        enumeration = range(dX.shape[0])
        for i in enumeration:
            plt.annotate(i, xy=(X[i], Y[i]))

    if only_points:
        plt.show()
        return

    c_sac = np.array(mpl.colors.to_rgb(path_color))
    if seq_colormap:
        ls = np.linspace(0, 1, len(fixation_size))
    else:
        ls = np.ones(len(fixation_size))

    if not is_vectors:
        for i in range(len(X) - 1):
            plt.plot(
                [X.iloc[i], X.iloc[i + 1]],
                [Y.iloc[i], Y.iloc[i + 1]],
                color=c_sac * ls[i],
                linewidth=path_width,
            )
        if regression_color is not None:
            c_reg = np.array(mpl.colors.to_rgb(regression_color))
            regX = [(X.iloc[i - 1], X.iloc[i]) for i in reg_only.index]
            regY = [(Y.iloc[i - 1], Y.iloc[i]) for i in reg_only.index]
            for i in range(len(regX)):
                plt.plot(regX[i], regY[i], color=c_reg * ls[i], linewidth=path_width)
    else:
        for i in range(len(X) - 1):
            plt.quiver(
                X.iloc[i],
                Y.iloc[i],
                X.iloc[i + 1] - X.iloc[i],
                Y.iloc[i + 1] - Y.iloc[i],
                angles="xy",
                scale_units="xy",
                color=c_sac * ls[i],
                scale=1,
                width=path_width / 500,
                edgecolor="yellow",
                linewidth=path_width / 1000,
            )
        if regression_color is not None:
            c_reg = np.array(mpl.colors.to_rgb(regression_color))
            regX = np.array([(X.iloc[i - 1], X.iloc[i]) for i in reg_only.index])
            regY = np.array([(Y.iloc[i - 1], Y.iloc[i]) for i in reg_only.index])
            for i in range(len(regX)):
                plt.quiver(
                    regX[i][0],
                    regY[i][0],
                    regX[i][1] - regX[i][0],
                    regY[i][1] - regY[i][0],
                    angles="xy",
                    scale_units="xy",
                    color=c_reg * ls[i],
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
    if path_to_img is not None:
        plt.axis("off")
        plt.savefig(path_to_img)
        plt.axis("on")
    # plt.show()
    return
