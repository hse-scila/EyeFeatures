from typing import Dict, Tuple, Union
from numpy.typing import NDArray

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from eyetracking.utils import _select_regressions


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
    fig_size: tuple[float, float] = (10.0, 10.0),
    points_width: float = 75,
    path_width: float = 0.004,
    points_color: str = None,
    path_color: str = "green",
    points_enumeration: bool = False,
    regression_color: str = None,
    micro_sac_color: str = None,
    is_vectors: bool = False,
    min_dispersion: float = 1.2,
    max_velocity: float = 4.7,
    aoi_c: Dict[str, str] = None,
    only_points: bool = False,
    seq_colormap: bool = False,
    show_hull: bool = False,
    show_legend: bool = False,
    path_to_img: str = None,
    with_axes: bool = False,
    axes_limits: tuple = None,
    rule: Tuple[int, ...] = None,
    deviation: Union[int, Tuple[int, ...]] = None,
):
    plt.figure(figsize=fig_size)
    eps = 1e-8
    marks = ("o", "^", "s", "*", "p")
    legend = dict()
    data = data_.copy()
    if aoi is not None:
        data.dropna(subset=[aoi], inplace=True)

    data.reset_index(inplace=True, drop=True)
    X, Y = data[x], data[y]
    dX, dY = data[x].diff(), data[y].diff()
    XY = pd.concat([dX, dY], axis=1)
    fixation_size = np.full(X.shape[0], points_width)
    reg_only = XY[(XY.iloc[:, 0] < 0) | (XY.iloc[:, 1] > 0)]

    if duration is not None:
        dur_intervals = np.linspace(0, data[duration].max(), 6)
        for i in range(1, len(dur_intervals)):
            # legend.append(
            #     f"[{round(dur_intervals[i - 1], 2)}, {round(dur_intervals[i], 2)})"
            # )
            legend[marks[i - 1]] = (
                f"[{round(dur_intervals[i - 1], 2)}, {round(dur_intervals[i], 2)})"
            )
            data.loc[
                (data["duration"] >= dur_intervals[i - 1])
                & (data["duration"] < dur_intervals[i]),
                "mark",
            ] = marks[i - 1]
        markers = marks
    else:
        markers = [marks[0]]
        data["mark"] = marks[0]

    if aoi is not None:
        if aoi_c is None:
            get_aoi_cm = plt.cm.get_cmap(lut=len(data[aoi].drop_duplicates().values))
            aoi_c = dict()
            for i in range(len(data[aoi].unique())):
                aoi_c[data[aoi].unique()[i]] = get_aoi_cm(i)
        data["color"] = data[aoi].map(aoi_c)

    if dispersion is not None:  # Not used
        disp = data[dispersion]
        disp /= disp.max()
        fixation_size = np.array(disp * points_width)

    plt.axis(axes_limits)

    if img_path is not None:
        plt.imshow(plt.imread(img_path))

    for i in range(len(markers)):
        if duration is not None:
            plt.scatter(
                x=data[x][data["mark"] == markers[i]],
                y=data[y][data["mark"] == markers[i]],
                color=data["color"][data["mark"] == markers[i]],
                marker=markers[i],
                edgecolors="black",
                label=legend[marks[i - 1]],
            )
        else:
            plt.scatter(
                x=data[x][data["mark"] == markers[i]],
                y=data[y][data["mark"] == markers[i]],
                color=data["color"][data["mark"] == markers[i]],
                marker=markers[i],
                edgecolors="black",
            )

    if aoi is not None and show_hull:
        for area in data[aoi].drop_duplicates().values:
            points = data[data[aoi] == area]
            assert points[x].shape[0] > 2, "Error: Need more points for aoi"
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
            # legend[area] = area
    elif aoi is not None and not show_hull:
        for area in data[aoi].drop_duplicates().values:
            plt.fill(
                data.loc[(data[aoi] == area), (data.columns.isin([x]))].values[0],
                data.loc[(data[aoi] == area), (data.columns.isin([y]))].values[0],
                color=aoi_c[area],
                alpha=0.5,
                label=area,
            )

    if show_legend:
        plt.legend()

    if points_enumeration:
        enumeration = range(dX.shape[0])
        for i in enumeration:
            plt.annotate(str(i), xy=(X[i], Y[i]))

    if not only_points:
        c_sac = np.array(mpl.colors.to_rgb(path_color))
        if seq_colormap:
            ls = np.linspace(0, 1, X.shape[0])
        else:
            ls = np.ones(X.shape[0])

        if not is_vectors:
            for i in range(len(X) - 1):

                plt.plot(
                    [X.iloc[i], X.iloc[i + 1]],
                    [Y.iloc[i], Y.iloc[i + 1]],
                    color=c_sac * ls[i],
                    linewidth=path_width,
                )
            if regression_color is not None:
                mask = _select_regressions(dX, dY, rule, deviation)
                c_reg = np.array(mpl.colors.to_rgb(regression_color))
                regX = dX[mask]
                for i in regX.index:
                    if i != 0:
                        plt.plot(
                            [X.iloc[i - 1], X.iloc[i]],
                            [Y.iloc[i - 1], Y.iloc[i]],
                            color=c_reg * ls[i],
                            linewidth=path_width,
                        )
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
                    width=path_width,
                    edgecolor="yellow",
                    linewidth=path_width / 2,
                )
            if regression_color is not None:
                mask = _select_regressions(dX, dY, rule, deviation)

                c_reg = np.array(mpl.colors.to_rgb(regression_color))
                regX = dX[mask]
                for i in regX.index:
                    if i != 0:
                        plt.quiver(
                            X.iloc[i - 1],
                            Y.iloc[i - 1],
                            X.iloc[i] - X.iloc[i - 1],
                            Y.iloc[i] - Y.iloc[i - 1],
                            angles="xy",
                            scale_units="xy",
                            color=c_reg * ls[i],
                            scale=1,
                            width=path_width,
                            edgecolor="yellow",
                            linewidth=path_width / 2,
                        )

        if (
            micro_sac_color is not None
        ):  # TODO (doesn't work with normalized coordinates)
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
            dt = data[time_stamps] - (data[time_stamps] + data[duration] / 1000).shift(
                1
            )
            v = dr / (dt + eps)

            mic_sac = data[(data[dispersion] > min_dispersion) & (v < max_velocity)]
            mic_sacX = np.array([(X.iloc[i - 1], X.iloc[i]) for i in mic_sac.index])
            mic_sacY = np.array([(Y.iloc[i - 1], Y.iloc[i]) for i in mic_sac.index])
            for i in range(len(mic_sacX)):
                plt.plot(
                    mic_sacX[i],
                    mic_sacY[i],
                    color=micro_sac_color,
                    linewidth=path_width,
                )
    else:
        plt.show()
    if path_to_img is not None:
        if not with_axes:
            plt.axis("off")
        plt.savefig(path_to_img, dpi=100)
        plt.axis("on")

    return
