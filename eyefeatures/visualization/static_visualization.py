import io
from typing import Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from eyefeatures.utils import _select_regressions, _split_dataframe


def _cmap_generation(n: int):
    return mpl.colormaps["tab20"](n)


def scanpath_visualization(
    data_: pd.DataFrame,
    x: str,
    y: str,
    shape_column: str = None,
    aoi: str = None,
    img_path: str = None,
    fig_size: tuple[float, float] = (10.0, 10.0),
    points_width: float = 75,
    path_width: float = 0.004,
    points_color: str = "blue",
    path_color: str = "green",
    points_enumeration: bool = False,
    add_regressions: bool = False,
    regression_color: str = "red",
    is_vectors: bool = False,
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
    return_ndarray: bool = False,
    show_plot: bool = True,
    is_gray: bool = False,
    dpi: float = 100.0,
):
    """Function for scanpath and/or aoi visualization.

    Args:
        data_: DataFrame with fixations.
        x: x coordinate of fixation.
        y: y coordinate of fixation.
        size_column: label of the column, which is responsible for the size
                     of the fixations(points on plot).\n
                     It can be duration, dispersion, etc.
        shape_column: label of the column, which is responsible for the shape
                      of the fixations(points on plot).
        aoi: AOI of fixations.
        img_path: path to the background image.
        fig_size: size of plot.
        points_width: width of points.
        path_width: width of path.
        points_color: color of points.
        path_color: color of path.
        points_enumeration: whether to enumerate points.
        add_regressions: whether to add regressions.
        regression_color: color of regressions.
        is_vectors: whether to visualize saccades as vectors
        aoi_c: colormap for AOI.
        only_points: whether to only show points.
        seq_colormap: whether to show sequentially-colored saccades.
        show_hull: whether to show hull of AOI.
        show_legend: whether to show legend.
        path_to_img: path to save the plot image.
        with_axes: whether to show axes.
        axes_limits: limits of axes.
        rule: must be either 1) tuple of quadrants direction to classify
             regressions, 1st quadrant being upper-right square of plane and counting
             anti-clockwise or 2) tuple of angles in degrees (0 <= angle <= 360).
        deviation: if None, then `rule` is interpreted as quadrants. Otherwise,
                  `rule` is interpreted as angles. If integer, then is a
                  +-deviation for all angles. If tuple of integers, then
                  must be of the same length as `rule`, each value being
                  a corresponding deviation for each angle. Angle = 0 is positive
                  x-axis direction, rotating anti-clockwise.
        return_ndarray: whether to return numpy array of the plot image
                  (returns RGBA array).
        show_plot: whether to show the plot.
        is_gray: whether to use the gray scale.
        dpi: dpi for output image.
    """
    plt.figure(figsize=fig_size)

    marks = ("o", "^", "s", "*", "p")
    legend = dict()
    data = data_.copy()
    data["color"] = points_color
    if aoi is not None:
        data.dropna(subset=[aoi], inplace=True)

    data.reset_index(inplace=True, drop=True)
    X, Y = data[x], data[y]
    dX, dY = data[x].diff(), data[y].diff()

    if shape_column is not None:
        intervals = np.linspace(0, data[shape_column].max(), 6)
        for i in range(1, len(intervals)):
            legend[marks[i - 1]] = (
                f"[{round(intervals[i - 1], 2)}, {round(intervals[i], 2)})"
            )
            data.loc[
                (data[shape_column] >= intervals[i - 1])
                & (data[shape_column] < intervals[i]),
                "mark",
            ] = marks[i - 1]
        markers = marks
    else:
        markers = [marks[0]]
        data["mark"] = marks[0]

    if aoi is not None:
        if aoi_c is None:
            n_aois = len(data[aoi].unique())
            get_aoi_cm = mpl.colormaps["tab20"].resampled(n_aois)
            aoi_c = dict()
            for i, val in enumerate(data[aoi].unique()):
                aoi_c[val] = get_aoi_cm(i)
        data["color"] = data[aoi].map(aoi_c)

    plt.axis(axes_limits)

    if img_path is not None:
        plt.imshow(plt.imread(img_path))

    for i in range(len(markers)):
        if shape_column is not None:
            plt.scatter(
                x=data[x][data["mark"] == markers[i]],
                y=data[y][data["mark"] == markers[i]],
                color=data["color"][data["mark"] == markers[i]],
                marker=markers[i],
                edgecolors="black",
                label=legend[marks[i - 1]],
                s=points_width,
            )
        else:
            plt.scatter(
                x=data[x][data["mark"] == markers[i]],
                y=data[y][data["mark"] == markers[i]],
                color=data["color"][data["mark"] == markers[i]],
                marker=markers[i],
                edgecolors="black",
                s=points_width,
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
            if add_regressions:
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
                    linewidth=path_width,
                )
            if add_regressions:
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
                            linewidth=path_width,
                        )

    if not with_axes:
        plt.axis("off")

    arr = None

    if return_ndarray:
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, dpi=dpi, format="png")
        im = Image.open(image_buffer).convert("RGB")
        if is_gray:
            im = ImageOps.grayscale(im)
        arr = np.array(im) / 255
        image_buffer.close()

    if path_to_img is not None:
        plt.savefig(path_to_img, dpi=dpi)
        plt.axis("on")
    if not show_plot:
        plt.close()
    return arr


class Visualization(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        x: str,
        y: str,
        shape_column: str = None,
        aoi: str = None,
        img_path: str = None,
        fig_size: tuple[float, float] = (10.0, 10.0),
        points_width: float = 75,
        path_width: float = 0.004,
        points_color: str = "blue",
        path_color: str = "green",
        points_enumeration: bool = False,
        add_regressions: bool = False,
        regression_color: str = "red",
        is_vectors: bool = False,
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
        is_gray: bool = False,
    ):
        self.x = x
        self.y = y
        self.shape_column = shape_column
        self.aoi = aoi
        self.img_path = img_path
        self.fig_size = fig_size
        self.points_width = points_width
        self.points_color = points_color
        self.path_color = path_color
        self.add_regressions = add_regressions
        self.regression_color = regression_color
        self.is_vectors = is_vectors
        self.show_legend = show_legend
        self.path_to_img = path_to_img
        self.path_width = path_width
        self.points_enumeration = points_enumeration
        self.show_hull = show_hull
        self.aoi_c = aoi_c
        self.only_points = only_points
        self.seq_colormap = seq_colormap
        self.with_axes = with_axes
        self.axes_limits = axes_limits
        self.rule = rule
        self.deviation = deviation
        self.return_ndarray = True
        self.show_plot = False
        self.is_gray = is_gray

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return scanpath_visualization(data_=X, **self.__dict__)


def get_visualizations(
    data: pd.DataFrame,
    x: str,
    y: str,
    shape: tuple[int, int],
    pattern: str,
    dpi: float = 100.0,
    pk: List[str] = None,
):
    """Get visualizations.

    Args:
        data: input dataframe with fixations.
        x: X coordinate column name.
        y: Y coordinate column name.
        shape: height and width in pixels.
        pattern: visualization class to use.
        dpi: dpi for images.
        pk: list of column names used to split pd.DataFrame.

    Returns:
        output: tensor of shape [n, m, fig, fig, c], where\n
             n   : instances\n
             m   : patterns\n
             fig : fig size\n
             c   : (3 - RGB image; 1 - gray scaled image)
    """
    arr = []
    if pk is None:
        if pattern == "baseline":
            res = baseline_visualization(data, x, y, shape)
        elif pattern == "aoi":
            res = aoi_visualization(data, x, y, shape, aoi="AOI")
        elif pattern == "saccades":
            res = saccade_visualization(data, x, y, shape)
        else:
            raise ValueError(f"Unsupported pattern: {pattern}")
        arr.append(res)
    else:
        groups: List[Tuple[str, pd.DataFrame]] = _split_dataframe(
            data, pk, encode=False
        )
        for group_id, group_X in tqdm(groups):
            if pattern == "baseline":
                res = baseline_visualization(
                    group_X, x, y, shape, show_plot=False, dpi=dpi
                )
            elif pattern == "aoi":
                res = aoi_visualization(
                    group_X, x, y, shape, aoi="AOI", show_plot=False, dpi=dpi
                )
            elif pattern == "saccades":
                res = saccade_visualization(
                    group_X, x, y, shape, show_plot=False, dpi=dpi
                )
            else:
                raise ValueError(f"Unsupported pattern: {pattern}")
            arr.append(res)
    arr = np.transpose(np.array(arr), (0, 3, 1, 2))
    print(arr.shape)
    return arr


def baseline_visualization(
    data_: pd.DataFrame,
    x: str,
    y: str,
    shape: tuple[int, int] = (10, 10),
    path_width: float = 1,
    show_legend: bool = False,
    path_to_img: str = None,
    with_axes: bool = False,
    show_plot: bool = False,
    return_ndarray: bool = True,
    dpi: float = 100.0,
):
    return scanpath_visualization(
        data_,
        x,
        y,
        fig_size=shape,
        show_legend=show_legend,
        path_to_img=path_to_img,
        with_axes=with_axes,
        path_width=path_width,
        return_ndarray=return_ndarray,
        show_plot=show_plot,
        dpi=dpi,
    )


def aoi_visualization(
    data_: pd.DataFrame,
    x: str,
    y: str,
    shape: tuple[int, int] = (10, 10),
    aoi: str = "AOI",
    shape_column: str = None,
    img_path: str = None,
    points_width: float = 75,
    path_width: float = 1,
    points_color: str = None,
    aoi_c: Dict[str, str] = None,
    seq_colormap: bool = False,
    show_legend: bool = False,
    path_to_img: str = None,
    with_axes: bool = False,
    axes_limits: tuple = None,
    return_ndarray: bool = True,
    show_plot: bool = True,
    only_points: bool = True,
    dpi: float = 100.0,
):
    return scanpath_visualization(
        data_,
        x,
        y,
        shape_column=shape_column,
        aoi=aoi,
        img_path=img_path,
        fig_size=shape,
        points_width=points_width,
        path_width=path_width,
        points_color=points_color,
        seq_colormap=seq_colormap,
        show_legend=show_legend,
        aoi_c=aoi_c,
        with_axes=with_axes,
        axes_limits=axes_limits,
        path_to_img=path_to_img,
        show_hull=True,
        only_points=only_points,
        return_ndarray=return_ndarray,
        show_plot=show_plot,
        dpi=dpi,
    )


def saccade_visualization(
    data_: pd.DataFrame,
    x: str,
    y: str,
    shape: tuple[int, int] = (10, 10),
    shape_column: str = None,
    img_path: str = None,
    path_width: float = 1,
    path_color: str = "green",
    add_regressions: bool = False,
    regression_color: str = "red",
    is_vectors: bool = False,
    seq_colormap: bool = False,
    show_legend: bool = False,
    path_to_img: str = None,
    with_axes: bool = False,
    axes_limits: tuple = None,
    rule: Tuple[int, ...] = (2,),
    deviation: Union[int, Tuple[int, ...]] = None,
    return_ndarray: bool = True,
    show_plot: bool = True,
    dpi: float = 100.0,
):
    return scanpath_visualization(
        data_,
        x,
        y,
        shape_column=shape_column,
        img_path=img_path,
        fig_size=shape,
        path_width=path_width,
        seq_colormap=seq_colormap,
        show_legend=show_legend,
        path_to_img=path_to_img,
        with_axes=with_axes,
        axes_limits=axes_limits,
        rule=rule,
        deviation=deviation,
        path_color=path_color,
        regression_color=regression_color,
        add_regressions=add_regressions,
        is_vectors=is_vectors,
        return_ndarray=return_ndarray,
        show_plot=show_plot,
        dpi=dpi,
    )
