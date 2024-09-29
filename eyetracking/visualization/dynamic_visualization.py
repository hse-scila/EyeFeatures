import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
import plotly.graph_objects as go

from eyetracking.utils import _select_regressions


def scanpath_animation(
    data_: pd.DataFrame,
    x: str,
    y: str,
    path_color: str = "green",
    path_width: float = 0.5,
    points_color: str = "black",
    points_width: float = 6,
    add_regression: bool = False,
    regression_color: str = "red",
    meta_data: List[str] = None,
    rule: Tuple[int, ...] = None,
    deviation: Union[int, Tuple[int, ...]] = None,
    aoi: str = None,
    aoi_c: Dict[str, str] = None,
):
    data = data_.reset_index(drop=True)
    X = data[x].values
    Y = data[y].values
    dX = data[x].diff()
    dY = data[y].diff()
    indexes = data.index

    fig_dict = {"data": [], "layout": {}, "frames": []}

    fig_dict["layout"]["width"] = 600
    fig_dict["layout"]["height"] = 600
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": False},
                            "fromcurrent": True,
                            "transition": {
                                "duration": 300,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0.0,
            "yanchor": "top",
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Index:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0.0,
        "steps": [],
    }

    edges = {
        "x": X,
        "y": Y,
        "mode": "lines",
        "line": dict(color=path_color, width=path_width),
        "name": "saccades",
    }

    if not (aoi is None) and aoi_c is None:
        aoi_c = dict()
        areas = data[aoi].unique()
        for area in areas:
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
            aoi_c[area] = f"rgb({color[0]}, {color[1]}, {color[2]})"

    if add_regression:
        mask = _select_regressions(dX, dY, rule, deviation)
        reg_ind_x = dX[mask].index
        reg_ind = []
        for i in reg_ind_x:
            if i != 0:
                reg_ind.append(i - 1)
                reg_ind.append(i)
        data["is_reg"] = [1 if z in reg_ind else 0 for z in data.index]
        edges = []
        first_reg = True
        first_sac = True
        for i in range(1, len(data)):
            if data.loc[i - 1, "is_reg"] == data.loc[i, "is_reg"] == 1:
                edges.append(
                    {
                        "x": [data.loc[i - 1, x], data.loc[i, x]],
                        "y": [data.loc[i - 1, y], data.loc[i, y]],
                        "mode": "lines",
                        "line": dict(color=regression_color, width=path_width),
                        "name": "regressions",
                        "showlegend": first_reg,
                    }
                )
                first_reg = False
            else:
                edges.append(
                    {
                        "x": [data.loc[i - 1, x], data.loc[i, x]],
                        "y": [data.loc[i - 1, y], data.loc[i, y]],
                        "mode": "lines",
                        "line": dict(color=path_color, width=path_width),
                        "name": "saccades",
                        "showlegend": first_sac,
                    }
                )
                first_sac = False
        fig_dict["data"].extend(edges)
    else:
        fig_dict["data"].append(edges)

    if not (aoi is None):
        areas = data[aoi].unique()
        nodes = []
        for area in areas:
            annotate = []
            data_area = data[data[aoi] == area]
            indexes_area = data_area.index
            for i in range(len(indexes_area)):
                row = data_area.loc[
                    indexes_area[i], data_area.columns.intersection(meta_data)
                ].values
                comments = []
                for j in range(len(meta_data)):
                    comments.append(f"{meta_data[j]}: {row[j]}")
                annotate.append("<br>".join(comments))
            nodes.append(
                {
                    "x": data_area[x].values,
                    "y": data_area[y].values,
                    "mode": "markers",
                    "marker": dict(color=aoi_c[area], size=points_width),
                    "name": area,
                    "text": annotate,
                }
            )
        fig_dict["data"].extend(nodes)
    else:
        annotate = []
        for i in range(len(indexes)):
            row = data.loc[indexes[i], data.columns.intersection(meta_data)].values
            comments = []
            for j in range(len(meta_data)):
                comments.append(f"{meta_data[j]}: {row[j]}")
            annotate.append("<br>".join(comments))
        nodes = {
            "x": X,
            "y": Y,
            "mode": "markers",
            "marker": dict(color=points_color, size=points_width),
            "name": "fixations",
            "text": annotate,
        }
        fig_dict["data"].append(nodes)

    fig_dict["data"].append(
        {
            "x": [X[0]],
            "y": [Y[0]],
            "mode": "markers",
            "marker": dict(color="red"),
            "name": "tracker",
        }
    )

    for i in range(len(indexes)):
        frame = {"data": [], "name": str(i)}
        if add_regression:
            frame["data"].extend(edges)
        else:
            frame["data"].append(edges)
        if not (aoi is None):
            frame["data"].extend(nodes)
        else:
            frame["data"].append(nodes)
        frame["data"].append(
            {
                "x": [X[i]],
                "y": [Y[i]],
                "mode": "markers",
                "marker": dict(color="red"),
                "name": "tracker",
            }
        )
        fig_dict["frames"].append(frame)
        slider_step = {
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 300, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                },
            ],
            "label": str(i),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)
    fig.show()
