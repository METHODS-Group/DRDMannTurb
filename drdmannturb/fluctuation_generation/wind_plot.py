"""
Common utilities and Plotly integration for visualizing generated wind field.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .cmap_util import FIELD_COLORSCALE


def create_grid(spacing: tuple[float, float, float], shape: tuple[int, int, int]) -> np.ndarray:
    """Creates a 3D grid (meshgrid) from given spacing between grid points and desired shape (which should match the
    shape of the generated wind field, for example).

    Parameters
    ----------
    spacing : tuple[float, float, float]
        Spacing array that determines the spacing of points to be used in each dimension of the 3D field. Typically, of
        the form grid_dimensions (a 3x1 vector representing the dimensions of the domain) divided by the grid_levels,
        which determine the resolution of the wind field in each respective dimension.
    shape : tuple[int, int, int]
        Number of points in each dimension.

    Returns
    -------
    np.ndarray
       np.meshgrid object consisting of points at the provided spacing and with the specified counts in each dimension.
       This is 'ij' indexed (not Cartesian!).
    """
    x = np.array([spacing[0] * n for n in range(shape[0])])
    y = np.array([spacing[1] * n for n in range(shape[1])])
    z = np.array([spacing[2] * n for n in range(shape[2])])

    return np.meshgrid(x, y, z, indexing="ij")


def format_wind_field(
    wind_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Creates a copy of the given wind-field that has a C-layout; this is a wrapper around np.copy.

    Parameters
    ----------
    wind_field : np.ndarray
        3D wind field, typically of shape :math:`(Nx, Ny, Nz, 3)` (not C-layout, to be reshaped).


    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Triple consisting of wind field values in each x, y, z directions.
    """
    return tuple([np.copy(wind_field[..., i], order="C") for i in range(3)])


def plot_velocity_components(
    spacing: tuple[float, float, float],
    wind_field: np.ndarray,
    surface_count=25,
    reshape=True,
) -> go.Figure:
    """Plots x, y, z components of given wind field over provided spacing. Note that the same spacing is used for all 3
    velocity components.

    Parameters
    ----------
    spacing : tuple[float, float, float]
        Spacing array that determines the spacing of points to be used in each dimension of the 3D field. Typically, of
        the form grid_dimensions (a 3x1 vector representing the dimensions of the domain) divided by the grid_levels,
        which determine the resolution of the wind field in each respective dimension.
    wind_field : np.ndarray
        3D wind field, typically of shape (Nx, Ny, Nz, 3) (not C-layout, to be reshaped).
    surface_count : int, optional
        Number of surfaces to be used for each velocity component, by default 25
    reshape : bool, optional
        Whether to re-format the given wind field into C-order, typically the desirable choice to match the order of
        entries of the wind field and the provided spacing, by default True

    Returns
    -------
    go.Figure
        Plotly Figure object to be used in visualization.
    """
    X, Y, Z = create_grid(spacing, wind_field.shape)

    formatted_wind_field = format_wind_field(wind_field) if reshape else wind_field

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("x Component", "y Component", "z Component"),
        specs=[[{"type": "volume"}, {"type": "volume"}, {"type": "volume"}]],
        horizontal_spacing=0.01,
    )

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=formatted_wind_field[0].flatten(),
            surface_count=surface_count,
            coloraxis="coloraxis",
            opacity=0.5,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=formatted_wind_field[1].flatten(),
            coloraxis="coloraxis",
            surface_count=surface_count,
            opacity=0.5,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=formatted_wind_field[2].flatten(),
            coloraxis="coloraxis",
            surface_count=surface_count,
            opacity=0.5,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
        scene3=dict(aspectmode="data"),
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene2=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene3=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    fig.update_coloraxes(colorscale=FIELD_COLORSCALE)

    return fig


def plot_velocity_magnitude(
    spacing: tuple[float, float, float],
    wind_field: np.ndarray,
    surf_count=75,
    reshape=True,
    transparent=False,
) -> go.Figure:
    """Produces a 3D plot of the wind velocity magnitude in a specified domain. This returns a Plotly figure for use of
    downstream visualization.

    Parameters
    ----------
    spacing : tuple[float, float, float]
        Spacing array that determines the spacing of points to be used in each dimension of the 3D field. Typically, of
        the form grid_dimensions (a 3x1 vector representing the dimensions of the domain) divided by the grid_levels,
        which determine the resolution of the wind field in each respective dimension.
    wind_field : np.ndarray
        3D wind field, typically of shape (Nx, Ny, Nz, 3) (not C-layout, to be reshaped).
    surf_count : int, optional
        Number of surfaces to be used, by default 75
    reshape : bool, optional
        Whether to re-format the given wind field into C-order, typically the desirable choice to match the order of
        entries of the wind field and the provided spacing, by default True
    transparent : bool, optional
        Whether to set the background of the plot to a transparent background, which renders the same on different
        backgrounds on which this ``Figure`` could be embedded.

    Returns
    -------
    go.Figure
        Plotly Figure object to be used in visualization.
    """
    X, Y, Z = create_grid(spacing, wind_field.shape)

    formatted_wind_field = format_wind_field(wind_field) if reshape else wind_field

    wind_magnitude = np.sqrt(formatted_wind_field[0] ** 2 + formatted_wind_field[1] ** 2 + formatted_wind_field[2] ** 2)

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=wind_magnitude.flatten(),
            surface_count=surf_count,
            opacityscale=[
                [0, 0.75],
                [0.25, 0.5],
                [0.35, 0.35],
                [0.5, 0.1],
                [0.75, 0.25],
                [0.9, 0.35],
                [1, 1],
            ],
            opacity=0.85,
            colorscale=FIELD_COLORSCALE,
            colorbar={
                "title": "|U(x)|",
            },
            showscale=True,
        ),
    )

    fig.update_layout(scene=dict(aspectmode="data"))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
    )

    fig.update_layout(title_text="Fluctuation Vector Magnitude", title_x=0.5)

    if transparent:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    return fig
