import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_grid(spacing, shape):
    """_summary_

    Parameters
    ----------
    spacing : _type_
        _description_
    shape : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    x = np.array([spacing[0] * n for n in range(shape[0])])
    y = np.array([spacing[1] * n for n in range(shape[1])])
    z = np.array([spacing[2] * n for n in range(shape[2])])

    return np.meshgrid(x, y, z)


def format_wind_field(wind_field):
    """_summary_

    Parameters
    ----------
    wind_field : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return tuple([np.copy(wind_field[..., i], order="C") for i in range(3)])


def plot_velocity_components(spacing, wind_field, surface_count=25, reshape=True):
    """Plots x, y, z components of given wind field over provided spacing.

    Parameters
    ----------
    spacing : _type_
        _description_
    wind_field : _type_
        _description_
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

    fig.update_coloraxes(colorscale="spectral_r")

    return fig


def plot_velocity_magnitude(
    spacing, wind_field, surf_count=75, reshape=True
) -> go.Figure:
    """Produces a 3D plot of the wind velocity magnitude in a specified domain. This returns a Plotly figure for use of downstream visualization.

    Parameters
    ----------
    spacing : _type_
        Spacing array that determines the number of points to be used in each dimension of the 3D field. Typically, of the form grid_dimensions (a 3x1 vector representing the dimensions of the domain) divided by the grid_levels, which determine the resolution of the wind field in each respective dimension.
    wind_field : _type_
        _description_
    surf_count : int, optional
        Number of surfaces to be used, by default 75
    reshape : bool, optional
        _description_, by default True

    Returns
    -------
    go.Figure
        Plotly Figure object to be used in visualization.
    """

    X, Y, Z = create_grid(spacing, wind_field.shape)

    formatted_wind_field = format_wind_field(wind_field) if reshape else wind_field

    wind_magnitude = np.sqrt(
        formatted_wind_field[0] ** 2
        + formatted_wind_field[1] ** 2
        + formatted_wind_field[2] ** 2
    )

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=wind_magnitude.flatten(),
            surface_count=surf_count,
            opacity=0.5,
            colorscale="spectral_r",
            colorbar={"title": "|U(x)|"},
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

    return fig
