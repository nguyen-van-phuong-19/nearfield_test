import numpy as np

"""Interactive Plotly helpers with lazy import.

These functions return Plotly Figure objects if plotly is installed. Import is
performed lazily to avoid a hard dependency when running tests that don't use
interactive plots.
"""


def _go():  # pragma: no cover - import helper
    import plotly.graph_objects as go  # type: ignore
    return go


def _reshape_map(theta_deg: np.ndarray, phi_deg: np.ndarray, values_db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    th = np.asarray(theta_deg).reshape(-1)
    ph = np.asarray(phi_deg).reshape(-1)
    v = np.asarray(values_db)
    n_th = th.shape[0]
    n_ph = ph.shape[0]
    if v.ndim == 1:
        if v.shape[0] != n_th * n_ph:
            raise ValueError("values_db size mismatch with theta/phi grid")
        Z = v.reshape(n_ph, n_th)
    elif v.ndim == 2:
        if v.shape != (n_ph, n_th):
            raise ValueError("values_db 2D must be (len(phi), len(theta))")
        Z = v
    else:
        raise ValueError("values_db must be 1D or 2D")
    return th, ph, Z


def heatmap_theta_phi(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    values_db: np.ndarray,
    title: str,
) -> "go.Figure":
    th, ph, Z = _reshape_map(theta_deg, phi_deg, values_db)
    go = _go()
    fig = go.Figure(
        data=go.Heatmap(
            x=th,
            y=ph,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
            hovertemplate="theta=%{x:.1f}°, phi=%{y:.1f}°<br>G=%{z:.2f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="theta (deg)",
        yaxis_title="phi (deg)",
        template="plotly_white",
    )
    return fig


def surface_theta_phi(
    theta_deg: np.ndarray,
    phi_deg: np.ndarray,
    values_db: np.ndarray,
    title: str,
) -> "go.Figure":
    th, ph, Z = _reshape_map(theta_deg, phi_deg, values_db)
    TH, PH = np.meshgrid(th, ph, indexing="xy")
    go = _go()
    fig = go.Figure(
        data=go.Surface(
            x=TH,
            y=PH,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
            hovertemplate="theta=%{x:.1f}°, phi=%{y:.1f}°<br>G=%{z:.2f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="theta (deg)",
            yaxis_title="phi (deg)",
            zaxis_title="G (dB)",
        ),
        template="plotly_white",
    )
    return fig


def line_radial_slice(
    r_vals: np.ndarray,
    values_db: np.ndarray,
    title: str,
) -> "go.Figure":
    r = np.asarray(r_vals).reshape(-1)
    v = np.asarray(values_db).reshape(-1)
    go = _go()
    fig = go.Figure(
        data=go.Scatter(
            x=r,
            y=v,
            mode="lines+markers",
            hovertemplate="r=%{x:.2f} m<br>G=%{y:.2f} dB<extra></extra>",
            name="Gain",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="r (m)",
        yaxis_title="G (dB)",
        template="plotly_white",
    )
    return fig

