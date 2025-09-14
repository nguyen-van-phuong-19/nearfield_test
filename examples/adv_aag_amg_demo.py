import numpy as np
from nearfield.heatmaps import (
    make_theta_phi_grid,
    build_steering_on_angular_slice,
)
from nearfield.metrics_aag_amg import per_point_gains
from nearfield.plotting_interactive import heatmap_theta_phi
import plotly.io as pio


SPEED_OF_LIGHT = 299_792_458.0


def make_array(layout: str, nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
    layout = layout.lower()
    if layout == "ula":
        ny = 1
    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * (dy if layout == "upa" else 0.0)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    M = nx * ny
    xyz = np.zeros((M, 3), dtype=np.float64)
    xyz[:, 0] = xv.reshape(-1)
    xyz[:, 1] = yv.reshape(-1)
    return xyz


def main():
    nx = ny = 32
    dx = dy = 0.5 * (SPEED_OF_LIGHT / 28e9)
    xyz = make_array("upa", nx, ny, dx, dy)
    fc = 28e9
    r_fixed = 8.0
    theta = np.arange(-60, 60 + 3, 3, dtype=np.float64)
    phi = np.arange(-30, 30 + 3, 3, dtype=np.float64)
    grid = make_theta_phi_grid(theta, phi)
    A = build_steering_on_angular_slice(xyz, fc, r_fixed, grid)

    # AMAG map (ideal) -> equals 0 dB if steering rows are unit-norm
    amag_db = 10.0 * np.log10(np.maximum(np.sum(np.abs(A) ** 2, axis=1), 1e-12))

    # Codebook-selected AAG: use a coarser angular grid as codebook
    theta_cb = np.arange(-60, 60 + 6, 6, dtype=np.float64)
    phi_cb = np.arange(-30, 30 + 6, 6, dtype=np.float64)
    cb_grid = make_theta_phi_grid(theta_cb, phi_cb)
    C = build_steering_on_angular_slice(xyz, fc, r_fixed, cb_grid)
    # For each a in A, select best codeword
    sims = A @ np.conjugate(C.T)  # (P,Nc)
    idx = np.argmax(np.abs(sims), axis=1)
    W = C[idx]  # (P,M)
    aag_db = per_point_gains(W, A, mode="db")
    loss_db = amag_db - aag_db

    # Plot and export
    fig1 = heatmap_theta_phi(theta, phi, amag_db.reshape(len(phi), len(theta)), "AMAG (dB)")
    fig2 = heatmap_theta_phi(theta, phi, aag_db.reshape(len(phi), len(theta)), "AAG (dB, codebook-selected)")
    fig3 = heatmap_theta_phi(theta, phi, loss_db.reshape(len(phi), len(theta)), "Loss (AMAG - AAG) dB")
    pio.write_html(fig1, file="adv_maps_amag.html", include_plotlyjs="cdn", auto_open=False)
    pio.write_html(fig2, file="adv_maps_aag.html", include_plotlyjs="cdn", auto_open=False)
    pio.write_html(fig3, file="adv_maps_loss.html", include_plotlyjs="cdn", auto_open=False)
    print("Saved adv_maps_*.html")


if __name__ == "__main__":
    main()

