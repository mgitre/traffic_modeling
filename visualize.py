"""
visualize.py
============
Side-by-side animation: real data vs model simulation.

Model (same as experiment_data_pos.py):

    a = alpha * (v_next - v) / h^k
      + beta * ( v_ideal * (tanh(h-ds) + tanh(L+ds)) / (1+tanh(L+ds)) - v )

Layout (2 cols x 2 rows):
  [ring: real]        [ring: model]
  [pos/time: real]    [pos/time: model]

Usage
-----
  python visualize.py --data case2.data [options]

Key options
-----------
  --alpha, --k, --beta, --v_ideal, --ds   model parameters
  --t_end     seconds to animate           (default 100)
  --fps       animation frame rate         (default 30)
  --save      save to .mp4 or .gif instead of showing live
"""

import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

# ── Constants (must match experiment_data_pos.py) ────────────────────────────
CIRCUIT_LEN = 230.0
CAR_LENGTH  = 3.5

# ════════════════════════════════════════════════════════════════════════════
#  Data loading  (identical to experiment_data_pos.py)
# ════════════════════════════════════════════════════════════════════════════

def parse_data(filepath: str) -> list[np.ndarray]:
    """Return list of arrays per car, columns = [position, time]."""
    cars, current = [], []
    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                if current:
                    cars.append(np.array(current))
                    current = []
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    pos, t = float(parts[0]), float(parts[1])
                    current.append((pos, t))
                except ValueError:
                    pass
    if current:
        cars.append(np.array(current))
    return cars


def unwrap_positions(pos: np.ndarray, L: float = CIRCUIT_LEN) -> np.ndarray:
    out = pos.copy()
    for i in range(1, len(out)):
        diff = out[i] - out[i - 1]
        if diff < -L / 2:
            out[i:] += L
        elif diff > L / 2:
            out[i:] -= L
    return out


def fit_spline(times: np.ndarray, positions: np.ndarray,
               s_factor: float = 0.3) -> UnivariateSpline:
    order = np.argsort(times)
    t = times[order]
    p = unwrap_positions(positions[order])
    return UnivariateSpline(t, p, s=s_factor * len(t), k=4)


def build_interpolants(splines: list, t_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate all splines on t_eval and wrap to ring.
    Returns (N_CARS, T) position array.
    """
    return np.array([sp(t_eval) % CIRCUIT_LEN for sp in splines])


# ════════════════════════════════════════════════════════════════════════════
#  Model  (vectorised, identical to experiment_data_pos.py)
# ════════════════════════════════════════════════════════════════════════════

def make_rhs(alpha: float, k: float, beta: float,
             v_ideal: float, ds: float):
    """
    Vectorised RHS for the car-following ODE.

    State layout:  [x0, v0, x1, v1, …, x_{N-1}, v_{N-1}]
    Car i follows car (i-1) mod N (circular, ascending ring-position order).
    """
    L      = CAR_LENGTH
    denom  = 1.0 + np.tanh(L + ds)
    tanh_L = np.tanh(L + ds)

    def rhs(_t, state):
        x      = state[0::2]
        v      = state[1::2]
        x_next = np.roll(x, 1)
        v_next = np.roll(v, 1)

        h = (x_next - x) % CIRCUIT_LEN - L
        h = np.maximum(h, 1e-2)

        term1 = alpha * (v_next - v) / h ** k
        v_des = v_ideal * (np.tanh(h - ds) + tanh_L) / denom
        term2 = beta * (v_des - v)

        deriv       = np.empty_like(state)
        deriv[0::2] = v
        deriv[1::2] = term1 + term2
        return deriv

    return rhs


def run_simulation(alpha: float, k: float, beta: float,
                   v_ideal: float, ds: float,
                   t_eval: np.ndarray,
                   splines: list) -> np.ndarray:
    """
    Simulate all cars from t_eval[0] using spline ICs.

    Cars are sorted by ring position at t0 so that np.roll gives the correct
    physical leader chain — same convention as experiment_data_pos.py.

    Returns (N_CARS, T) ring-position array in original spline order.
    """
    n_cars = len(splines)
    t0     = t_eval[0]
    rhs    = make_rhs(alpha, k, beta, v_ideal, ds)

    # ── ICs from splines, sorted by descending ring position ─────────────────
    pos0 = np.array([sp(t0) % CIRCUIT_LEN for sp in splines])
    vel0 = np.array([max(float(sp.derivative(1)(t0)), 0.0) for sp in splines])

    order     = np.argsort(-pos0)   # descending: car[0] leads
    inv_order = np.argsort(order)

    ic       = np.empty(2 * n_cars)
    ic[0::2] = pos0[vel0 >= 0][order] if False else pos0[order]  # sorted pos
    ic[1::2] = vel0[order]

    print(f"  Initial positions (sorted): "
          f"{np.round(pos0[order], 1).tolist()}")
    print(f"  Initial velocities (sorted): "
          f"{np.round(vel0[order], 2).tolist()}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = solve_ivp(
            rhs,
            (t_eval[0], t_eval[-1]),
            ic,
            method="Radau",
            t_eval=t_eval,
            rtol=1e-4, atol=1e-4,
        )

    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    # Unshuffle back to original spline order
    sim_sorted = sol.y[0::2, :] % CIRCUIT_LEN   # (N_CARS, T) in sorted order
    return sim_sorted[inv_order, :]              # (N_CARS, T) in spline order


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def ring_to_xy(positions: np.ndarray):
    """Convert ring positions (m) to (x, y) on a circle."""
    radius = CIRCUIT_LEN / (2 * np.pi)
    angles = (positions / CIRCUIT_LEN) * 2 * np.pi
    return radius * np.cos(angles), radius * np.sin(angles)


def car_colors(n: int):
    cmap   = plt.cm.plasma
    colors = [cmap(i / n) for i in range(n)]
    return colors


def style_dark(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9", labelsize=7)
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.4)


def plot_ring_trajectory(ax, t_eval, positions, color, lw, alpha):
    """Draw position-over-time, lifting pen at lap crossings."""
    t_seg, x_seg = [], []
    prev = None
    for t, x in zip(t_eval, positions):
        if np.isnan(x):
            if t_seg:
                ax.plot(t_seg, x_seg, color=color, lw=lw, alpha=alpha)
                t_seg, x_seg = [], []
            prev = None
            continue
        if prev is not None and abs(x - prev) > CIRCUIT_LEN * 0.4:
            ax.plot(t_seg, x_seg, color=color, lw=lw, alpha=alpha)
            t_seg, x_seg = [], []
        t_seg.append(t)
        x_seg.append(x)
        prev = x
    if t_seg:
        ax.plot(t_seg, x_seg, color=color, lw=lw, alpha=alpha)


# ════════════════════════════════════════════════════════════════════════════
#  Animation
# ════════════════════════════════════════════════════════════════════════════

def build_animation(real_pos: np.ndarray, sim_pos: np.ndarray,
                    t_eval: np.ndarray, params: tuple,
                    fps: int = 30, save_path: str = None):
    alpha, k, beta, v_ideal, ds = params
    n_cars = real_pos.shape[0]
    radius = CIRCUIT_LEN / (2 * np.pi)
    colors = car_colors(n_cars)
    hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                  for r, g, b, _ in colors]

    t_min, t_max = t_eval[0], t_eval[-1]

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
    fig.suptitle(
        f"MSTF ring road  ·  α={alpha:.3f}  k={k:.3f}  "
        f"β={beta:.3f}  v_ideal={v_ideal:.2f} m/s  ds={ds:.2f} m",
        color="#c8cdd8", fontsize=9, y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.04, right=0.97,
        top=0.94, bottom=0.08,
        hspace=0.35, wspace=0.25,
    )

    ax_rr = fig.add_subplot(gs[0, 0], facecolor="#0d1117")   # ring  — real
    ax_rm = fig.add_subplot(gs[0, 1], facecolor="#0d1117")   # ring  — model
    ax_pr = fig.add_subplot(gs[1, 0], facecolor="#0d1117")   # pos/t — real
    ax_pm = fig.add_subplot(gs[1, 1], facecolor="#0d1117")   # pos/t — model

    # ── Ring axes ─────────────────────────────────────────────────────────────
    pad = 1.45
    theta = np.linspace(0, 2 * np.pi, 300)
    for ax, title in [(ax_rr, "real"), (ax_rm, "model")]:
        ax.set_aspect("equal")
        ax.set_xlim(-radius * pad, radius * pad)
        ax.set_ylim(-radius * pad, radius * pad)
        ax.axis("off")
        ax.set_title(title, color="#7a8499", fontsize=9, pad=4)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta),
                color="#2a2f3d", lw=1.2, ls="--", zorder=1)

    dummy = np.zeros((n_cars, 2))
    scat_real  = ax_rr.scatter(dummy[:, 0], dummy[:, 1],
                                s=55, zorder=3, c=hex_colors)
    scat_model = ax_rm.scatter(dummy[:, 0], dummy[:, 1],
                                s=55, zorder=3, c=hex_colors)

    time_text_r = ax_rr.text(0, -radius * 1.32, "", ha="center",
                              color="#555e72", fontsize=8)
    time_text_m = ax_rm.text(0, -radius * 1.32, "", ha="center",
                              color="#555e72", fontsize=8)

    # ── Position-over-time axes ───────────────────────────────────────────────
    for ax, title in [(ax_pr, "position — real"), (ax_pm, "position — model")]:
        style_dark(ax)
        ax.set_title(title, color="#7a8499", fontsize=9, pad=4)
        ax.set_xlim(t_min, t_max)
        ax.set_ylim(-2, CIRCUIT_LEN + 2)
        ax.set_xlabel("time  (s)", color="#555e72", fontsize=8)
        ax.set_ylabel("position  (m)", color="#555e72", fontsize=8)

    # Static low-alpha traces
    for i in range(n_cars):
        c   = colors[i]
        lw  = 2.5
        alp = 0.2
        plot_ring_trajectory(ax_pr, t_eval, real_pos[i], c, lw, alp)
        plot_ring_trajectory(ax_pm, t_eval, sim_pos[i],  c, lw, alp)

    cursor_r = ax_pr.axvline(x=t_min, color="#ffffff", lw=0.8, alpha=0.4)
    cursor_m = ax_pm.axvline(x=t_min, color="#ffffff", lw=0.8, alpha=0.4)

    # ── Update function ───────────────────────────────────────────────────────
    def update(frame):
        t = t_eval[frame]

        xr, yr = ring_to_xy(real_pos[:, frame])
        xm, ym = ring_to_xy(sim_pos[:, frame])
        scat_real .set_offsets(np.column_stack([xr, yr]))
        scat_model.set_offsets(np.column_stack([xm, ym]))

        label = f"t = {t:.1f} s"
        time_text_r.set_text(label)
        time_text_m.set_text(label)
        cursor_r.set_xdata([t, t])
        cursor_m.set_xdata([t, t])

        return scat_real, scat_model, time_text_r, time_text_m, cursor_r, cursor_m

    ani = FuncAnimation(fig, update, frames=len(t_eval),
                        blit=True, interval=1000 / fps)

    if save_path:
        print(f"Saving to {save_path} …")
        writer = (PillowWriter(fps=fps) if save_path.endswith(".gif")
                  else FFMpegWriter(fps=fps, bitrate=1800))
        ani.save(save_path, writer=writer, dpi=140)
        print("Done.")
    else:
        plt.show()

    return ani


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualise real vs model traffic  (experiment_data_pos model)"
    )
    parser.add_argument("--data",    required=True,
                        help="Path to case2.data")
    parser.add_argument("--alpha",   type=float, default=3.0,
                        help="Speed-matching gain")
    parser.add_argument("--k",       type=float, default=0.75,
                        help="Headway exponent")
    parser.add_argument("--beta",    type=float, default=0.0,
                        help="OV attraction gain (set 0 to disable OV term)")
    parser.add_argument("--v_ideal", type=float, default=9.0,
                        help="Free-flow speed for OV curve (m/s)")
    parser.add_argument("--ds",      type=float, default=6.0,
                        help="OV inflection-point headway (m)")
    parser.add_argument("--t_start", type=float, default=0.0,
                        help="Start time for simulation (s)")
    parser.add_argument("--t_end",   type=float, default=100.0,
                        help="End time for simulation (s)")
    parser.add_argument("--n_t",     type=int,   default=500,
                        help="Number of time steps")
    parser.add_argument("--fps",     type=int,   default=30)
    parser.add_argument("--save",    default=None,
                        help="Save to file (e.g. out.mp4 or out.gif)")
    args = parser.parse_args()

    params = (args.alpha, args.k, args.beta, args.v_ideal, args.ds)

    # ── Load and fit splines ──────────────────────────────────────────────────
    print(f"Loading {args.data} …")
    cars = parse_data(args.data)
    print(f"  Found {len(cars)} car(s).")

    print("Fitting splines …")
    splines = [fit_spline(car[:, 1], car[:, 0]) for car in cars]

    # ── Evaluation grid ───────────────────────────────────────────────────────
    # Clamp to the range actually covered by the splines
    sp_t_min = max(np.min([car[:, 1].min() for car in cars]), args.t_start)
    sp_t_max = min(np.max([car[:, 1].max() for car in cars]), args.t_end)
    t_eval   = np.linspace(sp_t_min, sp_t_max, args.n_t)

    # ── Real positions from splines ───────────────────────────────────────────
    print("Interpolating real positions …")
    real_pos = build_interpolants(splines, t_eval)   # (N_CARS, T)

    # ── Simulate ──────────────────────────────────────────────────────────────
    print(f"Running Radau simulation "
          f"(t=[{sp_t_min:.1f}, {sp_t_max:.1f}] s, {args.n_t} steps) …")
    print(f"  Parameters: alpha={args.alpha}  k={args.k}  beta={args.beta}  "
          f"v_ideal={args.v_ideal}  ds={args.ds}")
    sim_pos = run_simulation(*params, t_eval, splines)
    print("Simulation done.")

    # ── Animate ───────────────────────────────────────────────────────────────
    print("Building animation …")
    build_animation(real_pos, sim_pos, t_eval, params,
                    fps=args.fps, save_path=args.save)


if __name__ == "__main__":
    main()