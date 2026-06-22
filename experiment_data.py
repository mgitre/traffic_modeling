"""
Traffic Experiment Case 2 — Spline Fitting + Car-Following Analysis
Reads case2.data, fits a smoothing spline to each car's trajectory, then at
N_STEPS evenly-spaced time points computes for each car:
  • headway      : gap to the NEXT car ahead (circular, on the 230 m loop)
  • v_self       : own velocity  (spline 1st derivative)
  • v_next       : velocity of the next car ahead
  • accel        : own acceleration (spline 2nd derivative)
Then plots accel vs each of those three quantities, and fits the model:

    a = alpha * (v_next - v) / h^k
      + beta * ( v_ideal * (tanh(h-ds) + tanh(L+ds)) / (1+tanh(L+ds)) - v )

Usage:
    python experiment_data.py                    # looks for case2.data in same dir
    python experiment_data.py path/to/case2.data
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

# ── 1. Parse ──────────────────────────────────────────────────────────────────
def parse_data(filepath: str) -> list[np.ndarray]:
    """Return list of arrays (one per car), columns = [position, time]."""
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

# ── 2. Spline fitting ─────────────────────────────────────────────────────────
CIRCUIT_LEN = 230.0
CAR_LENGTH  = 3.5          # metres — used in the model term

def unwrap_positions(pos: np.ndarray, L: float = CIRCUIT_LEN) -> np.ndarray:
    out = pos.copy()
    for i in range(1, len(out)):
        diff = out[i] - out[i - 1]
        if diff < -L / 2:
            out[i:] += L
        elif diff > L / 2:
            out[i:] -= L
    return out

def fit_spline(times: np.ndarray, positions: np.ndarray, s_factor: float = 0.0):
    order = np.argsort(times)
    t = times[order]
    p = unwrap_positions(positions[order])
    spline = UnivariateSpline(t, p, s=s_factor * len(t), k=4)
    return spline, t, p

# ── 3. Trajectory plot ────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5)

def plot_trajectories(splines, raw_data, s_factor):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0d1117")
    colors = cm.plasma(np.linspace(0.05, 0.95, len(splines)))
    t_fine = np.linspace(0, 500, 2000)
    ax_raw, ax_sp = axes
    for i, (spline, (t_sorted, p_unwrapped)) in enumerate(zip(splines, raw_data)):
        ax_raw.scatter(t_sorted, p_unwrapped, s=1.5, color=colors[i], alpha=0.6)
        ax_sp.plot(t_fine, spline(t_fine), lw=0.9, color=colors[i], alpha=0.85)
    for ax, title, ylabel in [
        (ax_raw, "Raw trajectories  (all cars)", "Cumulative position  (m)"),
        (ax_sp,  f"Smoothing splines  (s={s_factor}·N)", "Cumulative position  (m)"),
    ]:
        style_ax(ax)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("Time  (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 500)
    sm = cm.ScalarMappable(cmap="plasma",
                           norm=plt.Normalize(vmin=1, vmax=len(splines)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Car index", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")
    fig.suptitle("Traffic Experiment — Case 2  (MSTF dataset)",
                 fontsize=15, color="#e6edf3", y=1.01)
    plt.tight_layout()
    fig.savefig("case2_splines.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → case2_splines.png")
    plt.show()

# ── 4. Car-following quantities ───────────────────────────────────────────────
def compute_following_data(splines: list, t_eval: np.ndarray, L: float = CIRCUIT_LEN):
    n_cars = len(splines)
    n_t    = len(t_eval)
    pos_circ  = np.zeros((n_cars, n_t))
    vel       = np.zeros((n_cars, n_t))
    acc       = np.zeros((n_cars, n_t))
    for i, sp in enumerate(splines):
        pos_circ[i] = sp(t_eval)        % L
        vel[i]      = sp.derivative(1)(t_eval)
        acc[i]      = sp.derivative(2)(t_eval)
    headway = np.zeros((n_cars, n_t))
    v_next  = np.zeros((n_cars, n_t))
    for k in range(n_t):
        positions = pos_circ[:, k]
        for i in range(n_cars):
            gaps = (positions - positions[i]) % L
            gaps[i] = np.inf
            j = int(np.argmin(gaps))
            headway[i, k] = max(gaps[j] - CAR_LENGTH, 0.0)  # net gap
            v_next[i, k]  = vel[j, k]
    return headway, vel, acc, v_next

# ── 5. Scatter plots: accel vs following quantities ───────────────────────────
def plot_following(headway, vel, acc, v_next, n_cars):
    dv = v_next - vel
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")
    colors = cm.plasma(np.linspace(0.05, 0.95, n_cars))
    panels = [
        (headway, "Headway  (m)",                        "Spacing to next car"),
        (vel,     "Own velocity  (m/s)",                  "Self speed"),
        (dv,      "Speed delta  Δv = v_next − v  (m/s)", "Speed delta"),
    ]
    for ax, (x_data, xlabel, title) in zip(axes, panels):
        style_ax(ax)
        for i in range(n_cars):
            ax.scatter(x_data[i], acc[i], s=1.5, alpha=0.4, color=colors[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Acceleration  (m/s²)")
        ax.set_title(title, fontsize=12, pad=8)
    sm = cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=1, vmax=n_cars))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Car index", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c9d1d9")
    fig.suptitle("Acceleration vs car-following quantities  (Case 2)",
                 fontsize=14, color="#e6edf3", y=1.02)
    plt.tight_layout()
    fig.savefig("case2_following.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → case2_following.png")
    plt.show()

# -- 7. Model fitting -------------------------------------------------------
#
#   a = alpha * dv / h^k
#     + beta * ( v_ideal * (tanh(h-ds) + tanh(L+ds)) / (1+tanh(L+ds)) - v )
#
# ds is bounded to [1, 20] m so the tanh inflection point sits within the
# observed headway range (empirical OV shows the transition at ~5-15 m).
#
# Free parameters: alpha, k, beta, v_ideal, ds   (L = CAR_LENGTH fixed)
#
def car_following_model(X, alpha, k, beta, v_ideal, ds):
    h, dv, v = X[0], X[1], X[2]
    L = CAR_LENGTH
    h_safe = np.where(h < 1e-2, 1e-2, h)
    term1 = alpha * dv / h_safe**k
    denom = 1.0 + np.tanh(L + ds)
    v_desired = v_ideal * (np.tanh(h - ds) + np.tanh(L + ds)) / denom
    term2 = beta * (v_desired - v)
    return term1 + term2


def fit_model(headway, vel, acc, v_next):
    dv = v_next - vel

    h_flat  = headway.ravel()
    dv_flat = dv.ravel()
    a_flat  = acc.ravel()
    v_flat  = vel.ravel()

    mask = h_flat > 0.5
    h_flat  = h_flat[mask]
    dv_flat = dv_flat[mask]
    a_flat  = a_flat[mask]
    v_flat  = v_flat[mask]

    print(f"  headway range : {h_flat.min():.2f} - {h_flat.max():.2f} m  "
          f"(median {np.median(h_flat):.2f} m)")
    print(f"  speed range   : {v_flat.min():.2f} - {v_flat.max():.2f} m/s  "
          f"(median {np.median(v_flat):.2f} m/s)")

    X = np.vstack([h_flat, dv_flat, v_flat])

    # Subsample to at most 20k points -- curve_fit scales badly with N
    MAX_FIT_PTS = 20_000
    if X.shape[1] > MAX_FIT_PTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[1], MAX_FIT_PTS, replace=False)
        X_fit, a_fit = X[:, idx], a_flat[idx]
        print(f"  subsampled to {MAX_FIT_PTS} points for fitting "
              f"(from {X.shape[1]})")
    else:
        X_fit, a_fit = X, a_flat

    v0 = float(np.median(np.abs(v_flat)))
    p0     = [1.0,  1.0,  1.0,  v0,   8.0]
    bounds = ([0, 0,  0.0,  0.1,   0.5],
              [ 200, 3.0,  200, 20.0,  20.0])  # v_ideal capped at 10 m/s

    print("Fitting model ...")
    try:
        popt, pcov = curve_fit(
            car_following_model, X_fit, a_fit,
            p0=p0, bounds=bounds,
            max_nfev=20_000, ftol=1e-6, xtol=1e-6
        )
    except RuntimeError as e:
        print(f"  curve_fit did not converge: {e}")
        return None, None, None, None

    perr   = np.sqrt(np.diag(pcov))
    a_pred = car_following_model(X, *popt)
    ss_res = np.sum((a_flat - a_pred) ** 2)
    ss_tot = np.sum((a_flat - a_flat.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot

    names = ["alpha", "k", "beta", "v_ideal", "ds"]
    print("\n-- Model fit results ------------------------------------------")
    print(f"  {'Parameter':<12} {'Value':>12}  {'+-1s':>12}")
    print(f"  {'-'*38}")
    for n, val, e in zip(names, popt, perr):
        print(f"  {n:<12} {val:>12.5f}  {e:>12.5f}")
    print(f"\n  R2   = {r2:.6f}")
    print(f"  RMSE = {np.sqrt(ss_res / len(a_flat)):.6f}  m/s2")
    print("---------------------------------------------------------------\n")

    return popt, perr, r2, (h_flat, dv_flat, a_flat, a_pred)


def plot_model_fit(popt, perr, r2, fit_data, n_cars):
    h_flat, dv_flat, a_flat, a_pred = fit_data
    alpha, k, beta, v_ideal, ds = popt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1117")

    # left: predicted vs actual
    ax = axes[0]
    style_ax(ax)
    lim = max(np.abs(a_flat).max(), np.abs(a_pred).max()) * 1.05
    ax.scatter(a_flat, a_pred, s=0.8, alpha=0.15, color="#bb86fc")
    ax.plot([-lim, lim], [-lim, lim], color="#f0f0f0", lw=1.0, ls="--")
    ax.set_xlabel("Measured acceleration  (m/s2)")
    ax.set_ylabel("Predicted acceleration  (m/s2)")
    ax.set_title(f"Predicted vs Measured   R2 = {r2:.4f}", fontsize=12, pad=8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # middle: residuals vs headway
    ax = axes[1]
    style_ax(ax)
    resid = a_flat - a_pred
    ax.scatter(h_flat, resid, s=0.8, alpha=0.15, color="#03dac6")
    ax.axhline(0, color="#f0f0f0", lw=0.8, ls="--")
    ax.set_xlabel("Headway  (m)")
    ax.set_ylabel("Residual  (m/s2)")
    ax.set_title("Residuals vs Headway", fontsize=12, pad=8)

    # right: residuals vs speed delta
    ax = axes[2]
    style_ax(ax)
    ax.scatter(dv_flat, resid, s=0.8, alpha=0.15, color="#cf6679")
    ax.axhline(0, color="#f0f0f0", lw=0.8, ls="--")
    ax.set_xlabel("Speed delta  dv  (m/s)")
    ax.set_ylabel("Residual  (m/s2)")
    ax.set_title("Residuals vs Speed delta", fontsize=12, pad=8)

    # Parameter box
    names = ["alpha", "k", "beta", "v_ideal", "ds"]
    units = ["", "", "", "m/s", "m"]
    lines_txt = [f"{n} = {v:.4f} +/- {e:.4f} {u}"
                 for n, v, e, u in zip(names, popt, perr, units)]
    axes[0].text(
        0.03, 0.97, "\n".join(lines_txt),
        transform=axes[0].transAxes,
        fontsize=7.5, verticalalignment="top",
        color="#e6edf3",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                  edgecolor="#30363d", alpha=0.9),
        fontfamily="monospace",
    )

    fig.suptitle("Car-following model fit  (Case 2)",
                 fontsize=14, color="#e6edf3", y=1.02)
    plt.tight_layout()
    fig.savefig("case2_model_fit.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved -> case2_model_fit.png")
    plt.show()


# -- 8. Empirical optimal velocity function -----------------------------------
def plot_optimal_velocity(headway, vel, acc, v_next, popt):
    h_flat = headway.ravel()
    v_flat = vel.ravel()

    # Fit tanh: v = A * (tanh(B * (h - C)) + 1) / 2
    def tanh_ov(h, A, B, C):
        return A * (np.tanh(B * (h - C)) + 1) / 2

    from scipy.optimize import curve_fit as cf
    try:
        tp, _ = cf(tanh_ov, h_flat, v_flat,
                   p0=[7.0, 0.3, 10.0],
                   bounds=([0, 0, 0], [15, 5, 40]),
                   max_nfev=10_000)
        h_line = np.linspace(h_flat.min(), h_flat.max(), 300)
        v_line = tanh_ov(h_line, *tp)
        tanh_label = f"tanh fit  (A={tp[0]:.2f}, B={tp[1]:.2f}, C={tp[2]:.2f})"
    except RuntimeError:
        h_line = v_line = tanh_label = None

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    style_ax(ax)

    ax.scatter(h_flat, v_flat, s=0.8, alpha=0.1, color="#bb86fc",
               rasterized=True)

    if h_line is not None:
        ax.plot(h_line, v_line, color="#ff7043", lw=2.0, label=tanh_label)
        ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9")

    ax.set_xlabel("Headway  (m)")
    ax.set_ylabel("Speed  (m/s)")
    ax.set_title("Speed vs Headway", fontsize=13, pad=10)
    fig.suptitle("Case 2 — OV curve", fontsize=14, color="#e6edf3")
    plt.tight_layout()
    fig.savefig("case2_ov.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved -> case2_ov.png")
    plt.show()

# ── 8. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else "case2.data"
    print(f"Reading {data_file} …")
    cars = parse_data(data_file)
    print(f"Found {len(cars)} car(s).")
    if not cars:
        print("No data found — check the file path.")
        sys.exit(1)

    SMOOTHING = 0.3
    N_STEPS   = 5_000

    # Fit splines
    splines, raw_data = [], []
    for car in cars:
        sp, t_s, p_u = fit_spline(car[:, 1], car[:, 0], SMOOTHING)
        splines.append(sp)
        raw_data.append((t_s, p_u))

    # Trajectory plots
    plot_trajectories(splines, raw_data, SMOOTHING)

    # Car-following analysis
    t_eval = np.linspace(5, 495, N_STEPS)
    print(f"Computing car-following quantities at {N_STEPS} timesteps …")
    headway, vel, acc, v_next = compute_following_data(splines, t_eval)

    plot_following(headway, vel, acc, v_next, len(cars))

    # Model fitting
    popt, perr, r2, fit_data = fit_model(headway, vel, acc, v_next)
    if popt is not None:
        plot_model_fit(popt, perr, r2, fit_data, len(cars))
        plot_optimal_velocity(headway, vel, acc, v_next, popt)