"""
Traffic Experiment Case 2 — Position-Loss Car-Following Fit
============================================================
Same model as experiment_data.py:

    a = alpha * (v_next - v) / h^k
      + beta * ( v_ideal * (tanh(h-ds) + tanh(L+ds)) / (1+tanh(L+ds)) - v )

Instead of fitting to observed accelerations we:
  1. Sample N_ANCHORS evenly-spaced "anchor" time-points from the dataset.
  2. At each anchor t0, use the real spline-derived (position, velocity) as
     initial conditions and integrate the ODE forward H_STEPS steps of dt=1/3 s.
  3. Compare predicted ring-positions to real ring-positions at t0+1…t0+H.
  4. Weight the horizon steps with a geometric decay (default 80/10/5/3/2 %)
     so that near-term accuracy dominates.
  5. Minimise total weighted position error with differential_evolution.

Performance optimisations vs the naive version
-----------------------------------------------
  A) Vectorised RHS — the per-car Python loop inside make_rhs() is replaced
     with NumPy array ops (roll, where).  Radau still drives the solve;
     we just make each RHS call ~20× faster.

  B) Precomputed spline table — all spline positions AND velocities at every
     anchor × horizon time are evaluated ONCE before optimisation and stored
     in two (N_ANCHORS, H_STEPS+1, N_CARS) float32 arrays.  rollout_loss()
     indexes into them rather than calling sp(t) on every function evaluation.

v_ideal is constrained to [8, 10] m/s — the free-flow speed the OV curve
saturates to.  No hard velocity clipping is applied during integration.

Usage:
    python experiment_data_pos.py                     # case2.data in same dir
    python experiment_data_pos.py path/to/case2.data
"""

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import UnivariateSpline
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# ── Constants ─────────────────────────────────────────────────────────────────
CIRCUIT_LEN   = 230.0
CAR_LENGTH    = 3.1
V_IDEAL_MIN   = 8.0         # m/s — search bounds for v_ideal only
V_IDEAL_MAX   = 13.0

DT          = 1.0 / 3.0    # data sample interval (s)
H_STEPS     = 6            # horizon length in steps (15 x 1/3 s = 5 s)
_k          = np.arange(1, H_STEPS + 1)
HORIZON_W   = (0.6) ** _k
HORIZON_W  /= HORIZON_W.sum()   # normalise so weights sum to 1

N_ANCHORS   = 50

# ── 1. Parse ──────────────────────────────────────────────────────────────────
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


# ── 2. Spline fitting ─────────────────────────────────────────────────────────
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
    return UnivariateSpline(t, p, s=s_factor * len(t), k=3)


# ── Sanity checks ─────────────────────────────────────────────────────────────
def sanity_check_data(cars: list[np.ndarray]) -> None:
    """Basic plausibility checks on the raw parsed data."""
    print("\nSanity checks — raw data:")
    n = len(cars)

    t_mins = [c[:, 1].min() for c in cars]
    t_maxs = [c[:, 1].max() for c in cars]
    n_pts  = [len(c) for c in cars]

    print(f"  Cars: {n}")
    print(f"  Points per car: min={min(n_pts)}  max={max(n_pts)}  "
          f"mean={np.mean(n_pts):.0f}")
    print(f"  Time range per car: "
          f"[{min(t_mins):.1f}, {max(t_maxs):.1f}] s  "
          f"(spread in t_min={max(t_mins)-min(t_mins):.2f} s, "
          f"spread in t_max={max(t_maxs)-min(t_maxs):.2f} s)")

    # All cars should cover roughly the same time window
    if max(t_mins) - min(t_mins) > 5.0:
        print("  WARNING: cars have very different start times — "
              "check parser or data format")
    if max(t_maxs) - min(t_maxs) > 5.0:
        print("  WARNING: cars have very different end times")

    # Positions should stay on the ring
    for i, c in enumerate(cars):
        pos = c[:, 0]
        if pos.min() < 0 or pos.max() > CIRCUIT_LEN:
            print(f"  WARNING: car {i} has positions outside "
                  f"[0, {CIRCUIT_LEN}] m: "
                  f"[{pos.min():.1f}, {pos.max():.1f}]")

    if min(n_pts) < 10:
        print(f"  WARNING: some cars have very few data points (<10)")

    print("  OK")


def sanity_check_splines(splines: list, t_lo: float = 10.0,
                         t_hi: float = 490.0) -> None:
    """Check spline velocities and positions are physically plausible."""
    print("\nSanity checks — splines:")
    t_probe = np.linspace(t_lo, t_hi, 500)
    all_vel, all_pos = [], []
    for sp in splines:
        all_vel.append(sp.derivative(1)(t_probe))
        all_pos.append(sp(t_probe) % CIRCUIT_LEN)

    vel_all = np.concatenate(all_vel)
    pos_all = np.concatenate(all_pos)

    print(f"  Velocity  — min={vel_all.min():.2f}  max={vel_all.max():.2f}  "
          f"mean={vel_all.mean():.2f}  m/s")
    print(f"  Position  — min={pos_all.min():.2f}  max={pos_all.max():.2f}  m  "
          f"(should be within [0, {CIRCUIT_LEN}])")

    if vel_all.max() > 20.0:
        print(f"  WARNING: spline velocities exceed 20 m/s — "
              f"possible bad unwrap or smoothing too low")
    if vel_all.min() < -0.5:
        print(f"  WARNING: spline velocities go below -0.5 m/s — "
              f"smoothing artefact or unwrap error (floored to 0 in tables)")
    if pos_all.min() < 0 or pos_all.max() > CIRCUIT_LEN:
        print(f"  WARNING: wrapped positions outside [0, {CIRCUIT_LEN}]")

    print("  OK")


def sanity_check_tables(pos_table: np.ndarray, vel_table: np.ndarray,
                        sort_order: np.ndarray) -> None:
    """Check the precomputed tables and sort order look sensible."""
    print("\nSanity checks — spline tables + sort order:")
    n_anchors, _, n_cars = pos_table.shape

    # After sorting, headways between adjacent cars should be in (0, CIRCUIT_LEN)
    # and roughly equal (CIRCUIT_LEN / n_cars on average)
    expected_h = CIRCUIT_LEN / n_cars
    sample_idx = np.linspace(0, n_anchors - 1, min(50, n_anchors), dtype=int)
    headways = []
    for a_idx in sample_idx:
        order = sort_order[a_idx]
        pos_sorted = pos_table[a_idx, 0, order].astype(float)  # descending
        for i in range(n_cars):
            gap = (pos_sorted[i - 1] - pos_sorted[i]) % CIRCUIT_LEN - CAR_LENGTH
            headways.append(gap)

    headways = np.array(headways)
    print(f"  Headways (sorted chain, {len(sample_idx)} anchors): "
          f"min={headways.min():.2f}  max={headways.max():.2f}  "
          f"mean={headways.mean():.2f}  m  "
          f"(expected ~{expected_h:.1f} m)")

    if headways.min() < 0:
        print(f"  WARNING: negative headways detected — "
              f"sort order or CAR_LENGTH may be wrong")
    if headways.max() > CIRCUIT_LEN * 0.9:
        print(f"  WARNING: very large headways — possible gap in data "
              f"or too few cars")
    if headways.mean() < expected_h * 0.5 or headways.mean() > expected_h * 2.0:
        print(f"  WARNING: mean headway far from expected {expected_h:.1f} m")

    # Velocity table: no NaNs or Infs
    if not np.all(np.isfinite(vel_table)):
        print("  WARNING: NaN or Inf in velocity table")
    if not np.all(np.isfinite(pos_table)):
        print("  WARNING: NaN or Inf in position table")

    print("  OK")


def sanity_check_fit(popt: np.ndarray, bounds: list,
                     loss: float) -> None:
    """Warn if any parameter landed at or very near a bound."""
    print("\nSanity checks — fit results:")
    names = ["alpha", "k", "beta", "v_ideal", "ds"]
    tol   = 0.01   # fraction of range considered "at the bound"
    at_bound = []
    for name, val, (lo, hi) in zip(names, popt, bounds):
        margin = tol * (hi - lo)
        if val <= lo + margin:
            at_bound.append(f"{name} at lower bound ({lo})")
        elif val >= hi - margin:
            at_bound.append(f"{name} at upper bound ({hi})")
    if at_bound:
        print(f"  WARNING: parameter(s) at boundary — optimiser may not "
              f"have converged or bounds are too tight:")
        for msg in at_bound:
            print(f"    {msg}")
    else:
        print(f"  All parameters away from bounds — good.")

    if loss >= 1e8:
        print(f"  WARNING: final loss={loss:.3e} is huge — "
              f"integration likely failed throughout")
    elif loss > 10.0:
        print(f"  WARNING: final loss={loss:.4f} m² seems high — "
              f"model may not fit the data well")
    else:
        print(f"  Final loss={loss:.6f} m²  RMSE={np.sqrt(loss):.4f} m")

    print("  OK")



# ── 3. Vectorised ODE RHS ─────────────────────────────────────────────────────
def make_rhs(alpha: float, k: float, beta: float,
             v_ideal: float, ds: float):
    """
    Return a vectorised RHS for the car-following ODE.

    State layout:  [x0, v0, x1, v1, …, x_{N-1}, v_{N-1}]
    Car i follows car (i-1) mod N (circular leader chain).

    All per-car quantities are computed as NumPy array operations —
    no Python loop over cars.
    """
    L      = CAR_LENGTH
    # Scalar constants that don't change between RHS calls
    denom  = 1.0 + np.tanh(L + ds)
    tanh_L = np.tanh(L + ds)

    def rhs(_t, state):
        # ── unpack positions and velocities ──────────────────────────────────
        x = state[0::2]          # (N,)
        v = state[1::2]          # (N,)

        # ── leader = car (i-1) mod N, i.e. np.roll by +1 ────────────────────
        x_next = np.roll(x, 1)
        v_next = np.roll(v, 1)

        # ── headway (net gap, floored) ────────────────────────────────────────
        h = (x_next - x) % CIRCUIT_LEN - L
        h = np.maximum(h, 1e-2)

        # ── Term 1: speed-matching ────────────────────────────────────────────
        term1 = alpha * (v_next - v) / h ** k

        # ── Term 2: OV attraction ─────────────────────────────────────────────
        v_des = v_ideal * (np.tanh(h - ds) + tanh_L) / denom
        term2 = beta * (v_des - v)

        a = term1 + term2

        # ── Pack derivative ───────────────────────────────────────────────────
        deriv        = np.empty_like(state)
        deriv[0::2]  = v
        deriv[1::2]  = a
        return deriv

    return rhs


# ── 4. Precompute spline lookup tables ────────────────────────────────────────
def build_spline_tables(splines: list,
                        anchor_times: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate every spline at every (anchor_time + horizon_offset) once,
    and precompute the ring-position sort order at each anchor.

    Returns
    -------
    pos_table  : float32 array  (N_ANCHORS, H_STEPS+1, N_CARS)
        Ring positions (mod CIRCUIT_LEN), in original spline order.
    vel_table  : float32 array  (N_ANCHORS, H_STEPS+1, N_CARS)
        Velocities from spline 1st derivative, in original spline order.
    sort_order : int32 array    (N_ANCHORS, N_CARS)
        For each anchor, indices that sort cars by descending ring position
        at t0.  Applying this reindexes the state so car[i] follows car[i-1],
        matching the np.roll(x, 1) leader convention in make_rhs.
    """
    n_anchors = len(anchor_times)
    n_cars    = len(splines)

    offsets   = np.arange(H_STEPS + 1) * DT
    all_times = anchor_times[:, None] + offsets[None, :]   # (N_A, H+1)

    pos_table = np.empty((n_anchors, H_STEPS + 1, n_cars), dtype=np.float32)
    vel_table = np.empty((n_anchors, H_STEPS + 1, n_cars), dtype=np.float32)

    for i, sp in enumerate(splines):
        sp_d1    = sp.derivative(1)
        t_flat   = all_times.ravel()
        pos_flat = sp(t_flat) % CIRCUIT_LEN
        vel_flat = np.maximum(sp_d1(t_flat), 0.0)   # cars don't reverse; floor artifact
        pos_table[:, :, i] = pos_flat.reshape(n_anchors, H_STEPS + 1)
        vel_table[:, :, i] = vel_flat.reshape(n_anchors, H_STEPS + 1)

    # Sort cars by descending ring position at t0 (axis-1 index 0).
    # Descending so car[0] leads and car[i] follows car[i-1], matching
    # the np.roll(x, 1) convention in make_rhs.
    sort_order = np.argsort(
        -pos_table[:, 0, :],   # (N_ANCHORS, N_CARS), negate for descending
        axis=1
    ).astype(np.int32)

    print(f"  Spline tables built: {pos_table.nbytes / 1e6:.1f} MB each "
          f"({n_anchors} anchors x {H_STEPS+1} steps x {n_cars} cars)")
    return pos_table, vel_table, sort_order


# ── 5. Forward roll-out and position loss ────────────────────────────────────
def rollout_loss(params: np.ndarray,
                 pos_table: np.ndarray,
                 vel_table: np.ndarray,
                 sort_order: np.ndarray,
                 anchor_times: np.ndarray) -> float:
    """
    Objective function for differential_evolution.

    For each anchor:
      - Reorders cars by ring position (sort_order) so the leader chain is
        physically correct before handing state to the ODE.
      - Reads ICs from the precomputed table (no spline calls).
      - Runs one Radau solve.
      - Computes the horizon-weighted ring-position MSE, unshuffling the
        predicted positions back to original car order for comparison.
    """
    alpha, k, beta, v_ideal, ds = params
    n_anchors, _, n_cars = pos_table.shape
    rhs = make_rhs(alpha, k, beta, v_ideal, ds)

    t_horizon  = np.arange(1, H_STEPS + 1) * DT
    total_loss = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for a_idx, t0 in enumerate(anchor_times):
            # ── Sort cars by ring position so leader chain is physical ─────
            order    = sort_order[a_idx]          # (N_CARS,) index array
            inv_order = np.argsort(order)         # inverse permutation

            # ── ICs in sorted order ───────────────────────────────────────
            ic       = np.empty(2 * n_cars)
            ic[0::2] = pos_table[a_idx, 0, order]
            ic[1::2] = vel_table[a_idx, 0, order]

            # ── Integrate ─────────────────────────────────────────────────
            try:
                sol = solve_ivp(
                    rhs,
                    (t0, t0 + H_STEPS * DT),
                    ic,
                    method="Radau",
                    t_eval=t0 + t_horizon,
                    rtol=1e-4, atol=1e-4,
                )
            except Exception:
                return 1e9

            if not sol.success or sol.y.shape[1] < H_STEPS:
                return 1e9

            # ── Weighted position error ────────────────────────────────────
            # Predicted positions are in sorted order; unshuffle to match
            # pos_table which is in original spline order.
            x_pred_sorted = sol.y[0::2, :] % CIRCUIT_LEN      # (N_CARS, H_STEPS)
            x_pred = x_pred_sorted[inv_order, :]               # back to spline order
            x_real = pos_table[a_idx, 1:, :].T.astype(float)  # (N_CARS, H_STEPS)

            diff = (x_pred - x_real + CIRCUIT_LEN / 2) % CIRCUIT_LEN - CIRCUIT_LEN / 2
            total_loss += np.sum(HORIZON_W[None, :] * diff ** 2)

    return total_loss / (n_anchors * n_cars)


# ── 6. Style helpers ──────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(True, color="#21262d", linewidth=0.5)


# ── 7. Trajectory plots ───────────────────────────────────────────────────────
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
        (ax_raw, "Raw trajectories  (all cars)",          "Cumulative position  (m)"),
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
    # fig.savefig("case2_pos_splines.png", dpi=150, bbox_inches="tight",
    #             facecolor=fig.get_facecolor())
    # print("Saved → case2_pos_splines.png")
    plt.show()


# ── 8. Diagnostic plots ───────────────────────────────────────────────────────
def plot_pos_fit(splines, popt, anchor_times, n_cars,
                 pos_table=None, vel_table=None, sort_order=None):
    alpha, k, beta, v_ideal, ds = popt
    rhs = make_rhs(alpha, k, beta, v_ideal, ds)
 
    LONG_H       = H_STEPS      # same horizon as training — safe with any params
    long_horizon = np.arange(1, LONG_H + 1) * DT
    colors       = cm.plasma(np.linspace(0.05, 0.95, n_cars))
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0d1117")
    ax_pos, ax_err = axes
 
    plot_anchors = np.linspace(anchor_times[0], anchor_times[-1], 20)
 
    # Accumulate all (real, pred) points per car, then scatter once per car
    real_by_car = [[] for _ in range(n_cars)]
    pred_by_car = [[] for _ in range(n_cars)]
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p_idx, t0 in enumerate(plot_anchors):
            if pos_table is not None and sort_order is not None:
                tbl_idx   = int(round(p_idx * (len(anchor_times) - 1)
                                      / max(len(plot_anchors) - 1, 1)))
                order     = sort_order[tbl_idx]
                inv_order = np.argsort(order)
                ic        = np.empty(2 * n_cars)
                ic[0::2]  = pos_table[tbl_idx, 0, order]
                ic[1::2]  = vel_table[tbl_idx, 0, order]
            else:
                pos0      = np.array([sp(t0) % CIRCUIT_LEN for sp in splines])
                vel0      = np.array([max(float(sp.derivative(1)(t0)), 0.0)
                                      for sp in splines])
                order     = np.argsort(-pos0)
                inv_order = np.argsort(order)
                ic        = np.empty(2 * n_cars)
                ic[0::2]  = pos0[order]
                ic[1::2]  = vel0[order]
 
            try:
                sol = solve_ivp(rhs, (t0, t0 + LONG_H * DT), ic,
                                method="Radau", t_eval=t0 + long_horizon,
                                max_step=DT, rtol=1e-4, atol=1e-4)
            except Exception:
                continue
            if not sol.success:
                continue
 
            x_pred_sorted = sol.y[0::2, :] % CIRCUIT_LEN   # (N_CARS, LONG_H)
            x_pred_orig   = x_pred_sorted[inv_order, :]     # back to spline order
 
            # Batch spline eval per car across all horizon steps
            for i, sp in enumerate(splines):
                x_real_steps = sp(t0 + long_horizon) % CIRCUIT_LEN  # (LONG_H,)
                real_by_car[i].extend(x_real_steps.tolist())
                pred_by_car[i].extend(x_pred_orig[i, :].tolist())
 
    # One scatter call per car
    all_real, all_pred = [], []
    for i in range(n_cars):
        xr = np.array(real_by_car[i])
        xp = np.array(pred_by_car[i])
        if len(xr):
            ax_pos.scatter(xr, xp, s=1.0, alpha=0.15,
                           color=colors[i % n_cars], rasterized=True)
            all_real.append(xr)
            all_pred.append(xp)
 
    all_real = np.concatenate(all_real) if all_real else np.array([])
    all_pred = np.concatenate(all_pred) if all_pred else np.array([])
 
    style_ax(ax_pos)
    lim = CIRCUIT_LEN
    ax_pos.plot([0, lim], [0, lim], color="#f0f0f0", lw=1.0, ls="--")
    ax_pos.set_xlabel("Real position  (m)  [ring]")
    ax_pos.set_ylabel("Predicted position  (m)  [ring]")
    ax_pos.set_title("Predicted vs Real position  (short-horizon rollout)",
                     fontsize=11, pad=8)
    ax_pos.set_xlim(0, lim)
    ax_pos.set_ylim(0, lim)
 
    style_ax(ax_err)
    errors = (all_pred - all_real + CIRCUIT_LEN/2) % CIRCUIT_LEN - CIRCUIT_LEN/2
    ax_err.hist(errors, bins=80, color="#bb86fc", alpha=0.8, edgecolor="none")
    ax_err.axvline(0, color="#f0f0f0", lw=1.0, ls="--")
    ax_err.set_xlabel("Position error  (m)  (positive = ahead of real)")
    ax_err.set_ylabel("Count")
    ax_err.set_title(f"Position error distribution   RMSE = "
                     f"{np.sqrt(np.mean(errors**2)):.3f} m", fontsize=11, pad=8)
 
    names = ["alpha", "k", "beta", "v_ideal", "ds"]
    units = ["", "", "", "m/s", "m"]
    lines = [f"{n} = {v:.4f} {u}" for n, v, u in zip(names, popt, units)]
    axes[0].text(
        0.03, 0.97, "\n".join(lines),
        transform=axes[0].transAxes, fontsize=8,
        verticalalignment="top", color="#e6edf3",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                  edgecolor="#30363d", alpha=0.9),
        fontfamily="monospace",
    )
    fig.suptitle("Position-loss car-following fit  (Case 2)",
                 fontsize=14, color="#e6edf3", y=1.02)
    plt.tight_layout()
    fig.savefig("case2_pos_fit_2.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → case2_pos_fit_2.png")
    plt.show()
 

def plot_optimal_velocity(splines, popt):
    alpha, k, beta, v_ideal, ds = popt
    n_cars = len(splines)
    t_eval = np.linspace(10, 490, 2000)
    L      = CAR_LENGTH

    all_pos = np.array([sp(t_eval) % CIRCUIT_LEN for sp in splines])
    all_vel = np.array([sp.derivative(1)(t_eval)  for sp in splines])

    h_emp, v_emp = [], []
    for j in range(len(t_eval)):
        positions = all_pos[:, j]
        for i in range(n_cars):
            gaps     = (positions - positions[i]) % CIRCUIT_LEN
            gaps[i]  = np.inf
            jj       = int(np.argmin(gaps))
            h_net    = max(gaps[jj] - L, 0.0)
            h_emp.append(h_net)
            v_emp.append(all_vel[i, j])

    h_emp  = np.array(h_emp)
    v_emp  = np.array(v_emp)
    h_line = np.linspace(0, h_emp.max() * 1.05, 400)
    denom  = 1.0 + np.tanh(L + ds)
    v_line = v_ideal * (np.tanh(h_line - ds) + np.tanh(L + ds)) / denom

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    style_ax(ax)
    ax.scatter(h_emp, v_emp, s=0.8, alpha=0.08, color="#bb86fc", rasterized=True)
    ax.plot(h_line, np.clip(v_line, 0, None),
            color="#ff7043", lw=2.2,
            label=f"Model OV  (v_ideal={v_ideal:.2f} m/s, ds={ds:.2f} m)")
    ax.set_xlabel("Headway  (m)")
    ax.set_ylabel("Speed  (m/s)")
    ax.set_title("Speed vs Headway — fitted OV curve", fontsize=13, pad=10)
    ax.legend(fontsize=9, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="#c9d1d9")
    fig.suptitle("Case 2 — Position-loss fit", fontsize=14, color="#e6edf3")
    plt.tight_layout()
    fig.savefig("case2_pos_ov.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → case2_pos_ov.png")
    plt.show()


# ── 9. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else "case2.data"
    print(f"Reading {data_file} …")
    cars = parse_data(data_file)
    print(f"Found {len(cars)} car(s).")
    if not cars:
        print("No data found — check the file path.")
        sys.exit(1)
    sanity_check_data(cars)

    SMOOTHING = 0.15

    # ── Fit splines ───────────────────────────────────────────────────────────
    print("Fitting smoothing splines …")
    splines, raw_data = [], []
    for car in cars:
        sp    = fit_spline(car[:, 1], car[:, 0], SMOOTHING)
        order = np.argsort(car[:, 1])
        splines.append(sp)
        raw_data.append((car[order, 1], unwrap_positions(car[order, 0])))

    sanity_check_splines(splines)
    plot_trajectories(splines, raw_data, SMOOTHING)

    # ── Anchor times ──────────────────────────────────────────────────────────
    t_lo         = 300
    t_hi         = 500.0 - H_STEPS * DT
    anchor_times = np.linspace(t_lo, t_hi, N_ANCHORS)

    print(f"\nPosition-loss objective: {N_ANCHORS} anchors, "
          f"horizon = {H_STEPS} steps ({H_STEPS * DT:.2f} s)")
    print(f"Horizon weights: { {i+1: round(float(w),4) for i,w in enumerate(HORIZON_W)} }")

    # ── Precompute spline lookup tables (B) ───────────────────────────────────
    print("\nPrecomputing spline tables …")
    pos_table, vel_table, sort_order = build_spline_tables(splines, anchor_times)
    sanity_check_tables(pos_table, vel_table, sort_order)

    # ── Parameter bounds ──────────────────────────────────────────────────────
    bounds = [
        (0.0,         15.0),   # alpha
        (0.2,          3.0),   # k
        (0.0,        2.0),   # beta
        (V_IDEAL_MIN, V_IDEAL_MAX),  # v_ideal — free-flow speed, bounded [8, 10] m/s
        (0.5,         12.0),   # ds
    ]

    # Sanity check
    p0    = np.array([4.7, 0.93, 0.03, 10.0, 6.4])
    loss0 = rollout_loss(p0, pos_table, vel_table, sort_order, anchor_times)
    print(f"\nInitial loss (p0={p0}): {loss0:.6f}")

    bounds_arr = np.array(bounds)
    rng = np.random.default_rng(42)
    pop_size = 8 * len(bounds)  # popsize * n_params
    population = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1], 
                            size=(pop_size, len(bounds)))
    population[0] = p0

    # ── Global optimisation ───────────────────────────────────────────────────
    print("\nRunning differential_evolution …")
    print("  popsize=8, maxiter=200, workers=-1 (parallel)")
    print("  (press Ctrl-C at any time to stop and use the best result so far)\n")

    # Track the best parameter vector explicitly in the callback so we can
    # recover it on early exit — result.x is not accessible mid-run.
    callback_state = {"best_loss": np.inf, "best_x": p0.copy(), "gen": 0}

    def de_callback(xk, convergence):
        loss = rollout_loss(xk, pos_table, vel_table, sort_order, anchor_times)
        callback_state["gen"] += 1
        if loss < callback_state["best_loss"]:
            callback_state["best_loss"] = loss
            callback_state["best_x"]    = np.array(xk)
            alpha, k, beta, v_ideal, ds = xk
            print(f"  gen {callback_state['gen']:4d} | loss={loss:.6f} | "
                  f"alpha={alpha:.3f} k={k:.3f} beta={beta:.3f} "
                  f"v_ideal={v_ideal:.3f} ds={ds:.3f}")
        return False

    interrupted = False
    try:
        result = differential_evolution(
            rollout_loss,
            bounds=bounds,
            args=(pos_table, vel_table, sort_order, anchor_times),
            strategy="best1bin",
            maxiter=200,
            popsize=8,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            workers=-1,
            callback=de_callback,
            polish=True,
            disp=False,
            init=population
        )
        popt      = result.x
        best_loss = result.fun
        converged = result.success
        message   = result.message
    except KeyboardInterrupt:
        interrupted = True
        popt      = callback_state["best_x"]
        best_loss = callback_state["best_loss"]
        converged = False
        message   = "interrupted by user"
        print(f"\n  Stopped at generation {callback_state['gen']} — "
              f"using best so far (loss={best_loss:.6f})")

    names = ["alpha", "k", "beta", "v_ideal", "ds"]
    units = ["",      "",  "",     "m/s",      "m"]

    print("\n── Position-loss model fit results ───────────────────────────────")
    print(f"  {'Parameter':<12} {'Value':>12}  {'Unit'}")
    print(f"  {'-'*34}")
    for n, val, u in zip(names, popt, units):
        print(f"  {n:<12} {val:>12.5f}  {u}")
    print(f"\n  Final loss = {best_loss:.8f}  m²  (weighted mean sq ring-pos error)")
    print(f"  RMSE       ≈ {np.sqrt(best_loss):.4f}  m")
    print(f"  Converged  = {converged}")
    print(f"  Message    : {message}")
    print("──────────────────────────────────────────────────────────────────\n")

    sanity_check_fit(popt, bounds, best_loss)
    plot_pos_fit(splines, popt, anchor_times, len(cars))
    plot_optimal_velocity(splines, popt)