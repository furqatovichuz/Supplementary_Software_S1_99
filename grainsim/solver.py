"""Coupled T-M-W-N solver with Darcy ventilation (Sections 2.2-2.5).

One time step (first order, operator split):
  1. biological heat Q_r + Q_ins and metabolic water S_M evaluated at the start of the step;
  2. backward-Euler transport of T, W (diffusion + upwind Darcy advection + Robin) and M
     (diffusion + Robin) with constant, once-factorized sparse LU matrices;
  3. locally conservative implicit grain-air moisture exchange in every cell
     (total water and sensible + vapour-latent energy conserved exactly, W solved by bisection);
  4. exact positivity-preserving logistic insect update with rates frozen at the updated T, M;
  5. implicit insect diffusion.
"""
import json
import time
from pathlib import Path

import numpy as np

from . import closures as C
from .darcy import solve_pressure
from .operators import FACES, Grid, ImplicitOperator

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.json"


def load_params(path=None, **overrides):
    with open(path or DEFAULT_CONFIG) as f:
        p = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    p.update(overrides)
    return p


# --------------------------------------------------------------------------- exchange
def exchange_step(T, M, W, dt, p, iters=80):
    """Backward-Euler grain-air exchange, solved cell by cell.

    rho_a (W' - W)/dt = k_c,int a_v (M' - M_eq(T', W'))
    M' = M - rho_a (W' - W) / rho                 (total water conserved)
    T' = T - L rho_a (W' - W) / (rho c)           (sensible + vapour-latent energy conserved)
    """
    rho, rho_a, L, c = p["rho"], p["rho_air"], p["L"], p["c"]
    k = p["k_c_int"] * p["a_v"]

    def residual(Wn):
        dW = Wn - W
        Mn = M - rho_a * dW / rho
        Tn = T - L * rho_a * dW / (rho * c)
        return rho_a * dW / dt - k * (Mn - C.M_eq(Tn, Wn, p))

    lo = np.zeros_like(W)
    hi = np.maximum(W, W + rho * (M - p["M_min"]) / rho_a)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = residual(mid) > 0.0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
        if np.max(hi - lo) < 1e-16:
            break
    Wn = 0.5 * (lo + hi)
    dW = Wn - W
    return T - L * rho_a * dW / (rho * c), M - rho_a * dW / rho, Wn


# --------------------------------------------------------------------------- logistic
def logistic_exact(N, r, mu, dt, K):
    """Exact solution of dN/dt = (r - mu) N - (r/K) N^2 over dt with frozen r, mu (N >= 0 preserved)."""
    a = r - mu
    b = r / K
    at = a * dt
    small = np.abs(at) < 1e-12
    phi = np.where(small, dt, np.expm1(np.where(small, 1.0, at)) / np.where(small, 1.0, a))
    return N * np.exp(at) / (1.0 + b * N * phi)


# --------------------------------------------------------------------------- simulation
class Simulation:
    def __init__(self, p, ventilated=True, biology=True, exchange=True, boundaries=True,
                 v_override=None):
        self.p = p
        g = self.g = Grid(p["Lx"], p["Ly"], p["Lz"], int(p["Nx"]), int(p["Ny"]), int(p["Nz"]))
        dt = self.dt = float(p["dt"])
        self.biology, self.do_exchange = biology, exchange
        v_in = p["v_in"] if ventilated else 0.0
        if v_override is not None:
            v_in = v_override
        self.v_in = v_in
        if v_in > 0:
            _, vz, self.p_fan = solve_pressure(g, p, v_in)
            self.darcy_velocity_error = float(np.max(np.abs(vz - v_in)) / v_in)
            vz_use = float(np.mean(vz))
        else:
            self.p_fan, self.darcy_velocity_error, vz_use = p["p_atm"], 0.0, 0.0

        side_top = [f for f in FACES if f != "zmin"]  # base is adiabatic / impermeable
        rob = (lambda coef: {f: coef for f in side_top}) if boundaries else (lambda coef: {})
        self.opT = ImplicitOperator(g, p["rho"] * p["c"], dt, p["lambda_eff"], rob(p["h"]),
                                    (p["rho_air"] * p["c_air"], vz_use))
        self.opW = ImplicitOperator(g, p["rho_air"], dt, p["rho_air"] * p["D_W"], rob(p["k_c"]),
                                    (p["rho_air"], vz_use))
        self.opM = ImplicitOperator(g, p["rho"], dt, p["rho"] * p["D_M"], rob(p["k_c"]))
        self.opN = ImplicitOperator(g, 1.0, dt, p["D_N"])  # zero flux everywhere

        n = g.n
        self.T = np.full(n, float(p["T0"]))
        self.M = np.full(n, float(p["M0"]))
        self.W = np.full(n, float(p["W0"]))
        self.N = np.full(n, float(p["N0"]))
        self.t = 0.0

    # conserved totals (per bin) for the closed-box checks
    def total_water(self):
        return float(np.sum(self.p["rho"] * self.M + self.p["rho_air"] * self.W) * self.g.V)

    def total_energy(self):
        p = self.p
        return float(np.sum(p["rho"] * p["c"] * self.T + p["L"] * p["rho_air"] * self.W) * self.g.V)

    def step(self):
        p, dt = self.p, self.dt
        T, M, W, N = self.T, self.M, self.W, self.N
        if self.biology:
            qT = C.Q_resp(T, M, p) + C.Q_ins(N, p)
            sW = C.S_M(N, p)
        else:
            qT = sW = 0.0
        T = self.opT.step(T, qT, p["T_amb"], p["T_amb"])      # T_fan = T_amb
        W = self.opW.step(W, sW, p["W_amb"], p["W_amb"])      # W_fan = W_amb
        M = self.opM.step(M, 0.0, p["M_amb"])
        if self.do_exchange:
            T, M, W = exchange_step(T, M, W, dt, p)
        if self.biology:
            N = logistic_exact(N, C.growth_rate(T, M, p), C.mortality(T, p), dt, p["K_cap"])
            N = self.opN.step(N)
        for name, arr in (("T", T), ("M", M), ("W", W), ("N", N)):
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"non-finite {name} at t={self.t}")
        for name, arr in (("M", M), ("W", W), ("N", N)):
            if np.min(arr) < 0:
                raise FloatingPointError(f"negative {name} at t={self.t}")
        self.T, self.M, self.W, self.N = T, M, W, N
        self.t += dt

    def stats(self):
        out = {"t_days": self.t / C.SEC_PER_DAY}
        for name in "TMWN":
            a = getattr(self, name)
            out[f"{name}_mean"] = float(a.mean())
            out[f"{name}_min"] = float(a.min())
            out[f"{name}_max"] = float(a.max())
        return out

    def fields(self):
        return {k: getattr(self, k).reshape(self.g.shape).copy() for k in "TMWN"}

    def run(self, t_final_days=None, diag_interval=None, snapshot_days=(), verbose=True):
        p = self.p
        t_final = (t_final_days if t_final_days is not None else p["t_final_days"]) * C.SEC_PER_DAY
        nsteps = int(round(t_final / self.dt))
        di = max(1, int(round((diag_interval or p["diagnostic_interval_s"]) / self.dt)))
        snap_steps = {int(round(d * C.SEC_PER_DAY / self.dt)): d for d in snapshot_days}
        series = [self.stats()]
        snaps = {}
        t0 = time.time()
        for s in range(1, nsteps + 1):
            self.step()
            if s % di == 0 or s == nsteps:
                series.append(self.stats())
            if s in snap_steps:
                snaps[snap_steps[s]] = self.fields()
            if verbose and s % max(1, nsteps // 10) == 0:
                st = series[-1]
                print(f"  day {self.t / 86400:6.2f}  T_mean={st['T_mean']:.3f}  T_max={st['T_max']:.3f}"
                      f"  M={st['M_mean']:.5f}  W={st['W_mean']:.5f}  N={st['N_mean']:.2f}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
        keys = series[0].keys()
        ts = {k: np.array([r[k] for r in series]) for k in keys}
        return ts, snaps
