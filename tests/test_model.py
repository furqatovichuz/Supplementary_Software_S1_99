"""Unit tests (pytest).  Run:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grainsim import Simulation, exchange_step, load_params, logistic_exact  # noqa: E402
from grainsim import closures as C  # noqa: E402
from grainsim.verification import (check_analytical_diffusion, check_closed_box,  # noqa: E402
                                   check_darcy, check_initial_conditions)

P = load_params()


def test_initial_conditions():
    assert check_initial_conditions(P)["passed"]


def test_closed_box_conservation():
    r = check_closed_box(P, hours=1.0)
    assert r["water_rel_error"] < 2e-9 and r["energy_rel_error"] < 2e-9


def test_analytical_second_order():
    r = check_analytical_diffusion(P)
    assert min(r["observed_order_h_ratio"]) > 1.9


def test_darcy_fan_pressure():
    r = check_darcy(P)
    assert abs(r["p_fan_Pa"] - 101527.72) < 1e-6
    assert r["velocity_rel_error"] < 1e-10


def test_exchange_conserves_water_and_energy():
    rng = np.random.default_rng(0)
    T = rng.uniform(5, 40, 500)
    M = rng.uniform(0.10, 0.20, 500)
    W = rng.uniform(0.003, 0.03, 500)
    Tn, Mn, Wn = exchange_step(T, M, W, 150.0, P)
    water0 = P["rho"] * M + P["rho_air"] * W
    water1 = P["rho"] * Mn + P["rho_air"] * Wn
    en0 = P["rho"] * P["c"] * T + P["L"] * P["rho_air"] * W
    en1 = P["rho"] * P["c"] * Tn + P["L"] * P["rho_air"] * Wn
    assert np.max(np.abs(water1 - water0) / water0) < 1e-14
    assert np.max(np.abs(en1 - en0) / np.abs(en0)) < 1e-13
    # implicit residual satisfied
    res = P["rho_air"] * (Wn - W) / 150.0 - P["k_c_int"] * P["a_v"] * (Mn - C.M_eq(Tn, Wn, P))
    assert np.max(np.abs(res)) < 1e-12


def test_logistic_exact_and_positive():
    N0 = np.array([0.0, 1.0, 50.0, 800.0])
    r, mu, K, t = 0.2 / 86400, 0.05 / 86400, 500.0, 5 * 86400.0
    N = logistic_exact(N0, r, mu, t, K)
    a, b = r - mu, r / K
    exact = a * N0 * np.exp(a * t) / (a + b * N0 * (np.exp(a * t) - 1))
    assert np.allclose(N, exact, rtol=1e-12)
    assert np.all(logistic_exact(N0, 0.0, 1.0 / 86400, t, K) >= 0)
    assert np.allclose(logistic_exact(N0, 1e-20, 1e-20, t, K), N0)


def test_short_run_finite_and_positive():
    p = load_params(Nx=5, Ny=5, Nz=7, dt=600.0)
    for vent in (False, True):
        s = Simulation(p, ventilated=vent)
        s.run(t_final_days=1.0, verbose=False)
        assert np.all(np.isfinite(s.T)) and s.M.min() > 0 and s.W.min() > 0 and s.N.min() >= 0
