"""Numerical verification checks of Section 2.6."""
import numpy as np

from .darcy import solve_pressure
from .operators import Grid, ImplicitOperator
from .solver import Simulation


def check_initial_conditions(p):
    s = Simulation(p, ventilated=True)
    err = max(float(np.max(np.abs(getattr(s, k) - p[k + "0"]))) for k in "TMWN")
    return {"max_abs_error": err, "passed": err == 0.0}


def check_closed_box(p, hours=1.0, tol=2e-9):
    """Insulated closed box: exchange on; boundaries, ventilation and biology off."""
    s = Simulation(p, ventilated=False, biology=False, boundaries=False)
    w0, e0 = s.total_water(), s.total_energy()
    for _ in range(int(round(hours * 3600 / s.dt))):
        s.step()
    ew = abs(s.total_water() - w0) / abs(w0)
    ee = abs(s.total_energy() - e0) / abs(e0)
    return {"hours": hours, "dt_s": s.dt, "water_rel_error": ew, "energy_rel_error": ee,
            "tolerance": tol, "passed": ew < tol and ee < tol,
            "W_mean_final": float(s.W.mean()), "M_mean_final": float(s.M.mean())}


def check_analytical_diffusion(p, cells=(9, 17, 33), amplitude=5.0, hours=1.0, dt=10.0):
    """Neumann cosine-mode heat-diffusion benchmark (zero flux on all faces)."""
    alpha = p["lambda_eff"] / (p["rho"] * p["c"])
    L = p["Lx"]
    t_end = hours * 3600.0
    errors = []
    for n in cells:
        g = Grid(L, p["Ly"], p["Lz"], n, 1, 1)
        op = ImplicitOperator(g, p["rho"] * p["c"], dt, p["lambda_eff"])
        T = p["T0"] + amplitude * np.cos(np.pi * g.xc / L)
        for _ in range(int(round(t_end / dt))):
            T = op.step(T)
        exact = p["T0"] + amplitude * np.cos(np.pi * g.xc / L) * np.exp(-alpha * (np.pi / L) ** 2 * t_end)
        errors.append(float(np.sqrt(np.mean((T - exact) ** 2))))
    order_log2 = [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(cells) - 1)]
    order_h = [float(np.log(errors[i] / errors[i + 1]) / np.log(cells[i + 1] / cells[i]))
               for i in range(len(cells) - 1)]
    return {"cells": list(cells), "dt_s": dt, "L2_errors_degC": errors,
            "observed_order_log2": order_log2, "observed_order_h_ratio": order_h,
            "passed": min(order_h) > 1.8}


def check_darcy(p):
    g = Grid(p["Lx"], p["Ly"], p["Lz"], int(p["Nx"]), int(p["Ny"]), int(p["Nz"]))
    _, vz, p_fan = solve_pressure(g, p, p["v_in"])
    err = float(np.max(np.abs(vz - p["v_in"])) / p["v_in"])
    return {"p_fan_Pa": p_fan, "gauge_Pa": p_fan - p["p_atm"], "velocity_rel_error": err,
            "passed": err < 1e-10}


def run_all(p):
    return {
        "initial_conditions": check_initial_conditions(p),
        "closed_box_conservation": check_closed_box(p),
        "analytical_heat_diffusion": check_analytical_diffusion(p),
        "darcy_pressure_drop": check_darcy(p),
    }
