"""Empirical closures of Section 2.4 (temperatures in degC, moisture as wet-basis fraction)."""
import numpy as np

SEC_PER_DAY = 86400.0


def p_sat(T):
    """Saturation vapour pressure, Pa (Magnus form used in Section 2.4)."""
    Tc = np.maximum(T, -200.0)  # guard only for extreme trial values inside the bisection bracket
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


def relative_humidity(T, W, p):
    """RH = [p_atm W / (0.62198 + W)] / p_sat(T)."""
    return (p["p_atm"] * W / (0.62198 + W)) / p_sat(T)


def M_eq(T, W, p):
    """M_eq = clip[M_ref + a_RH (RH - RH_ref) - a_T (T - T_ref), M_min, M_max]."""
    rh = relative_humidity(T, W, p)
    m = p["M_ref"] + p["a_RH"] * (rh - p["RH_ref"]) - p["a_T"] * (T - p["T_ref"])
    return np.clip(m, p["M_min"], p["M_max"])


def Q_resp(T, M, p):
    """Grain respiration heat, W m^-3."""
    num = p["q_max"] * np.exp(-((T - p["T_opt_r"]) / p["sigma_r"]) ** 2)
    return num / (1.0 + np.exp(-(M - p["M_th"]) / p["Delta_M"]))


def Q_ins(N, p):
    """Insect metabolic heat, W m^-3."""
    return p["q_N"] * N


def S_M(N, p):
    """Insect metabolic water released to the air phase, kg m^-3 s^-1."""
    return p["s_N"] * N


def growth_rate(T, M, p):
    """Eq. (4): r(T, M), s^-1."""
    return (p["r_max"] / SEC_PER_DAY) * np.exp(
        -((T - p["T_opt"]) / p["sigma_T"]) ** 2 - ((M - p["M_opt"]) / p["sigma_M"]) ** 2
    )


def mortality(T, p):
    """mu_d(T), s^-1."""
    cold = np.maximum(p["T_c"] - T, 0.0) ** 2
    hot = np.maximum(T - p["T_h"], 0.0) ** 2
    return (p["mu_0"] + p["b_c"] * cold + p["b_h"] * hot) / SEC_PER_DAY
