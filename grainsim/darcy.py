"""Darcy pressure field, Eqs. (1)-(2) with boundary conditions (9)-(12)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def fan_pressure(p, v_in):
    """Inlet pressure giving the target superficial velocity through a homogeneous bed."""
    return p["p_atm"] + p["mu_air"] * v_in * p["Lz"] / p["kappa"]


def solve_pressure(g, p, v_in):
    """Solve div(kappa/mu grad p) = 0 with Dirichlet p at z=0 and z=Lz, zero flux on the side walls.

    Returns cell pressures (Nx,Ny,Nz), vertical face velocities (Nx,Ny,Nz+1) and p_fan.
    """
    K = p["kappa"] / p["mu_air"]
    p_fan = fan_pressure(p, v_in)
    Nx, Ny, Nz = g.shape
    n = g.n
    idx = np.arange(n).reshape(g.shape)
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    for axis, d in enumerate((g.dx, g.dy, g.dz)):
        c = K / d**2
        Nn = g.shape[axis]
        if Nn < 2:
            continue
        a = np.take(idx, np.arange(Nn - 1), axis=axis).ravel()
        bb = np.take(idx, np.arange(1, Nn), axis=axis).ravel()
        rows += [a, bb, a, bb]
        cols += [a, bb, bb, a]
        vals += [np.full(a.size, c), np.full(a.size, c), np.full(a.size, -c), np.full(a.size, -c)]
    cb = K / (0.5 * g.dz * g.dz)
    bot = idx[:, :, 0].ravel()
    top = idx[:, :, -1].ravel()
    rows += [bot, top]
    cols += [bot, top]
    vals += [np.full(bot.size, cb), np.full(top.size, cb)]
    dp = p_fan - p["p_atm"]  # solve for gauge pressure to avoid round-off on 1e5 Pa values
    b[bot] += cb * dp
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    pg = spla.spsolve(A.tocsc(), b).reshape(g.shape)
    vz = np.empty((Nx, Ny, Nz + 1))
    vz[:, :, 0] = -K * (pg[:, :, 0] - dp) / (0.5 * g.dz)
    vz[:, :, 1:-1] = -K * np.diff(pg, axis=2) / g.dz
    vz[:, :, -1] = -K * (0.0 - pg[:, :, -1]) / (0.5 * g.dz)
    return pg + p["p_atm"], vz, p_fan
