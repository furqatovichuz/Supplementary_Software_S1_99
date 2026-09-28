"""Cell-centred finite-volume grid and implicit transport operators (Section 2.5)."""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


@dataclass
class Grid:
    Lx: float
    Ly: float
    Lz: float
    Nx: int
    Ny: int
    Nz: int

    def __post_init__(self):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.V = self.dx * self.dy * self.dz
        self.n = self.Nx * self.Ny * self.Nz
        self.shape = (self.Nx, self.Ny, self.Nz)
        self.xc = (np.arange(self.Nx) + 0.5) * self.dx
        self.yc = (np.arange(self.Ny) + 0.5) * self.dy
        self.zc = (np.arange(self.Nz) + 0.5) * self.dz

    def idx(self, i, j, k):
        return (i * self.Ny + j) * self.Nz + k

    def area(self, axis):
        return (self.dy * self.dz, self.dx * self.dz, self.dx * self.dy)[axis]

    def spacing(self, axis):
        return (self.dx, self.dy, self.dz)[axis]


def _face_cells(g, face):
    """Flat indices of the cells adjacent to a boundary face."""
    I, J, K = np.meshgrid(np.arange(g.Nx), np.arange(g.Ny), np.arange(g.Nz), indexing="ij")
    sel = {
        "xmin": I == 0, "xmax": I == g.Nx - 1,
        "ymin": J == 0, "ymax": J == g.Ny - 1,
        "zmin": K == 0, "zmax": K == g.Nz - 1,
    }[face]
    return g.idx(I[sel], J[sel], K[sel])


def build_operator(g, capacity, dt, diff, robin=None, adv=None):
    """Assemble the backward-Euler matrix for

        capacity * dphi/dt + adv_coef * v_z dphi/dz = div(diff grad phi) + ...

    on a uniform grid, per unit cell volume.

    robin : dict face -> transfer coefficient (flux = coef*(phi - phi_amb)), applied through a
            film-plus-half-cell resistance.  Faces not listed are zero-flux.
    adv   : (adv_coef, v_z) upward Darcy velocity, first-order upwind, inflow at z = 0.

    Returns (A, b_amb, b_in): matrix A and vectors so that the right-hand side is
        capacity/dt * phi_old + source + b_amb * phi_amb + b_in * phi_inlet.
    """
    robin = robin or {}
    n = g.n
    rows, cols, vals = [], [], []
    diag = np.full(n, capacity / dt)
    b_amb = np.zeros(n)
    b_in = np.zeros(n)
    I, J, K = np.meshgrid(np.arange(g.Nx), np.arange(g.Ny), np.arange(g.Nz), indexing="ij")
    P = g.idx(I, J, K)

    # interior diffusion faces
    for axis, Nn in enumerate((g.Nx, g.Ny, g.Nz)):
        d = g.spacing(axis)
        coef = diff * g.area(axis) / d / g.V
        if Nn < 2 or coef == 0.0:
            continue
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[axis] = slice(0, Nn - 1)
        sl_hi[axis] = slice(1, Nn)
        a = P[tuple(sl_lo)].ravel()
        b = P[tuple(sl_hi)].ravel()
        np.add.at(diag, a, coef)
        np.add.at(diag, b, coef)
        rows += [a, b]
        cols += [b, a]
        vals += [np.full(a.size, -coef), np.full(b.size, -coef)]

    # Robin boundaries: film + half-cell resistance
    for face, htc in robin.items():
        if htc is None or htc == 0.0:
            continue
        axis = "xyz".index(face[0])
        half = 0.5 * g.spacing(axis)
        G = 1.0 / (1.0 / htc + (half / diff if diff > 0 else np.inf))
        cells = _face_cells(g, face)
        coef = G * g.area(axis) / g.V
        np.add.at(diag, cells, coef)
        np.add.at(b_amb, cells, coef)

    # upwind Darcy advection (v_z >= 0, bottom inlet, top outlet)
    if adv is not None:
        ac, vz = adv
        if vz != 0.0:
            F = ac * vz * g.area(2) / g.V
            diag += F  # outflow through each cell's top face
            lower = P[:, :, 1:].ravel()
            upper_of = P[:, :, :-1].ravel()
            rows.append(lower)
            cols.append(upper_of)
            vals.append(np.full(lower.size, -F))
            b_in[_face_cells(g, "zmin")] += F

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return A, b_amb, b_in


class ImplicitOperator:
    """Constant backward-Euler operator factorized once by sparse LU."""

    def __init__(self, g, capacity, dt, diff, robin=None, adv=None):
        self.capacity, self.dt = capacity, dt
        self.A, self.b_amb, self.b_in = build_operator(g, capacity, dt, diff, robin, adv)
        self.lu = spla.splu(self.A)

    def step(self, phi, source=0.0, phi_amb=0.0, phi_in=0.0):
        rhs = self.capacity / self.dt * phi + source + self.b_amb * phi_amb + self.b_in * phi_in
        return self.lu.solve(rhs)
