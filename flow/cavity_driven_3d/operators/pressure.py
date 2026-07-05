"""
Pressure Poisson solver.

Solves

    ∇²p = rhs

using Jacobi iterations.
"""

import torch

from flow.cavity_driven_3d.operators.boundary_conditions import apply_pressure_bc


# ============================================================
# Pressure Poisson Solver
# ============================================================

def pressure_poisson(
    p,
    rhs,
    dx,
    dy,
    dz,
    nit=100,
):
    """
    Solve

        ∇²p = rhs

    Parameters
    ----------
    p : (Nx,Ny,Nz)
        Initial pressure guess

    rhs : (Nx,Ny,Nz)
        Right-hand side

    Returns
    -------
    p
    """

    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz

    denom = 2.0 * (
        dy2 * dz2
        + dx2 * dz2
        + dx2 * dy2
    )

    for _ in range(nit):

        pn = p.clone()

        p[1:-1,1:-1,1:-1] = (

            (
                pn[2:,1:-1,1:-1]
                +
                pn[:-2,1:-1,1:-1]
            ) * dy2 * dz2

            +

            (
                pn[1:-1,2:,1:-1]
                +
                pn[1:-1,:-2,1:-1]
            ) * dx2 * dz2

            +

            (
                pn[1:-1,1:-1,2:]
                +
                pn[1:-1,1:-1,:-2]
            ) * dx2 * dy2

            -

            rhs[1:-1,1:-1,1:-1]
            * dx2 * dy2 * dz2

        ) / denom

        apply_pressure_bc(p)

    return p
