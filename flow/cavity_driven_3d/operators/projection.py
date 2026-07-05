import torch

from flow.cavity_driven_3d.operators.derivatives import laplacian
from flow.cavity_driven_3d.operators.vector_calculus import (
    convection,
    divergence,
    gradient,
)


# ============================================================
# Tentative Velocity
# ============================================================

def tentative_velocity(
    u,
    v,
    w,
    nu,
    dt,
    dx,
    dy,
    dz,
):
    """
    Computes the intermediate velocity field

        u*

    without the pressure gradient.

    Returns
    -------
    u_star, v_star, w_star
    """

    u_star = (
        u
        - dt * convection(u, u, v, w, dx, dy, dz)
        + dt * nu * laplacian(u, dx, dy, dz)
    )

    v_star = (
        v
        - dt * convection(v, u, v, w, dx, dy, dz)
        + dt * nu * laplacian(v, dx, dy, dz)
    )

    w_star = (
        w
        - dt * convection(w, u, v, w, dx, dy, dz)
        + dt * nu * laplacian(w, dx, dy, dz)
    )

    return (
        u_star,
        v_star,
        w_star,
    )


# ============================================================
# Pressure RHS
# ============================================================

def pressure_rhs(
    u_star,
    v_star,
    w_star,
    rho,
    dt,
    dx,
    dy,
    dz,
):
    """
    Computes

        rhs = (rho/dt) div(u*)

    for the pressure Poisson equation.
    """

    return (
        rho / dt
    ) * divergence(
        u_star,
        v_star,
        w_star,
        dx,
        dy,
        dz,
    )


# ============================================================
# Projection Step
# ============================================================

def projection_step(
    u_star,
    v_star,
    w_star,
    p,
    rho,
    dt,
    dx,
    dy,
    dz,
):
    """
    Pressure correction

        u = u* - dt/rho grad(p)

    Returns
    -------
    u,v,w
    """

    dpdx, dpdy, dpdz = gradient(
        p,
        dx,
        dy,
        dz,
    )

    u = u_star - dt / rho * dpdx
    v = v_star - dt / rho * dpdy
    w = w_star - dt / rho * dpdz

    return (
        u,
        v,
        w,
    )
