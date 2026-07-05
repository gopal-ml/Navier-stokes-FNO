"""
Diagnostic operators for CFD simulations.
"""

import torch

from flow.cavity_driven_3d.operators.vector_calculus import divergence


# ============================================================
# CFL Number
# ============================================================

def compute_cfl(
    u,
    v,
    w,
    dt,
    dx,
    dy,
    dz,
):
    """
    Compute the CFL number.
    """

    speed = torch.sqrt(
        u**2 +
        v**2 +
        w**2
    )

    return speed.max() * dt / min(dx, dy, dz)


# ============================================================
# Divergence
# ============================================================

def compute_divergence(
    u,
    v,
    w,
    dx,
    dy,
    dz,
):
    """
    Compute ∇·u.
    """

    return divergence(
        u,
        v,
        w,
        dx,
        dy,
        dz,
    )


# ============================================================
# Divergence Error
# ============================================================

def divergence_error(
    u,
    v,
    w,
    dx,
    dy,
    dz,
):
    """
    Returns

        max divergence
        mean divergence
    """

    div = compute_divergence(
        u,
        v,
        w,
        dx,
        dy,
        dz,
    )

    return (
        div.abs().max(),
        div.abs().mean(),
    )


# ============================================================
# Kinetic Energy
# ============================================================

def kinetic_energy(
    u,
    v,
    w,
):
    """
    Total kinetic energy.
    """

    return 0.5 * torch.mean(
        u**2 +
        v**2 +
        w**2
    )


# ============================================================
# Velocity Magnitude
# ============================================================

def velocity_magnitude(
    u,
    v,
    w,
):
    """
    |u|
    """

    return torch.sqrt(
        u**2 +
        v**2 +
        w**2
    )
