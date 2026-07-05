import torch

from flow.cavity_driven_3d.operators.derivatives import (
    ddx,
    ddy,
    ddz,
)


# ============================================================
# Gradient
# ============================================================

def gradient(
    f,
    dx,
    dy,
    dz,
):
    """
    Gradient of a scalar field.

    Parameters
    ----------
    f : (Nx, Ny, Nz)

    Returns
    -------
    dfdx, dfdy, dfdz
    """

    return (
        ddx(f, dx),
        ddy(f, dy),
        ddz(f, dz),
    )


# ============================================================
# Divergence
# ============================================================

def divergence(
    u,
    v,
    w,
    dx,
    dy,
    dz,
):
    """
    Divergence of a vector field.

    ∇·u = ux + vy + wz
    """

    return (
        ddx(u, dx)
        +
        ddy(v, dy)
        +
        ddz(w, dz)
    )


# ============================================================
# Convection
# ============================================================

def convection(
    field,
    u,
    v,
    w,
    dx,
    dy,
    dz,
):
    """
    Computes the convective derivative

        (u · ∇)field

    where `field` may be any scalar component
    (e.g. u, v, w, temperature, concentration).

    Returns
    -------
    (Nx, Ny, Nz)
    """

    phix = ddx(field, dx)
    phiy = ddy(field, dy)
    phiz = ddz(field, dz)

    return (
        u * phix
        +
        v * phiy
        +
        w * phiz
    )
