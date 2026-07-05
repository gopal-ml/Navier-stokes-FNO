import torch


def ddx(f, dx):
    """
    Central difference in x-direction.

    Parameters
    ----------
    f : (Nx, Ny, Nz)

    Returns
    -------
    dfdx : (Nx, Ny, Nz)
    """

    dfdx = torch.zeros_like(f)

    dfdx[1:-1, :, :] = (
        f[2:, :, :]
        - f[:-2, :, :]
    ) / (2 * dx)

    return dfdx


def ddy(f, dy):
    """
    Central difference in y-direction.
    """

    dfdy = torch.zeros_like(f)

    dfdy[:, 1:-1, :] = (
        f[:, 2:, :]
        - f[:, :-2, :]
    ) / (2 * dy)

    return dfdy


def ddz(f, dz):
    """
    Central difference in z-direction.
    """

    dfdz = torch.zeros_like(f)

    dfdz[:, :, 1:-1] = (
        f[:, :, 2:]
        - f[:, :, :-2]
    ) / (2 * dz)

    return dfdz


def laplacian(f, dx, dy, dz):
    """
    3D Laplacian.
    """

    lap = torch.zeros_like(f)

    lap[1:-1,1:-1,1:-1] = (

        (
            f[2:,1:-1,1:-1]
            - 2*f[1:-1,1:-1,1:-1]
            + f[:-2,1:-1,1:-1]
        ) / dx**2

        +

        (
            f[1:-1,2:,1:-1]
            - 2*f[1:-1,1:-1,1:-1]
            + f[1:-1,:-2,1:-1]
        ) / dy**2

        +

        (
            f[1:-1,1:-1,2:]
            - 2*f[1:-1,1:-1,1:-1]
            + f[1:-1,1:-1,:-2]
        ) / dz**2
    )

    return lap
