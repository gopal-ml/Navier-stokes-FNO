# ============================================================
# IMPORTS
# ============================================================
import torch


# ============================================================
# Cavity Driven Chorin Projection solver
# ============================================================

# ==================================
# BUILD RHS
# ==================================

def build_up_b(
        rho,
        dt,
        dx,
        dy,
        u,
        v
    ):

    b = torch.zeros_like(u)

    du_dx = (
        u[1:-1, 2:]
        -
        u[1:-1, :-2]
    ) / (2 * dx)

    dv_dy = (
        v[2:, 1:-1]
        -
        v[:-2, 1:-1]
    ) / (2 * dy)

    du_dy = (
        u[2:, 1:-1]
        -
        u[:-2, 1:-1]
    ) / (2 * dy)

    dv_dx = (
        v[1:-1, 2:]
        -
        v[1:-1, :-2]
    ) / (2 * dx)

    b[1:-1, 1:-1] = rho * (

        (du_dx + dv_dy) / dt

        - du_dx**2

        - 2.0 * du_dy * dv_dx

        - dv_dy**2
    )

    return b

# ==================================
# PRESSURE POISSON
# ==================================

def pressure_poisson(
        p,
        dx,
        dy,
        b,
        nit=50
    ):

    for _ in range(nit):

        pn = p.clone()

        p[1:-1, 1:-1] = (
            (
                (
                    pn[1:-1, 2:]
                    +
                    pn[1:-1, :-2]
                )
                * dy**2
                +
                (
                    pn[2:, 1:-1]
                    +
                    pn[:-2, 1:-1]
                )
                * dx**2
            )
            /
            (
                2
                *
                (
                    dx**2
                    +
                    dy**2
                )
            )
            -
            (
                dx**2
                * dy**2
            )
            /
            (
                2
                *
                (
                    dx**2
                    +
                    dy**2
                )
            )
            * b[1:-1, 1:-1]
        )

        # dp/dx = 0

        p[:, -1] = p[:, -2]
        p[:, 0] = p[:, 1]

        # dp/dy = 0

        p[0, :] = p[1, :]

        # reference pressure

        p[-1, :] = 0

    return p

# ==================================
# Solver
# ==================================

def solve_cavity_driven(
        Nx = 64,
        Ny = 64, 
        nu = 0.01,
        rho = 1.0,
        T_final = 1.0,
        dt = 0.001,
        save_every = 20,
        device = None,
        verbose = False
    ):
    # ==================================
    # DEVICE
    # ==================================
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print("Using:", device)
    
    # ==================================
    # GRID 
    # ==================================
    
    x = torch.linspace(0, 1, Nx, device= device)
    y = torch.linspace(0, 1, Ny, device= device)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # ==================================
    # UNKNOWNS
    # ==================================

    u = torch.zeros(Ny, Nx, device= device)
    v = torch.zeros(Ny, Nx, device= device)
    p = torch.zeros(Ny, Nx, device= device)

    # ==================================
    # INITIAL CONDITION
    # ==================================


    # ==================================
    # BOUNDARY CONDITIONS
    # ==================================
    # top wall
    u[-1, :] = 1.0
    v[-1, :] = 0.0

    # other walls
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0

    v[0, :]  = 0
    v[:, 0]  = 0
    v[:, -1] = 0


    # ==================================
    # STORAGE
    # ==================================
    u_snapshots = []
    v_snapshots = []
    p_snapshots = []

    u_snapshots.append(u.clone())
    v_snapshots.append(v.clone())
    p_snapshots.append(p.clone())


    # ==================================
    # TIME INTEGRATION
    # ==================================

    nt = int(T_final / dt)

    if verbose:
        print(f"dt = {dt}")
        print(f"nt = {nt}")

    for step in range(nt):

        un = u.clone()
        vn = v.clone()

        b = build_up_b(
            rho,
            dt,
            dx,
            dy,
            un,
            vn
        )
        
        p = pressure_poisson(
            p,
            dx,
            dy,
            b 
        )

        # update u
        u[1:-1,1:-1] = (
            un[1:-1,1:-1]

            -
            un[1:-1,1:-1]
            *
            dt/dx
            *
            (
                un[1:-1,1:-1]
                -
                un[1:-1,:-2]
            )

            -
            vn[1:-1,1:-1]
            *
            dt/dy
            *
            (
                un[1:-1,1:-1]
                -
                un[:-2,1:-1]
            )

            -
            dt/(2*rho*dx)
            *
            (
                p[1:-1,2:]
                -
                p[1:-1,:-2]
            )

            +
            nu
            *
            (
                dt/dx**2
                *
                (
                    un[1:-1,2:]
                    -
                    2*un[1:-1,1:-1]
                    +
                    un[1:-1,:-2]
                )

                +

                dt/dy**2
                *
                (
                    un[2:,1:-1]
                    -
                    2*un[1:-1,1:-1]
                    +
                    un[:-2,1:-1]
                )
            )
        )

        # update v
        v[1:-1,1:-1] = (
            vn[1:-1,1:-1]

            -
            un[1:-1,1:-1]
            *
            dt/dx
            *
            (
                vn[1:-1,1:-1]
                -
                vn[1:-1,:-2]
            )

            -
            vn[1:-1,1:-1]
            *
            dt/dy
            *
            (
                vn[1:-1,1:-1]
                -
                vn[:-2,1:-1]
            )

            -
            dt/(2*rho*dy)
            *
            (
                p[2:,1:-1]
                -
                p[:-2,1:-1]
            )

            +
            nu
            *
            (
                dt/dx**2
                *
                (
                    vn[1:-1,2:]
                    -
                    2*vn[1:-1,1:-1]
                    +
                    vn[1:-1,:-2]
                )

                +

                dt/dy**2
                *
                (
                    vn[2:,1:-1]
                    -
                    2*vn[1:-1,1:-1]
                    +
                    vn[:-2,1:-1]
                )
            )
        )

        # boundary conditions
        u[-1,:] = 1.0
        v[-1,:] = 0.0

        u[0,:] = 0.0
        u[:,0] = 0.0
        u[:,-1] = 0.0

        v[0,:] = 0.0
        v[:,0] = 0.0
        v[:,-1] = 0.0

        # CFL calc
        max_speed = torch.sqrt(
            u**2 + v**2
        ).max()

        cfl = max_speed * dt / min(dx, dy)

        # save every 20 steps
        if step % save_every == 0:

            u_snapshots.append(
                u.cpu().clone()
            )

            v_snapshots.append(
                v.cpu().clone()
            )

            p_snapshots.append(
                p.cpu().clone()
            )

            if verbose and step % 100 == 0:
                print(
                    f"Step={step} "
                    f"CFL={cfl:.4f}"
                )
        
    return {
        "u_hist": u_snapshots,
        "v_hist": v_snapshots,
        "p_hist": p_snapshots,
        "x": x.cpu(),
        "y": y.cpu(),
        "nu": nu,
        "dt": dt
    }
