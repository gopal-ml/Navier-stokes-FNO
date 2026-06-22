# ============================================================
# IMPORTS
# ============================================================
import torch


# ============================================================
# Poiseuille Forward Euler solver
# ============================================================

def solve_poiseuille(nu, G,
                     Ny = 501, 
                     T_final = 1.0,
                     save_every = 20, 
                     verbose = False):
    
    # ==================================
    # GRID
    # ==================================
    # Ny = 501           # grid points

    y = torch.linspace(0, 1, Ny)
    dy = y[1] - y[0]
    
    # ==================================
    # STABLE TIME STEP
    # ==================================

    dt = 0.4 * dy**2 / nu

    Nt = int(T_final / dt)

    r = nu * dt / dy**2

    if verbose:
        print(f"dt = {dt:.6e}")
        print(f"Nt = {Nt}")
        print(f"r  = {r:.4f}")


    # ==================================
    # INITIAL CONDITION
    # ==================================

    u_field = torch.zeros(Ny)

    # ==================================
    # BOUNDARY CONDITIONS
    # ==================================

    u_field[0] = 0.0
    u_field[-1] = 0.0


    # ==================================
    # STORAGE
    # ==================================

    snapshots = []

    snapshots.append(
        u_field.clone()
    )


    # ==================================
    # TIME INTEGRATION
    # ==================================

    for step in range(Nt):

        u_new = u_field.clone()

        u_new[1:-1] = (
            u_field[1:-1]
            +
            r * (
                u_field[2:]
                - 2.0 * u_field[1:-1]
                + u_field[:-2]
            )
            +
            G * dt
        )

        # boundary conditions
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u_field = u_new

        # save every 20 steps

        if step % save_every == 0:

            snapshots.append(
                u_field.clone()
            )
        
    return snapshots
