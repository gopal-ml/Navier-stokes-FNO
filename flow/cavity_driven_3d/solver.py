# ============================================================
# IMPORTS
# ============================================================

import torch

from flow.cavity_driven_3d.operators.boundary_conditions import (
    apply_cavity_velocity_bc,
)

from flow.cavity_driven_3d.operators.diagnostics import (
    compute_cfl,
    divergence_error,
)
from flow.cavity_driven_3d.operators.projection import (
    tentative_velocity,
    pressure_rhs,
    projection_step,
)

from flow.cavity_driven_3d.operators.pressure import (
    pressure_poisson,
)

# ============================================================
# 3D Lid Driven Cavity Solver
# ============================================================

class CavitySolver3D:
    """
    3D incompressible lid-driven cavity solver based on
    Chorin's projection method.

    Notes
    -----
    Velocity components

        u : x velocity
        v : y velocity
        w : z velocity

    Domain

        x ∈ [0,1]
        y ∈ [0,1]
        z ∈ [0,1]
    """

    # ========================================================
    # Constructor
    # ========================================================

    def __init__(
        self,
        Nx=32,
        Ny=32,
        Nz=32,
        nu=0.01,
        rho=1.0,
        dt=1e-3,
        T_final=1.0,
        lid_velocity=1.0,
        pressure_iterations=100,
        save_every=20,
        device='cpu',
        verbose=False,
    ):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.nu = nu
        self.rho = rho

        self.dt = dt
        self.T_final = T_final

        self.nt = int(T_final / dt)

        self.lid_velocity = lid_velocity
        self.pressure_iterations = pressure_iterations

        self.save_every = save_every

        self.verbose = verbose

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

            elif torch.cuda.is_available():
                self.device = torch.device("cuda")

            else:
                self.device = torch.device("cpu")

        else:
            self.device = device

        print(f"Using: {self.device}")

        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        self.initialize_grid()

        # ----------------------------------------------------
        # Unknowns
        # ----------------------------------------------------

        self.initialize_fields()

        # ----------------------------------------------------
        # Storage
        # ----------------------------------------------------

        self.snapshots = {
            "u": [],
            "v": [],
            "w": [],
            "p": [],
        }

        self.save_snapshot()

    # ========================================================
    # Grid
    # ========================================================

    def initialize_grid(self):

        self.x = torch.linspace(
            0.0,
            1.0,
            self.Nx,
            device=self.device,
        )

        self.y = torch.linspace(
            0.0,
            1.0,
            self.Ny,
            device=self.device,
        )

        self.z = torch.linspace(
            0.0,
            1.0,
            self.Nz,
            device=self.device,
        )

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

    # ========================================================
    # Unknowns
    # ========================================================

    def initialize_fields(self):

        self.u = torch.zeros(
            self.Nx,
            self.Ny,
            self.Nz,
            device=self.device,
        )

        self.v = torch.zeros_like(self.u)

        self.w = torch.zeros_like(self.u)

        self.p = torch.zeros_like(self.u)

        apply_cavity_velocity_bc(
            self.u,
            self.v,
            self.w,
            self.lid_velocity,
        )

    # ========================================================
    # Snapshot
    # ========================================================

    def save_snapshot(self):

        self.snapshots["u"].append(
            self.u.detach().cpu().clone()
        )

        self.snapshots["v"].append(
            self.v.detach().cpu().clone()
        )

        self.snapshots["w"].append(
            self.w.detach().cpu().clone()
        )

        self.snapshots["p"].append(
            self.p.detach().cpu().clone()
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def diagnostics(self):

        cfl = compute_cfl(
            self.u,
            self.v,
            self.w,
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        max_div, mean_div = divergence_error(
            self.u,
            self.v,
            self.w,
            self.dx,
            self.dy,
            self.dz,
        )

        return {
            "cfl": cfl.item(),
            "max_div": max_div.item(),
            "mean_div": mean_div.item(),
        }

    # ========================================================
    # One Projection Step
    # ========================================================

    def step(self):
        """
        Perform one timestep using Chorin's projection method.

        Algorithm
        ---------
        1. Compute tentative velocity.
        2. Apply velocity BCs.
        3. Assemble pressure RHS.
        4. Solve pressure Poisson equation.
        5. Project velocity.
        6. Apply velocity BCs again.
        """

        # ----------------------------------------------------
        # Step 1 : Tentative velocity
        # ----------------------------------------------------

        u_star, v_star, w_star = tentative_velocity(
            self.u,
            self.v,
            self.w,
            self.nu,
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        # ----------------------------------------------------
        # Velocity BC
        # ----------------------------------------------------

        apply_cavity_velocity_bc(
            u_star,
            v_star,
            w_star,
            self.lid_velocity,
        )

        # ----------------------------------------------------
        # Step 2 : Pressure RHS
        # ----------------------------------------------------

        rhs = pressure_rhs(
            u_star,
            v_star,
            w_star,
            self.rho,
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        # ----------------------------------------------------
        # Step 3 : Pressure solve
        # ----------------------------------------------------

        self.p = pressure_poisson(
            self.p,
            rhs,
            self.dx,
            self.dy,
            self.dz,
            self.pressure_iterations,
        )

        # ----------------------------------------------------
        # Step 4 : Projection
        # ----------------------------------------------------

        self.u, self.v, self.w = projection_step(
            u_star,
            v_star,
            w_star,
            self.p,
            self.rho,
            self.dt,
            self.dx,
            self.dy,
            self.dz,
        )

        # ----------------------------------------------------
        # Final BC
        # ----------------------------------------------------

        apply_cavity_velocity_bc(
            self.u,
            self.v,
            self.w,
            self.lid_velocity,
        )

    # ========================================================
    # Solve
    # ========================================================

    def solve(self):
        """
        Solve the transient 3D lid-driven cavity problem.

        Returns
        -------
        self
        """

        if self.verbose:

            print(f"dt = {self.dt}")
            print(f"nt = {self.nt}")

        for step in range(self.nt):

            # --------------------------------------------
            # One timestep
            # --------------------------------------------

            self.step()

            # --------------------------------------------
            # Save solution
            # --------------------------------------------

            if step % self.save_every == 0:
                self.save_snapshot()

            # --------------------------------------------
            # Diagnostics
            # --------------------------------------------

            if self.verbose:

                if step % 100 == 0:

                    info = self.diagnostics()

                    print(
                        f"Step={step:6d} | "
                        f"CFL={info['cfl']:.4f} | "
                        f"Div(max)={info['max_div']:.3e} | "
                        f"Div(mean)={info['mean_div']:.3e}"
                    )

        return self

    # ========================================================
    # Export
    # ========================================================

    def results(self):

        return {

            "u_hist": self.snapshots["u"],

            "v_hist": self.snapshots["v"],

            "w_hist": self.snapshots["w"],

            "p_hist": self.snapshots["p"],

            "x": self.x.cpu(),

            "y": self.y.cpu(),

            "z": self.z.cpu(),

            "dt": self.dt,

            "nu": self.nu,

            "rho": self.rho,

            "Nx": self.Nx,

            "Ny": self.Ny,

            "Nz": self.Nz,
        }
    
    # ========================================================
    # Save
    # ========================================================

    def save(self, filename):

        torch.save(
            self.results(),
            filename,
        )

    # ========================================================
    # Load
    # ========================================================

    @staticmethod
    def load(filename):

        return torch.load(
            filename,
            map_location="cpu",
        )
