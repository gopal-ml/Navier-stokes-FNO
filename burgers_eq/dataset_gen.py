import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'using: {device}')

# ============================================================
# Generate random smooth initial conditions
# ============================================================

def generate_initial(batch_size, N=1024, K=32, device="cuda"):

    a_hat = torch.zeros(
        batch_size,
        N,
        dtype=torch.complex64,
        device=device
    )

    for k in range(1, K + 1):

        amp = (
            torch.randn(batch_size, device=device)
            + 1j * torch.randn(batch_size, device=device)
        ) / (k**2)

        a_hat[:, k] = amp
        a_hat[:, N-k] = torch.conj(amp)

    a = torch.fft.ifft(a_hat * N, dim=-1).real

    return a


# ============================================================
# Burgers pseudo-spectral RK4 solver
# ============================================================

def solve_burgers_batch(
    a,
    T=1.0,
    nu=1e-3,
    dt=1e-5
):

    batch_size, N = a.shape

    k = (
        torch.fft.fftfreq(
            N,
            d=1.0/N,
            device=a.device
        )
        * 2 * torch.pi
    )

    k = k[None, :]          # shape (1,N)

    u_hat = torch.fft.fft(a, dim=-1)

    def rhs(uh):

        u = torch.fft.ifft(uh, dim=-1).real

        ux = torch.fft.ifft(
            1j * k * uh,
            dim=-1
        ).real

        nonlinear = torch.fft.fft(
            -u * ux,
            dim=-1
        )
        # ===== 2/3 dealiasing =====
        cutoff = N // 3
        nonlinear[:, cutoff:-cutoff] = 0
        # ==========================

        diffusion = -nu * (k**2) * uh

        return nonlinear + diffusion

    n_steps = int(T / dt)

    for _ in range(n_steps):

        k1 = rhs(u_hat)

        k2 = rhs(
            u_hat
            + 0.5 * dt * k1
        )

        k3 = rhs(
            u_hat
            + 0.5 * dt * k2
        )

        k4 = rhs(
            u_hat
            + dt * k3
        )

        u_hat = (
            u_hat
            + dt/6
            * (
                k1
                + 2*k2
                + 2*k3
                + k4
            )
        )

    return torch.fft.ifft(
        u_hat,
        dim=-1
    ).real

# a = generate_initial(
#     batch_size=1,
#     N=1024,
#     K=32,
#     device=device
# )

# u = solve_burgers_batch(
#     a,
#     T=1.0,
#     nu=1e-3,
#     dt=1e-4
# )

# print(torch.isnan(u).any())
# print(torch.isinf(u).any())
# print(u.min(), u.max())

# ============================================================
# Dataset generation
# ============================================================

N_train = 200
N_test = 50

total_samples = N_train + N_test

reference_resolution = 1024
target_resolution = 256

batch_size = 16

factor = (
    reference_resolution
    // target_resolution
)

all_a = []
all_u = []

for start in tqdm(
    range(
        0,
        total_samples,
        batch_size
    )
):

    current_batch = min(
        batch_size,
        total_samples - start
    )

    a_hi = generate_initial(
        current_batch,
        N=reference_resolution,
        K=32,
        device=device
    )

    u_hi = solve_burgers_batch(
        a_hi,
        T=0.1,
        nu=1e-3,
        dt=1e-5
    )

    if torch.isnan(u_hi).any():
        raise RuntimeError("NaNs detected in solver output")

    a_ds = a_hi[:, ::factor]
    u_ds = u_hi[:, ::factor]

    all_a.append(
        a_ds.cpu()
    )

    all_u.append(
        u_ds.cpu()
    )

dataset_a = torch.cat(all_a)
dataset_u = torch.cat(all_u)

torch.save(
    {
        "a": dataset_a,
        "u": dataset_u
    },
    "burgers_dataset.pt"
)

print(dataset_a.shape)
print(dataset_u.shape)
