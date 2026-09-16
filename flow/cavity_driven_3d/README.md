# 3D Lid-Driven Cavity: Dataset Generation and FNO Training

This project solves transient, incompressible 3D lid-driven-cavity flow and trains a Fourier Neural Operator (FNO) to predict the velocity at the next saved time step.

The fluid occupies the unit cube \([0,1]^3\). All walls are no-slip; the moving lid imposes the `x`-velocity. The solver uses Chorin's projection method: it computes a tentative velocity, solves a pressure Poisson equation, and projects the field toward incompressibility.

## Workflow

```text
viscosity values → 3D CFD simulations → raw .pt files
                                      ↓
                         consecutive snapshots → dataset.pt
                                      ↓
                  FNO3d: (u, v, w, nu, x, y, z) → (u_next, v_next, w_next)
```

The input includes velocity, viscosity, and coordinates (7 channels). The target contains the next `u`, `v`, and `w` fields (3 channels). This is a one-step surrogate model; rollouts require repeatedly feeding a prediction back as the next velocity input.

## Requirements

Run commands from the `Navier` directory so that imports such as `flow...` and `core...` resolve correctly.

```bash
cd Navier
python -m pip install torch tqdm
```

The scripts automatically use MPS (Apple Silicon), then CUDA, then CPU for training. Raw generation currently explicitly uses CPU; change `device='cpu'` in `dataset/generate_raw_data.py` if a supported accelerator is desired.

## 1. Generate raw CFD data

Edit the experiment settings at the top of `flow/cavity_driven_3d/dataset/generate_raw_data.py`:

| Setting | Purpose | Current value |
| --- | --- | --- |
| `Nx`, `Ny`, `Nz` | Number of grid points in each direction | `32` |
| `dt` | CFD time step | `1e-3` |
| `T_final` | End time | `1.5` |
| `save_every` | Save one snapshot every N solver steps | `5` |
| `pressure_iterations` | Jacobi iterations for pressure Poisson solve | `100` |
| `nu_values` | Viscosities used to create simulations | 10 values from `0.005` to `0.05` |

Then run:

```bash
python flow/cavity_driven_3d/dataset/generate_raw_data.py
```

Each run creates `flow/cavity_driven_3d/data/raw/cavity3d_nu_*.pt`. A raw file stores `u_hist`, `v_hist`, `w_hist`, `p_hist`, the coordinate vectors, and solver metadata.

For a quick smoke test, first reduce the grid, end time, and pressure iterations, for example `16³`, `T_final = 0.02`, one viscosity, and 20 pressure iterations. Restore scientifically appropriate values before producing a training dataset.

### Data-quality checks

Use `verbose=True` when constructing `CavitySolver3D` to print CFL and divergence diagnostics every 100 steps. Before trusting a run, ensure the CFL is stable and divergence stays acceptably small. Also inspect fields in the notebooks under `notebooks/` and reject simulations containing NaN or Inf values.

## 2. Build the supervised dataset

The builder turns every consecutive pair of saved snapshots into one training example. With `N` stored snapshots for a viscosity, it yields `N - 1` examples for that viscosity.

```bash
mkdir -p flow/cavity_driven_3d/data/processed
python flow/cavity_driven_3d/dataset/build_dataset.py
```

By default the builder writes `validation_dataset.pt`, while the training script reads `dataset.pt`. Choose one consistent filename before training. For the training script as currently written, change the bottom of `build_dataset.py` to:

```python
save_path="flow/cavity_driven_3d/data/processed/dataset.pt",
```

and run the builder again. Alternatively, change the dataset path in `training/train.py` to `validation_dataset.pt`.

The processed file contains:

- `X`: `[samples, 7, Nx, Ny, Nz]` — current velocity, `nu`, and `x/y/z` coordinate channels.
- `Y`: `[samples, 3, Nx, Ny, Nz]` — velocity at the next saved snapshot.
- `sample_nu`: viscosity for every sample, used to split train and validation data.

These dense tensors can be large. Do not accidentally commit `data/raw/*.pt`, `data/processed/*.pt`, or checkpoints to source control unless that is intentional.

## 3. Train the FNO

Review the configuration in `flow/cavity_driven_3d/training/train.py`, then run:

```bash
python flow/cavity_driven_3d/training/train.py
```

The supplied split trains on viscosities `0.005` through `0.040` and validates on unseen viscosities `0.045` and `0.050`. This tests interpolation/extrapolation across the physical parameter rather than randomly mixing neighboring time snapshots from the same simulation into both splits.

Current FNO configuration:

```python
modesx = modesy = modesz = 12
width = 32
n_layers = 8
batch_size = 4
epochs = 10
```

The script normalizes velocity channels, trains with `MSELoss` and `AdamW`, appends metrics to `training/logs.txt`, and saves the best validation checkpoint to `checkpoints/best_fno_8.pt`. The checkpoint includes model architecture and normalization statistics needed for inference. `checkpoints/loss_history_8.pt` stores the complete loss curves.

## Evaluation and rollout

Use `notebooks/prediction.ipynb` to load the best checkpoint and compare predicted velocity fields with CFD data. Report errors after de-normalizing outputs with the stored `Y_mean` and `Y_std`; normalized MSE alone is not physically interpretable.

For multi-step rollout, preserve the final four static channels (`nu`, `x`, `y`, `z`), replace the first three input channels by each predicted velocity field, and apply the same input normalization used during training. Measure error growth over time and monitor boundary-condition and divergence errors, since one-step training does not guarantee a stable long rollout.

## Practical improvements

- Split by entire simulation/viscosity, as this project does, rather than randomly splitting consecutive frames.
- Add a separate held-out viscosity or grid resolution for a final test set; do not tune using the validation viscosities.
- Add relative L2 velocity error, divergence error, and lid/wall boundary-condition error to validation metrics.
- Train on several lid velocities, Reynolds-number regimes, and/or initial conditions if the intended model must generalize beyond the current setup.
- Consider storing per-simulation files or streaming samples instead of stacking every 3D sample in memory for larger grids.
