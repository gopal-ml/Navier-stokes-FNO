# Tandem submerged arc welding: fluid-flow surrogate

This workflow reuses `core.FNO3d` to predict the fluid-flow fields from the paper's author-provided simulation data. It does not solve the energy equation or predict temperature.

## Flow prediction target

The primary target is the next saved velocity state:

- `u`, `v`, `w`: three velocity components
- `p` (optional): pressure, if the authors provide it and you want the model to predict it

The inputs should include the current velocity state and relevant conditioning data. Depending on the files the authors can share, that may include coordinates, the VOF fluid fraction/free-surface representation, geometry or solid mask, and process parameters. A supplied temperature field may be included as an input if you want to retain thermal forcing effects while predicting only flow. That does not make this a heat-flow model: temperature is not an output and the code does not advance it.

The paper's full CFD model couples momentum to energy and tracks a changing free surface. Omitting temperature from the flow inputs also omits thermal effects such as temperature-dependent properties or surface-tension forcing. That would be a further modeling simplification beyond simply choosing velocity as the prediction target. The free-surface fraction is separate from temperature; retain it if the author's data includes it and the target should represent the changing weld-pool boundary.

## Data needed from the authors

I could not find a public archive of the paper's simulation fields. Ask for transient 3D snapshots for all four cases, including velocity components, pressure if available, VOF fraction/free-surface data, geometry/grid coordinates, timestamps, units, and case/process labels. Ask whether temperature or other thermal quantities can also be supplied for optional conditioning. We will not generate replacement CFD data.

## Prepared-data format

Once the authors provide data, a converter for their actual export format will pack it into a PyTorch file with:

- `X`: `[samples, input_channels, Nx, Ny, Nz]`; current flow state and chosen conditioning fields
- `Y`: `[samples, output_channels, Nx, Ny, Nz]`; only `u`, `v`, `w`, and optionally `p`
- `case_id`: one case label per sample so complete welding cases stay together during validation
- `input_channels`, `output_channels`: names in tensor order
- Optional `x`, `y`, `z`: coordinate vectors

Each sample pairs one current snapshot and its conditioning channels with the next saved flow snapshot. All cases must be mapped to a consistent Cartesian grid and physical units. The converter is intentionally not written yet because the author-export format is unknown.

## Train and validate

Run from the repository root and reserve one or more entire case IDs for validation:

```bash
python -m flow.tandem_saw.training \
  --data flow/tandem_saw/data/author_fields.pt \
  --val-cases Case4 \
  --output-dir flow/tandem_saw/runs/first \
  --epochs 100
```

Normalization statistics are calculated from training cases only. Validation reports physical-unit MSE, MAE, and relative L2. The article has four welding cases, so use leave-one-case-out runs where possible; many adjacent snapshots from one case do not constitute independent validation cases.

## Visualize flow fields

```bash
python -m flow.tandem_saw.visualize \
  --data flow/tandem_saw/data/author_fields.pt \
  --checkpoint flow/tandem_saw/runs/first/best_fno3d.pt \
  --sample 0 \
  --field u \
  --output flow/tandem_saw/runs/first/u_case.png
```

The plot compares the author's target, the FNO prediction, and signed error on the XY, XZ, and YZ mid-planes. Use `u`, `v`, `w`, or `p` if pressure is included in `Y`.
