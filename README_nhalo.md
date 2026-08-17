# Minihalo Subgrid Model

This directory contains the minihalo subgrid implementation used to generate spatially resolved minihalo-count fields for a large CubeP3M volume from a conditional relation calibrated on a smaller, higher-resolution simulation.

The main script is:

```text
nhalo_implimentation.py
```

It supports:

- unshifted or shifted small-box calibration catalogues;
- coarse simulation snapshots;
- fine/morphed density snapshots;
- stochastic minihalo realizations;
- optional expected halo-count fields;
- optional diagnostic plots;
- fixed random seeds for reproducibility;
- automatic reading of several CubeP3M binary density formats.

## 1. Physical idea

The model calibrates the conditional minihalo abundance

```text
P(N_halo | delta, z)
```

from a high-resolution small box and applies it to every cell of the large simulation.

The density contrast is

```text
delta = rho / <rho> - 1
```

for both calibration and target simulations.

The conditional distribution is represented by three density-dependent quantities:

```text
P0(delta)             = probability that N_halo = 0
mean_nonzero(delta)   = mean N_halo conditional on N_halo > 0
sigma(delta)          = scatter in ln(N_halo) for nonzero cells
```

These quantities are measured in statistically reliable calibration bins, smoothed between neighbouring bins, and continuously interpolated with density.

For a target cell, the code first determines whether it is empty using `P0`. If non-empty, the count is sampled from a lognormal distribution. The mean-corrected lognormal parameter is

```text
mu = ln(mean_nonzero) - sigma^2 / 2
```

so that the arithmetic mean of the sampled distribution is `mean_nonzero`.

The expected halo count is

```text
lambda_Nhalo = (1 - P0) * mean_nonzero
```

## 2. Why the current model differs from the original implementation

The original calibration used only one `6^3` partition of the small simulation:

```text
6^3 = 216 calibration cells
```

Some density bins therefore contained only a few halo-hosting cells. Applying those sparse distributions to a `250^3` target volume could replicate a handful of discrete halo counts across hundreds of thousands of cells, producing artificial horizontal structures in the `N_halo` versus density distribution.

The current implementation improves this by:

- using adaptive density bins;
- requiring a minimum number of nonzero calibration cells for a bin to be considered reliable;
- smoothing the zero probability, positive-halo mean and log scatter;
- interpolating the conditional relation continuously in density;
- using endpoint clipping for target densities outside the reliable calibration range;
- optionally using multiple shifted `6^3` partitions of the small simulation.

## 3. Main numerical parameters

Current defaults are:

```python
INPUT_GRID_SIZE = 250
SMALL_BOX_SIZE = 6.3
LARGE_BOX_SIZE = 244.0
STARTING_REDSHIFT = 30.0

BIN_WIDTH = 0.01
MIN_COUNT = 10
MAX_BIN_WIDTH = 0.10
MIN_RELIABLE_NONZERO = 10
SMOOTH_HALF_WINDOW = 1
CHUNK_SIZE = 1_000_000

OMEGA_M = 0.27
OMEGA_L = 0.73
```

`CHUNK_SIZE` controls the number of large-box cells sampled at once and can be reduced if memory becomes limiting.

## 4. Python requirements

The script requires:

```text
numpy
matplotlib
```

plus Python standard-library modules including `argparse`, `struct`, `dataclasses`, `pathlib`, and `typing`.

Run it inside the appropriate Python/pyC2Ray environment.

## 5. Redshift input files

### 5.1 Coarse redshift list

Default:

```text
./redshift_list.txt
```

This contains the actual simulation redshifts for which small-box calibration data exist.

Example:

```text
30.000
27.900
26.124
24.597
23.268
```

The loader also accepts a file whose first line contains the number of following redshifts.

Override with:

```bash
--coarse-redshift-file PATH
```

### 5.2 Fine redshift list

Default:

```text
./redshifts_fine.dat
```

This contains the complete fine/morphed timestep sequence used by the radiative-transfer calculation.

Example:

```text
26.401
26.124
25.854
25.590
25.333
25.082
24.837
24.597
```

Override with:

```bash
--fine-redshift-file PATH
```

## 6. Small-box calibration inputs

The calibration catalogue is an ASCII table with eight columns:

```text
density  halo_count_1  halo_count_2  ...  halo_count_7
```

The code reads it with:

```python
small_box_data = np.loadtxt(small_box_file)
```

and defines the total halo count in each calibration cell as:

```python
halo_num = np.sum(small_box_data[:, 1:], axis=1)
```

Despite the `.bin` extension, these calibration files are ASCII.

### 6.1 Unshifted calibration

Expected filename:

```text
{redshift:.3f}halo_num6.bin
```

Examples:

```text
30.000halo_num6.bin
24.597halo_num6.bin
23.268halo_num6.bin
```

Each file contains:

```text
6^3 = 216 rows
```

Run with:

```bash
python nhalo_implimentation.py --calibration unshifted
```

### 6.2 Shifted calibration

For `N` shifted coarse-grid realizations, the combined file must be named:

```text
{redshift:.3f}halo_num6_{N}shifts.bin
```

For 30 shifts:

```text
30.000halo_num6_30shifts.bin
24.597halo_num6_30shifts.bin
```

The expected number of rows is:

```text
N_shifts * 216
```

so 30 shifts give:

```text
6480 rows
```

Run with:

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30
```

Important: shifted grids are correlated views of the same physical small box. They improve sampling of `P(N_halo | delta)` but do not represent independent cosmological volume.

For abundance normalization, repeated halo counts are divided by the number of calibration realizations, equivalently using

```text
V_normalization = N_shifts * V_small
```

only as bookkeeping.

## 7. Large-box density inputs

The target grid is currently expected to be:

```text
250 x 250 x 250
```

The binary reader detects several formats automatically.

### Raw binary

```text
int32 nx
int32 ny
int32 nz
float32 density[nx * ny * nz]
```

with either little- or big-endian byte ordering.

### Fortran sequential binary

The reader supports both 4-byte and 8-byte record markers and either endian convention.

Logical contents:

```text
record 1: three int32 dimensions
record 2: float32 density field
```

The density payload is reshaped using:

```python
order="F"
```

to preserve CubeP3M/Fortran ordering.

## 8. Coarse-density mode

Without `--fine-redshifts`, the code operates at the actual simulation snapshots.

Default naming:

```text
{redshift:.3f}ntot_all.dat
```

Examples:

```text
30.000ntot_all.dat
27.900ntot_all.dat
24.597ntot_all.dat
```

Override the directory with:

```bash
--coarse-density-dir PATH
```

and the suffix with:

```bash
--coarse-density-suffix SUFFIX
```

## 9. Fine/morphed-redshift mode

Enable with:

```bash
--fine-redshifts
```

The code then reads the already-generated morphed density field at every fine redshift.

Default suffix:

```text
ntotcoarsened_all.dat
```

Examples:

```text
24.597ntotcoarsened_all.dat
24.363ntotcoarsened_all.dat
24.134ntotcoarsened_all.dat
23.910ntotcoarsened_all.dat
```

Override with:

```bash
--fine-density-dir PATH
--fine-density-suffix SUFFIX
```

### 9.1 How fine-redshift interpolation works

Suppose a fine redshift `z_f` lies between two real calibration snapshots:

```text
z_high > z_f > z_low
```

The code fits the conditional minihalo model separately at `z_high` and `z_low`.

It then reads the actual morphed density field at `z_f` and calculates:

```text
delta_f = rho_f / <rho_f> - 1
```

Both bounding conditional models are evaluated at this same `delta_f`:

```text
P0_high(delta_f)
mean_high(delta_f)
sigma_high(delta_f)

P0_low(delta_f)
mean_low(delta_f)
sigma_low(delta_f)
```

The two model predictions are then interpolated in cosmic time.

The interpolation fraction is:

```text
f_t = [t(z_f) - t(z_high)] / [t(z_low) - t(z_high)]
```

and

```text
P0_f    = (1 - f_t) P0_high    + f_t P0_low
mean_f  = (1 - f_t) mean_high  + f_t mean_low
sigma_f = (1 - f_t) sigma_high + f_t sigma_low
```

The script uses the flat-LambdaCDM analytic cosmic-time coordinate for `Omega_m = 0.27` and `Omega_L = 0.73`. Only the relative time coordinate is required, so the common Hubble-time normalization cancels in the interpolation fraction.

At an exact coarse redshift, `z_high = z_low`, so no redshift interpolation is performed.

## 10. Treatment outside the reliable density range

For each coarse calibration model, only bins with at least

```python
MIN_RELIABLE_NONZERO = 10
```

nonzero cells are used to define the continuous relation.

The target density is clipped to the reliable interval:

```text
delta_eval = clip(delta, delta_min_reliable, delta_max_reliable)
```

Therefore:

```text
delta < delta_min_reliable
```

uses the lowest reliable endpoint, while

```text
delta > delta_max_reliable
```

uses the highest reliable endpoint.

This is constant endpoint extrapolation rather than linear extrapolation.

In fine mode, clipping is performed independently for the two bounding coarse models before time interpolation.

## 11. Stochastic realization

For every target cell, the code evaluates:

```text
P0
mean_nonzero
sigma
```

The expected count is:

```text
lambda = (1 - P0) * mean_nonzero
```

A random draw decides whether the cell is empty. Nonzero cells are then sampled from the mean-corrected lognormal distribution and rounded to integer counts.

Sampling is performed in chunks to control memory use.

## 12. Reproducibility

The script uses NumPy's `default_rng`.

Default seed:

```text
12345
```

Override with:

```bash
--seed INTEGER
```

Example:

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --seed 42
```

## 13. Output directories

Coarse runs write to:

```text
results_coarse/
diagnostics_coarse/
logfiles_coarse/
```

Fine runs write to:

```text
results_fine/
diagnostics_fine/
logfiles_fine/
```

This prevents fine and coarse runs from overwriting each other.

## 14. Stochastic Nhalo outputs

Main output:

```text
results_<mode>/halo_num_<calibration>_z<redshift>.npy
```

Examples:

```text
results_coarse/halo_num_shifted_z24.597.npy
results_fine/halo_num_shifted_z24.363.npy
results_fine/halo_num_shifted_z24.134.npy
```

The file contains one halo count per `250^3` cell.

When reshaping it back to a 3-D pyC2Ray/CubeP3M grid, preserve Fortran ordering:

```python
nhalo = np.load(filename)
nhalo = nhalo.reshape((250, 250, 250), order="F")
```

## 15. Expected Nhalo outputs

Enable with:

```bash
--save-expected
```

This additionally writes:

```text
results_<mode>/expected_nhalo_<calibration>_z<redshift>.npy
```

These files contain:

```text
lambda = E[N_halo | delta, z] = (1 - P0) * mean_nonzero
```

rather than a stochastic realization.

They are useful for diagnostics and for applications requiring a smoother temporal estimate of the expected halo abundance.

## 16. Smoothed calibration outputs

For every coarse calibration snapshot used by the run:

```text
diagnostics_<mode>/smoothed_model_<calibration>_z<redshift>.txt
```

Columns:

```text
delta
prob_zero_raw
prob_zero_smooth
mean_nonzero_raw
mean_nonzero_smooth
sigma_raw
sigma_smooth
Ntotal
Nnonzero
```

These files record the actual conditional model used to generate the large-box catalogue.

## 17. Diagnostic plots

Enable with:

```bash
--plots
```

Examples:

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --plots
```

or:

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts --plots
```

Plots are saved in:

```text
results_coarse/
```

or:

```text
results_fine/
```

with names such as:

```text
scatterImp_shifted_z24.597.png
scatterImp_shifted_z24.134.png
```

Plots are disabled by default because producing one at every fine timestep increases runtime and disk I/O.

## 18. Number-density diagnostics

At the end of the run the code writes:

```text
number_density_ratio_<calibration>_<mode>.txt
number_density_ratio_<calibration>_<mode>.png
```

These are written in the directory from which the script is run.

The code also prints, for every redshift:

```text
interpolated small-box halos per physical box
large-box implemented halos
expected halos from simple volume scaling
expected halos from the density-conditioned model
implemented / density-conditioned expected
n_large / n_small
```

The most direct implementation check is:

```text
implemented / density-conditioned expected
```

because it tests whether the stochastic realization reproduces the expectation of the conditional model evaluated on the actual target density PDF.

## 19. Command-line options

```text
--calibration {unshifted,shifted}
--n-shifts N
--fine-redshifts
--coarse-redshift-file PATH
--fine-redshift-file PATH
--coarse-density-dir PATH
--fine-density-dir PATH
--coarse-density-suffix TEXT
--fine-density-suffix TEXT
--seed INTEGER
--save-expected
--plots
```

## 20. Common run configurations

### Unshifted calibration, coarse snapshots

```bash
python nhalo_implimentation.py --calibration unshifted
```

### Shifted calibration, coarse snapshots

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30
```

### Shifted calibration at all fine/morphed timesteps

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts
```

### Fine timesteps plus expected fields

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts --save-expected
```

### Fine timesteps plus plots

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts --plots
```

### Full diagnostic run

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts --save-expected --plots
```

### Custom redshift and density paths

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts --coarse-redshift-file ./redshifts.dat --fine-redshift-file ./redshifts_fine.dat --fine-density-dir /path/to/coarser_densities
```

## 21. Recommended production command for LW/pyC2Ray input

A typical production run is:

```bash
python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts --save-expected
```

This writes both a stochastic minihalo field and a continuous expected field at every fine timestep without the plotting overhead.

## 22. Scientific caveats

### Shifted calibrations are correlated

Shifted `6^3` grids improve sampling of the density-halo relation but do not increase the independent cosmological volume.

### Density tails remain extrapolated

Large-box densities outside the reliable calibration interval use the nearest reliable endpoint model.

### Stochastic snapshots are not merger trees

Each `halo_num_*.npy` file is a stochastic realization at one redshift. Differences between independently sampled adjacent files should not automatically be interpreted as a physical halo-formation history.

For temporally smooth applications, consider using `expected_nhalo_*.npy` or a dedicated temporally correlated realization.

### Small-box volume convention

`SMALL_BOX_SIZE` is currently set to:

```text
6.3
```

and the abundance normalization uses:

```text
V_small = 6.3^3
```

Confirm that this matches the intended units of the calibration volume before interpreting absolute number densities.

## 23. Suggested validation before production use

Check that:

1. the unshifted catalogue contains 216 rows, or the shifted catalogue contains `216 * N_shifts` rows;
2. target density files are detected as `250^3`;
3. all fine redshifts lie inside the coarse calibration-redshift range;
4. `implemented / density-conditioned expected` remains close to unity;
5. diagnostic `N_halo` versus density plots do not contain obvious sparse-bin artifacts;
6. the fraction of cells outside the reliable density range is understood;
7. `order="F"` is preserved when reshaping Nhalo fields for pyC2Ray;
8. shifted catalogues are treated as correlated repeated views rather than independent volumes;
9. the random seed is recorded for any production realization.

## 24. Example directory layout

```text
subgrid_model/
├── nhalo_implimentation.py
├── redshift_list.txt
├── redshifts_fine.dat
├── README.md
├── results_coarse/
├── results_fine/
├── diagnostics_coarse/
├── diagnostics_fine/
├── logfiles_coarse/
└── logfiles_fine/
```

Large density and calibration files are normally stored externally.

## 25. Model flow

```text
small high-resolution simulation
        |
        v
fit P(Nhalo | delta, z)
        |
        v
smoothed conditional model
        |
        +------------------------------+
        |                              |
        v                              v
coarse target density          morphed fine density
        |                              |
        v                              v
evaluate model             evaluate both coarse models
                                       |
                                       v
                              interpolate in cosmic time
                                       |
                    +------------------+------------------+
                    |                                     |
                    v                                     v
             expected Nhalo                       stochastic Nhalo
```

The same physical calibration is therefore used consistently at both native simulation snapshots and finer radiative-transfer timesteps.
