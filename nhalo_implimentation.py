# source /p/project/lgreion/david/hestia_bin/myenv/bin/activate
#
# Coarse redshifts:
# python nhalo_implimentation.py --calibration shifted --n-shifts 30
#
# Fine/morphed redshifts:
# python nhalo_implimentation.py --calibration shifted --n-shifts 30 --fine-redshifts
#
# Fine mode uses the existing morphed 250^3 density fields and interpolates
# the conditional Nhalo model between the two surrounding coarse snapshots.

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.rcParams.update({"font.size": 15})

BinaryFormat = Literal["raw", "sequential"]

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

small_box_cond_massfn_unshifted = Path("/e/project1/lgreion/david/cubepm_130314_6_1728_6.3Mpc_ext2/postprocessing")
small_box_cond_massfn_shifted = Path("/e/project1/lgreion/david/cubepm_130314_6_1728_6.3Mpc_ext2/halo_num6_shifted")
coarse_density_dir_default = Path("/e/project1/lgreion/david/LW_project/sph_smooth_cubepm_130329_10_4000_244Mpc_ext2_test/nc250")
fine_density_dir_default = Path("/e/project1/lgreion/david/LW_project/244RT_6.3Mpcsubgrid_250grid/coarser_densities")

parser = argparse.ArgumentParser(description="Generate large-box minihalo fields at coarse or morphed fine redshifts.")
parser.add_argument("--calibration", choices=("unshifted", "shifted"), default="unshifted")
parser.add_argument("--n-shifts", type=int, default=30)
parser.add_argument("--fine-redshifts", action="store_true")
parser.add_argument("--coarse-redshift-file", type=Path, default=Path("./redshift_list.txt"))
parser.add_argument("--fine-redshift-file", type=Path, default=Path("./redshifts_fine.dat"))
parser.add_argument("--coarse-density-dir", type=Path, default=coarse_density_dir_default)
parser.add_argument("--fine-density-dir", type=Path, default=fine_density_dir_default)
parser.add_argument("--coarse-density-suffix", default="ntot_all.dat")
parser.add_argument("--fine-density-suffix", default="ntotcoarsened_all.dat")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--save-expected", action="store_true")
parser.add_argument("--plots", action="store_true")
args = parser.parse_args()

USE_SHIFTED = args.calibration == "shifted"
N_CALIBRATION_REALIZATIONS = args.n_shifts if USE_SHIFTED else 1
RUN_MODE = "fine" if args.fine_redshifts else "coarse"

if N_CALIBRATION_REALIZATIONS < 1:
    raise ValueError("--n-shifts must be at least 1.")

logfile_dir = Path(f"./logfiles_{RUN_MODE}")
results_dir = Path(f"./results_{RUN_MODE}")
diagnostics_dir = Path(f"./diagnostics_{RUN_MODE}")

logfile_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
diagnostics_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(args.seed)
number_density_ratio_history = []
redshift_history = []


@dataclass(frozen=True)
class FileFormat:
    binary_format: BinaryFormat
    endian: Literal["<", ">"]
    marker_bytes: int
    data_offset: int
    dimensions: tuple[int, int, int]

    @property
    def endian_name(self):
        return "little-endian" if self.endian == "<" else "big-endian"

    @property
    def description(self):
        return f"raw binary, {self.endian_name}" if self.binary_format == "raw" else f"Fortran sequential binary, {self.marker_bytes}-byte record markers, {self.endian_name}"


@dataclass
class CalibrationModel:
    redshift: float
    reliable_delta: np.ndarray
    prob_zero: np.ndarray
    mean_nonzero: np.ndarray
    sigma: np.ndarray
    halo_density_small: float
    total_small_per_physical_box: float


def load_redshift_list(filename):
    values = np.atleast_1d(np.loadtxt(filename, dtype=float)).astype(np.float64)

    if values.size >= 2:
        first_as_int = int(round(values[0]))

        if np.isclose(values[0], first_as_int) and first_as_int == values.size - 1:
            values = values[1:]

    return values


def cosmic_time_coordinate(redshift):
    a = 1.0 / (1.0 + redshift)
    return np.arcsinh(np.sqrt(OMEGA_L / OMEGA_M) * a**1.5)


def interpolation_fraction_in_time(redshift, z_high, z_low):
    if np.isclose(z_high, z_low, atol=5.0e-7):
        return 0.0

    t_high = cosmic_time_coordinate(z_high)
    t_low = cosmic_time_coordinate(z_low)
    t_now = cosmic_time_coordinate(redshift)
    return float(np.clip((t_now - t_high) / (t_low - t_high), 0.0, 1.0))


def expected_data_bytes(dimensions):
    nx, ny, nz = dimensions
    return nx * ny * nz * np.dtype(np.float32).itemsize


def detect_raw_format(file_path, file_size):
    with file_path.open("rb") as handle:
        header = handle.read(12)

    if len(header) != 12:
        return None

    expected_dimensions = (INPUT_GRID_SIZE, INPUT_GRID_SIZE, INPUT_GRID_SIZE)

    for endian in ("<", ">"):
        dimensions = struct.unpack(f"{endian}3i", header)

        if dimensions == expected_dimensions and file_size == 12 + expected_data_bytes(dimensions):
            return FileFormat("raw", endian, 0, 12, dimensions)

    return None


def detect_sequential_format(file_path, file_size, marker_bytes):
    marker_code = "i" if marker_bytes == 4 else "q"
    expected_dimensions = (INPUT_GRID_SIZE, INPUT_GRID_SIZE, INPUT_GRID_SIZE)

    for endian in ("<", ">"):
        marker_struct = struct.Struct(f"{endian}{marker_code}")
        dimensions_struct = struct.Struct(f"{endian}3i")

        with file_path.open("rb") as handle:
            first_marker_raw = handle.read(marker_bytes)

            if len(first_marker_raw) != marker_bytes or marker_struct.unpack(first_marker_raw)[0] != 12:
                continue

            dimensions_raw = handle.read(12)

            if len(dimensions_raw) != 12:
                continue

            dimensions = dimensions_struct.unpack(dimensions_raw)

            if dimensions != expected_dimensions:
                continue

            end_header = handle.read(marker_bytes)
            start_data = handle.read(marker_bytes)

            if len(end_header) != marker_bytes or len(start_data) != marker_bytes:
                continue

            data_bytes = expected_data_bytes(dimensions)

            if marker_struct.unpack(end_header)[0] != 12 or marker_struct.unpack(start_data)[0] != data_bytes:
                continue

        required_size = marker_bytes + 12 + marker_bytes + marker_bytes + data_bytes + marker_bytes

        if file_size == required_size:
            data_offset = marker_bytes + 12 + marker_bytes + marker_bytes
            return FileFormat("sequential", endian, marker_bytes, data_offset, dimensions)

    return None


def detect_file_format(file_path):
    file_size = file_path.stat().st_size
    raw_format = detect_raw_format(file_path, file_size)

    if raw_format is not None:
        return raw_format

    for marker_bytes in (4, 8):
        sequential_format = detect_sequential_format(file_path, file_size, marker_bytes)

        if sequential_format is not None:
            return sequential_format

    raise ValueError(f"Could not identify binary format of {file_path}.")


def read_density(file_path, file_format):
    number_of_values = int(np.prod(file_format.dimensions))
    dtype = np.dtype(f"{file_format.endian}f4")

    with file_path.open("rb") as handle:
        handle.seek(file_format.data_offset)
        density_flat = np.fromfile(handle, dtype=dtype, count=number_of_values)

    if density_flat.size != number_of_values:
        raise ValueError(f"Expected {number_of_values:,} values in {file_path}, read {density_flat.size:,}.")

    density = density_flat.reshape(file_format.dimensions, order="F").astype(np.float32, copy=False)

    if not np.all(np.isfinite(density)):
        raise ValueError(f"{file_path} contains non-finite density values.")

    return density


def adaptive_bins_with_initial_empty(data, start, end, bin_width, min_count, max_bin_width):
    bin_edges = [start]
    current = start
    found_data = False
    tolerance = 1.0e-12

    while current < end - tolerance:
        if not found_data:
            next_edge = min(current + bin_width, end)
            count = np.count_nonzero((data >= current) & (data < next_edge))
            bin_edges.append(next_edge)
            current = next_edge
            found_data = count > 0
            continue

        width = bin_width
        selected_edge = None

        while width <= max_bin_width + tolerance:
            next_edge = min(current + width, end)
            count = np.count_nonzero((data >= current) & (data < next_edge))

            if count >= min_count or next_edge >= end - tolerance:
                selected_edge = next_edge
                break

            width += bin_width

        if selected_edge is None:
            selected_edge = min(current + max_bin_width, end)

        if selected_edge <= current + tolerance:
            break

        bin_edges.append(selected_edge)
        current = selected_edge

    if bin_edges[-1] < end:
        bin_edges.append(end)

    return np.unique(np.asarray(bin_edges))


def weighted_smooth(values, weights, half_window=1):
    smoothed = np.zeros_like(values, dtype=np.float64)

    for i in range(values.size):
        lo = max(0, i - half_window)
        hi = min(values.size, i + half_window + 1)
        local_weights = weights[lo:hi].astype(np.float64)
        local_values = values[lo:hi].astype(np.float64)

        if np.sum(local_weights) > 0:
            smoothed[i] = np.sum(local_values * local_weights) / np.sum(local_weights)
        else:
            smoothed[i] = values[i]

    return smoothed


def get_small_box_file(redshift):
    if USE_SHIFTED:
        return small_box_cond_massfn_shifted / f"{redshift:.3f}halo_num6_{N_CALIBRATION_REALIZATIONS}shifts.bin"

    return small_box_cond_massfn_unshifted / f"{redshift:.3f}halo_num6.bin"


def fit_calibration_model(redshift):
    small_box_file = get_small_box_file(redshift)

    if not small_box_file.exists():
        raise FileNotFoundError(f"Small-box halo file not found: {small_box_file}")

    small_box_data = np.loadtxt(small_box_file)
    expected_rows = N_CALIBRATION_REALIZATIONS * 6**3

    if small_box_data.shape[0] != expected_rows:
        raise RuntimeError(f"{small_box_file} has {small_box_data.shape[0]} rows, expected {expected_rows}.")

    dens_small = small_box_data[:, 0]
    halo_num = np.sum(small_box_data[:, 1:], axis=1)
    overdense_small = dens_small / np.mean(dens_small, dtype=np.float64)
    delta_small = overdense_small - 1.0
    nonzero_mask = halo_num > 0
    delta_nonzero = delta_small[nonzero_mask]
    halo_num_nonzero = halo_num[nonzero_mask]

    if halo_num_nonzero.size == 0:
        raise RuntimeError(f"No halos in calibration at z={redshift:.3f}.")

    bin_edges = adaptive_bins_with_initial_empty(delta_nonzero, -1.0, np.max(delta_nonzero), BIN_WIDTH, MIN_COUNT, MAX_BIN_WIDTH)
    bin_indices_nonzero = np.digitize(delta_nonzero, bin_edges)
    bin_indices_all = np.digitize(delta_small, bin_edges)
    n_bins = len(bin_edges) - 1

    delta_per_bin = np.zeros(n_bins)
    sigma_per_bin = np.zeros(n_bins)
    n_nonzero_per_bin = np.zeros(n_bins, dtype=int)
    mean_halo_nonzero_per_bin = np.zeros(n_bins)
    prob_zero = np.ones(n_bins)
    n_total_per_bin = np.zeros(n_bins, dtype=int)

    for i in range(1, len(bin_edges)):
        in_bin_halo_all = halo_num[bin_indices_all == i]
        in_bin_halo_nonzero = halo_num_nonzero[bin_indices_nonzero == i]
        in_bin_delta_nonzero = delta_nonzero[bin_indices_nonzero == i]

        if in_bin_halo_all.size > 0:
            n_total_per_bin[i - 1] = in_bin_halo_all.size
            prob_zero[i - 1] = np.count_nonzero(in_bin_halo_all == 0) / in_bin_halo_all.size

        if in_bin_halo_nonzero.size == 0:
            continue

        delta_per_bin[i - 1] = np.mean(in_bin_delta_nonzero)
        sigma_per_bin[i - 1] = np.std(np.log(in_bin_halo_nonzero))
        n_nonzero_per_bin[i - 1] = in_bin_halo_nonzero.size
        mean_halo_nonzero_per_bin[i - 1] = np.mean(in_bin_halo_nonzero)

    reliable_bins = np.flatnonzero(n_nonzero_per_bin >= MIN_RELIABLE_NONZERO)

    if reliable_bins.size < 2:
        raise RuntimeError(f"Fewer than two reliable calibration bins at z={redshift:.3f}.")

    reliable_delta = delta_per_bin[reliable_bins]
    reliable_order = np.argsort(reliable_delta)
    reliable_bins = reliable_bins[reliable_order]
    reliable_delta = reliable_delta[reliable_order]
    reliable_counts = n_total_per_bin[reliable_bins].astype(np.float64)
    reliable_nonzero_counts = n_nonzero_per_bin[reliable_bins].astype(np.float64)
    reliable_prob_zero = prob_zero[reliable_bins]
    reliable_sigma = sigma_per_bin[reliable_bins]
    reliable_mean_nonzero = mean_halo_nonzero_per_bin[reliable_bins]

    smoothed_prob_zero = np.clip(weighted_smooth(reliable_prob_zero, reliable_counts, SMOOTH_HALF_WINDOW), 0.0, 1.0)
    smoothed_sigma = np.maximum(weighted_smooth(reliable_sigma, reliable_nonzero_counts, SMOOTH_HALF_WINDOW), 0.05)
    smoothed_mean_nonzero = np.maximum(weighted_smooth(reliable_mean_nonzero, reliable_nonzero_counts, SMOOTH_HALF_WINDOW), 1.0)

    total_small_all_realizations = np.sum(halo_num)
    halo_density_small = total_small_all_realizations / (SMALL_BOX_SIZE**3 * N_CALIBRATION_REALIZATIONS)
    total_small_per_physical_box = total_small_all_realizations / N_CALIBRATION_REALIZATIONS

    smooth_data = np.column_stack((reliable_delta, reliable_prob_zero, smoothed_prob_zero, reliable_mean_nonzero, smoothed_mean_nonzero, reliable_sigma, smoothed_sigma, reliable_counts, reliable_nonzero_counts))
    np.savetxt(diagnostics_dir / f"smoothed_model_{args.calibration}_z{redshift:.3f}.txt", smooth_data, fmt="%.6e", header="delta prob_zero_raw prob_zero_smooth mean_nonzero_raw mean_nonzero_smooth sigma_raw sigma_smooth Ntotal Nnonzero")

    print(f"Fitted calibration z={redshift:.3f}: reliable delta=[{reliable_delta[0]:.4f},{reliable_delta[-1]:.4f}]")

    return CalibrationModel(redshift, reliable_delta, smoothed_prob_zero, smoothed_mean_nonzero, smoothed_sigma, halo_density_small, total_small_per_physical_box)


def find_bracketing_coarse_redshifts(redshift, coarse_redshifts):
    tolerance = 5.0e-4
    exact = np.where(np.abs(coarse_redshifts - redshift) <= tolerance)[0]

    if exact.size > 0:
        z_exact = float(coarse_redshifts[exact[0]])
        return z_exact, z_exact

    higher = coarse_redshifts[coarse_redshifts > redshift]
    lower = coarse_redshifts[coarse_redshifts < redshift]

    if higher.size == 0 or lower.size == 0:
        raise ValueError(f"z={redshift:.3f} lies outside the coarse calibration range.")

    return float(np.min(higher)), float(np.max(lower))


def evaluate_model(model, delta_values):
    delta_eval = np.clip(delta_values, model.reliable_delta[0], model.reliable_delta[-1])
    p0 = np.interp(delta_eval, model.reliable_delta, model.prob_zero)
    mean_nonzero = np.interp(delta_eval, model.reliable_delta, model.mean_nonzero)
    sigma = np.interp(delta_eval, model.reliable_delta, model.sigma)
    return p0, mean_nonzero, sigma


def get_density_file(redshift):
    if args.fine_redshifts:
        return args.fine_density_dir / f"{redshift:.3f}{args.fine_density_suffix}"

    return args.coarse_density_dir / f"{redshift:.3f}{args.coarse_density_suffix}"


coarse_redshifts = load_redshift_list(args.coarse_redshift_file)
coarse_redshifts = coarse_redshifts[coarse_redshifts <= STARTING_REDSHIFT + 5.0e-4]
coarse_redshifts = np.sort(coarse_redshifts)[::-1]

if args.fine_redshifts:
    target_redshifts = load_redshift_list(args.fine_redshift_file)
    target_redshifts = target_redshifts[target_redshifts <= STARTING_REDSHIFT + 5.0e-4]
    target_redshifts = target_redshifts[target_redshifts >= np.min(coarse_redshifts) - 5.0e-4]
    target_redshifts = np.sort(target_redshifts)[::-1]
else:
    target_redshifts = coarse_redshifts.copy()

print(f"Calibration mode: {args.calibration}")
print(f"Run mode: {RUN_MODE}")
print(f"Coarse calibration redshifts: {coarse_redshifts.size}")
print(f"Target redshifts: {target_redshifts.size}")
print(f"Density directory: {args.fine_density_dir if args.fine_redshifts else args.coarse_density_dir}")

calibration_cache = {}

for redshift in target_redshifts:
    print()
    print("=" * 90)
    print(f"Processing target z={redshift:.3f}")
    print("=" * 90)

    z_high, z_low = find_bracketing_coarse_redshifts(float(redshift), coarse_redshifts)

    for z_model in {z_high, z_low}:
        key = round(z_model, 3)

        if key not in calibration_cache:
            calibration_cache[key] = fit_calibration_model(z_model)

    model_high = calibration_cache[round(z_high, 3)]
    model_low = calibration_cache[round(z_low, 3)]
    time_fraction = interpolation_fraction_in_time(float(redshift), z_high, z_low)

    print(f"Calibration bracket: z_high={z_high:.3f}, z_low={z_low:.3f}, time fraction={time_fraction:.6f}")

    density_file = get_density_file(float(redshift))

    if not density_file.exists():
        raise FileNotFoundError(f"Density field not found: {density_file}")

    file_format = detect_file_format(density_file)
    dens_large_3d = read_density(density_file, file_format)
    dens_large = dens_large_3d.ravel(order="F")
    overdense_large = dens_large / np.mean(dens_large, dtype=np.float64)
    delta_large = overdense_large - 1.0

    Nhalo = np.zeros(delta_large.size, dtype=np.float64)
    expected_field = np.zeros(delta_large.size, dtype=np.float32) if args.save_expected else None
    expected_density_weighted_total = 0.0

    print(f"Density file: {density_file.name}")
    print(f"Format: {file_format.description}")
    print(f"delta range: {np.min(delta_large):.6f} to {np.max(delta_large):.6f}")

    for start in range(0, delta_large.size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, delta_large.size)
        delta_chunk = delta_large[start:end]

        p0_high, mean_high, sigma_high = evaluate_model(model_high, delta_chunk)

        if np.isclose(z_high, z_low):
            p0_chunk = p0_high
            mean_chunk = mean_high
            sigma_chunk = sigma_high
        else:
            p0_low, mean_low, sigma_low = evaluate_model(model_low, delta_chunk)
            p0_chunk = (1.0 - time_fraction) * p0_high + time_fraction * p0_low
            mean_chunk = (1.0 - time_fraction) * mean_high + time_fraction * mean_low
            sigma_chunk = (1.0 - time_fraction) * sigma_high + time_fraction * sigma_low

        p0_chunk = np.clip(p0_chunk, 0.0, 1.0)
        mean_chunk = np.maximum(mean_chunk, 1.0)
        sigma_chunk = np.maximum(sigma_chunk, 0.05)

        lambda_chunk = (1.0 - p0_chunk) * mean_chunk
        expected_density_weighted_total += np.sum(lambda_chunk)

        if expected_field is not None:
            expected_field[start:end] = lambda_chunk.astype(np.float32)

        nonzero_chunk_mask = rng.random(delta_chunk.size) >= p0_chunk

        if not np.any(nonzero_chunk_mask):
            continue

        sigma_nonzero = sigma_chunk[nonzero_chunk_mask]
        mean_nonzero_target = mean_chunk[nonzero_chunk_mask]
        mu_nonzero = np.log(mean_nonzero_target) - 0.5 * sigma_nonzero**2
        sampled_values = np.exp(rng.normal(mu_nonzero, sigma_nonzero))
        sampled_values = np.maximum(1.0, np.rint(sampled_values))
        chunk_output = np.zeros(delta_chunk.size, dtype=np.float64)
        chunk_output[nonzero_chunk_mask] = sampled_values
        Nhalo[start:end] = chunk_output

    halo_density_small_interp = (1.0 - time_fraction) * model_high.halo_density_small + time_fraction * model_low.halo_density_small
    total_small_interp = (1.0 - time_fraction) * model_high.total_small_per_physical_box + time_fraction * model_low.total_small_per_physical_box
    total_large = np.sum(Nhalo)
    volume_large = LARGE_BOX_SIZE**3
    halo_density_large = total_large / volume_large
    density_ratio = halo_density_large / halo_density_small_interp if halo_density_small_interp > 0 else np.nan
    expected_large_from_volume = halo_density_small_interp * volume_large
    density_weighted_ratio = total_large / expected_density_weighted_total if expected_density_weighted_total > 0 else np.nan

    print("Halo abundance check")
    print(f"Interpolated small-box halos per physical box: {total_small_interp:.3f}")
    print(f"Large-box implemented halos: {total_large:.0f}")
    print(f"Expected from simple volume scaling: {expected_large_from_volume:.6e}")
    print(f"Expected from interpolated density-conditioned model: {expected_density_weighted_total:.6e}")
    print(f"Implemented / density-conditioned expected: {density_weighted_ratio:.6f}")
    print(f"n_large / n_small: {density_ratio:.6f}")

    number_density_ratio_history.append(density_ratio)
    redshift_history.append(float(redshift))

    output_file = results_dir / f"halo_num_{args.calibration}_z{redshift:.3f}.npy"
    np.save(output_file, Nhalo)
    print(f"Saved: {output_file}")

    if expected_field is not None:
        expected_file = results_dir / f"expected_nhalo_{args.calibration}_z{redshift:.3f}.npy"
        np.save(expected_file, expected_field)
        print(f"Saved: {expected_file}")

    if args.plots:
        mock_mask = Nhalo > 0

        if np.any(mock_mask):
            x1 = np.log10(delta_large[mock_mask] + 1.0)
            y1 = np.log10(Nhalo[mock_mask])
            heatmap1, xedges1, yedges1 = np.histogram2d(x1, y1, bins=100)
            positive1 = heatmap1[heatmap1 > 0]

            if positive1.size > 0:
                norm = LogNorm(vmin=max(1.0, np.min(positive1)), vmax=np.max(heatmap1))
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(heatmap1.T, origin="lower", cmap="viridis", aspect="auto", extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]], norm=norm)
                ax.set_title(f"Implemented mock catalogue z={redshift:.3f}")
                ax.set_xlabel(r"$\log_{10}(\delta + 1)$")
                ax.set_ylabel(r"$\log_{10}(N_{\rm halo})$")
                fig.colorbar(im, ax=ax, label="Counts")
                plt.tight_layout()
                plt.savefig(results_dir / f"scatterImp_{args.calibration}_z{redshift:.3f}.png")
                plt.close(fig)

history = np.column_stack((redshift_history, number_density_ratio_history))
np.savetxt(f"number_density_ratio_{args.calibration}_{RUN_MODE}.txt", history, fmt="%.6e", header="redshift n_large_over_n_small")

plt.figure()
plt.plot(redshift_history, number_density_ratio_history)
plt.xlabel(r"$z$")
plt.ylabel(r"$n_{\rm halo,large}/n_{\rm halo,small}$")
plt.tight_layout()
plt.savefig(f"number_density_ratio_{args.calibration}_{RUN_MODE}.png")
plt.close()

print("Finished.")