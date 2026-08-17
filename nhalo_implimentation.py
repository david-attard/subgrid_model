# source /p/project/lgreion/david/hestia_bin/myenv/bin/activate

# Run using python nhalo_implimentation_new.py --calibration unshifted when using the unshifted files
# Run using python nhalo_implimentation_new.py --calibration shifted --n-shifts 30 when using the shifted files

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

small_box_cond_massfn_unshifted = Path("/e/project1/lgreion/david/cubepm_130314_6_1728_6.3Mpc_ext2/postprocessing")
small_box_cond_massfn_shifted = Path("/e/project1/lgreion/david/cubepm_130314_6_1728_6.3Mpc_ext2/halo_num6_shifted")
large_box_density_dir = Path("/e/project1/lgreion/david/LW_project/sph_smooth_cubepm_130329_10_4000_244Mpc_ext2_test/nc250")
logfile_dir = Path("./logfiles")
empty_halo_dir = Path("./empty_halo_num")
results_dir = Path("./results")
diagnostics_dir = Path("./diagnostics")

logfile_dir.mkdir(parents=True, exist_ok=True)
empty_halo_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
diagnostics_dir.mkdir(parents=True, exist_ok=True)

redshifts = np.atleast_1d(np.loadtxt("./redshift_list.txt"))
redshifts = redshifts[redshifts <= STARTING_REDSHIFT]

number_density_ratio_history = []
redshift_history = []


parser = argparse.ArgumentParser(description="Implement the large-box subgrid halo model using shifted or unshifted small-box calibration files.")
parser.add_argument("--calibration", choices=("unshifted", "shifted"), default="unshifted", help="Choose the small-box calibration catalogue.")
parser.add_argument("--n-shifts", type=int, default=30, help="Number of shifts contained in the combined shifted catalogue.")
args = parser.parse_args()

USE_SHIFTED = args.calibration == "shifted"
N_CALIBRATION_REALIZATIONS = args.n_shifts if USE_SHIFTED else 1

if N_CALIBRATION_REALIZATIONS < 1:
    raise ValueError("--n-shifts must be at least 1.")

print(f"Calibration mode: {'shifted' if USE_SHIFTED else 'unshifted'}")
print(f"Calibration realizations per physical box: {N_CALIBRATION_REALIZATIONS}")


@dataclass(frozen=True)
class FileFormat:
    binary_format: BinaryFormat
    endian: Literal["<", ">"]
    marker_bytes: int
    data_offset: int
    dimensions: tuple[int, int, int]

    @property
    def endian_name(self) -> str:
        return "little-endian" if self.endian == "<" else "big-endian"

    @property
    def description(self) -> str:
        return f"raw binary, {self.endian_name}" if self.binary_format == "raw" else f"Fortran sequential binary, {self.marker_bytes}-byte record markers, {self.endian_name}"


def expected_data_bytes(dimensions):
    nx, ny, nz = dimensions
    return nx * ny * nz * np.dtype(np.float32).itemsize


def detect_raw_format(file_path: Path, file_size: int):
    with file_path.open("rb") as handle:
        header = handle.read(12)

    if len(header) != 12:
        return None

    expected_dimensions = (INPUT_GRID_SIZE, INPUT_GRID_SIZE, INPUT_GRID_SIZE)

    for endian in ("<", ">"):
        dimensions = struct.unpack(f"{endian}3i", header)

        if dimensions != expected_dimensions:
            continue

        required_size = 12 + expected_data_bytes(dimensions)

        if file_size == required_size:
            return FileFormat(binary_format="raw", endian=endian, marker_bytes=0, data_offset=12, dimensions=dimensions)

    return None


def detect_sequential_format(file_path: Path, file_size: int, marker_bytes: int):
    if marker_bytes == 4:
        marker_code = "i"
    elif marker_bytes == 8:
        marker_code = "q"
    else:
        raise ValueError("Record markers must be 4 or 8 bytes.")

    header_record_bytes = 12
    expected_dimensions = (INPUT_GRID_SIZE, INPUT_GRID_SIZE, INPUT_GRID_SIZE)

    for endian in ("<", ">"):
        marker_struct = struct.Struct(f"{endian}{marker_code}")
        dimensions_struct = struct.Struct(f"{endian}3i")

        with file_path.open("rb") as handle:
            first_marker_raw = handle.read(marker_bytes)

            if len(first_marker_raw) != marker_bytes:
                continue

            if marker_struct.unpack(first_marker_raw)[0] != header_record_bytes:
                continue

            dimensions_raw = handle.read(header_record_bytes)

            if len(dimensions_raw) != header_record_bytes:
                continue

            dimensions = dimensions_struct.unpack(dimensions_raw)

            if dimensions != expected_dimensions:
                continue

            first_end_marker_raw = handle.read(marker_bytes)

            if len(first_end_marker_raw) != marker_bytes:
                continue

            if marker_struct.unpack(first_end_marker_raw)[0] != header_record_bytes:
                continue

            second_marker_raw = handle.read(marker_bytes)

            if len(second_marker_raw) != marker_bytes:
                continue

            data_bytes = expected_data_bytes(dimensions)

            if marker_struct.unpack(second_marker_raw)[0] != data_bytes:
                continue

        required_size = marker_bytes + header_record_bytes + marker_bytes + marker_bytes + data_bytes + marker_bytes

        if file_size != required_size:
            continue

        data_offset = marker_bytes + header_record_bytes + marker_bytes + marker_bytes
        return FileFormat(binary_format="sequential", endian=endian, marker_bytes=marker_bytes, data_offset=data_offset, dimensions=dimensions)

    return None


def detect_file_format(file_path: Path) -> FileFormat:
    file_size = file_path.stat().st_size
    raw_format = detect_raw_format(file_path, file_size)

    if raw_format is not None:
        return raw_format

    for marker_bytes in (4, 8):
        sequential_format = detect_sequential_format(file_path, file_size, marker_bytes)

        if sequential_format is not None:
            return sequential_format

    expected_raw_size = 12 + INPUT_GRID_SIZE**3 * np.dtype(np.float32).itemsize
    expected_sequential_4_size = expected_raw_size + 16
    expected_sequential_8_size = expected_raw_size + 32
    raise ValueError(f"Could not identify binary format of {file_path}. Actual size={file_size:,}, expected raw={expected_raw_size:,}, expected sequential-4={expected_sequential_4_size:,}, expected sequential-8={expected_sequential_8_size:,}.")


def read_density(file_path: Path, file_format: FileFormat) -> np.ndarray:
    number_of_values = int(np.prod(file_format.dimensions))
    dtype = np.dtype(f"{file_format.endian}f4")

    with file_path.open("rb") as handle:
        handle.seek(file_format.data_offset)
        density_flat = np.fromfile(handle, dtype=dtype, count=number_of_values)

    if density_flat.size != number_of_values:
        raise ValueError(f"Expected {number_of_values:,} density values in {file_path}, but read {density_flat.size:,}.")

    density = density_flat.reshape(file_format.dimensions, order="F")
    density = density.astype(np.float32, copy=False)

    if not np.all(np.isfinite(density)):
        non_finite = np.count_nonzero(~np.isfinite(density))
        raise ValueError(f"{file_path} contains {non_finite:,} non-finite density values.")

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


def interpolate_clipped(x, xp, fp):
    x_clipped = np.clip(x, xp[0], xp[-1])
    return np.interp(x_clipped, xp, fp)


def get_small_box_file(redshift):
    if USE_SHIFTED:
        return small_box_cond_massfn_shifted / f"{redshift:.3f}halo_num6_{N_CALIBRATION_REALIZATIONS}shifts.bin"

    return small_box_cond_massfn_unshifted / f"{redshift:.3f}halo_num6.bin"


print("Redshifts:", redshifts)

for redshift in redshifts:
    print()
    print("=" * 80)
    print(f"Processing z={redshift:.3f}")
    print("=" * 80)

    # Read the selected small-box calibration catalogue.
    small_box_file = get_small_box_file(redshift)

    if not small_box_file.exists():
        raise FileNotFoundError(f"Small-box halo file not found: {small_box_file}")

    small_box_data = np.loadtxt(small_box_file)
    dens_small = small_box_data[:, 0]
    halo_num = np.sum(small_box_data[:, 1:], axis=1)

    expected_rows = N_CALIBRATION_REALIZATIONS * 6**3

    if small_box_data.shape[0] != expected_rows:
        raise RuntimeError(f"{small_box_file} has {small_box_data.shape[0]} rows, but {expected_rows} rows are expected for calibration mode '{args.calibration}'.")

    print(f"Small-box calibration file: {small_box_file}")
    print(f"Small-box calibration rows: {small_box_data.shape[0]:,}")
    overdense_small = dens_small / np.mean(dens_small, dtype=np.float64)
    delta_small = overdense_small - 1.0
    nonzero_mask = halo_num > 0
    delta_nonzero = delta_small[nonzero_mask]
    halo_num_nonzero = halo_num[nonzero_mask]

    if halo_num_nonzero.size == 0:
        print(f"No halos in the small-box calibration at z={redshift:.3f}; skipping.")
        continue

    # Read large-box density.
    density_file = large_box_density_dir / f"{redshift:.3f}ntot_all.dat"

    if not density_file.exists():
        raise FileNotFoundError(f"Large-box density field not found: {density_file}")

    file_format = detect_file_format(density_file)
    dens_large_3d = read_density(density_file, file_format)
    dens_large = dens_large_3d.ravel(order="F")
    overdense_large = dens_large / np.mean(dens_large, dtype=np.float64)
    delta_large = overdense_large - 1.0
    Nhalo = np.zeros(delta_large.size, dtype=np.float64)

    print(f"Reading large-box density: {density_file.name}")
    print(f"Format: {file_format.description}")
    print(f"Large-box density shape: {dens_large_3d.shape}")
    print(f"Large-box mean density: {np.mean(dens_large, dtype=np.float64):.6f}")
    print(f"Large-box density range: {np.min(dens_large):.6f} to {np.max(dens_large):.6f}")
    print(f"Number of target cells: {Nhalo.size:,}")

    # Construct adaptive calibration bins.
    bin_edges = adaptive_bins_with_initial_empty(delta_nonzero, start=-1.0, end=np.max(delta_nonzero), bin_width=BIN_WIDTH, min_count=MIN_COUNT, max_bin_width=MAX_BIN_WIDTH)
    bin_indices_nonzero = np.digitize(delta_nonzero, bin_edges)
    bin_indices_all = np.digitize(delta_small, bin_edges)
    n_bins = len(bin_edges) - 1

    delta_per_bin = np.zeros(n_bins)
    sigma_per_bin = np.zeros(n_bins)
    n_nonzero_per_bin = np.zeros(n_bins, dtype=int)
    min_halo_per_bin = np.zeros(n_bins)
    max_halo_per_bin = np.zeros(n_bins)
    mean_halo_nonzero_per_bin = np.zeros(n_bins)
    prob_zero = np.ones(n_bins)
    total_halos_per_bin = np.zeros(n_bins)
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
        min_halo_per_bin[i - 1] = np.min(in_bin_halo_nonzero)
        max_halo_per_bin[i - 1] = np.max(in_bin_halo_nonzero)
        mean_halo_nonzero_per_bin[i - 1] = np.mean(in_bin_halo_nonzero)
        total_halos_per_bin[i - 1] = np.sum(in_bin_halo_nonzero)

    reliable_bins = np.flatnonzero(n_nonzero_per_bin >= MIN_RELIABLE_NONZERO)

    if reliable_bins.size < 2:
        print(f"Fewer than two reliable calibration bins at z={redshift:.3f}; skipping.")
        continue

    reliable_delta = delta_per_bin[reliable_bins]
    reliable_order = np.argsort(reliable_delta)
    reliable_bins = reliable_bins[reliable_order]
    reliable_delta = reliable_delta[reliable_order]
    reliable_counts = n_total_per_bin[reliable_bins].astype(np.float64)
    reliable_nonzero_counts = n_nonzero_per_bin[reliable_bins].astype(np.float64)
    reliable_prob_zero = prob_zero[reliable_bins]
    reliable_sigma = sigma_per_bin[reliable_bins]
    reliable_mean_nonzero = mean_halo_nonzero_per_bin[reliable_bins]

    smoothed_prob_zero = weighted_smooth(reliable_prob_zero, reliable_counts, SMOOTH_HALF_WINDOW)
    smoothed_sigma = weighted_smooth(reliable_sigma, reliable_nonzero_counts, SMOOTH_HALF_WINDOW)
    smoothed_mean_nonzero = weighted_smooth(reliable_mean_nonzero, reliable_nonzero_counts, SMOOTH_HALF_WINDOW)

    smoothed_prob_zero = np.clip(smoothed_prob_zero, 0.0, 1.0)
    smoothed_sigma = np.maximum(smoothed_sigma, 0.05)
    smoothed_mean_nonzero = np.maximum(smoothed_mean_nonzero, 1.0)

    min_reliable_delta = reliable_delta[0]
    max_reliable_delta = reliable_delta[-1]
    n_below_reliable = np.count_nonzero(delta_large < min_reliable_delta)
    n_above_reliable = np.count_nonzero(delta_large > max_reliable_delta)

    print()
    print("Reliable calibration range")
    print(f"Minimum reliable delta: {min_reliable_delta:.6f}")
    print(f"Maximum reliable delta: {max_reliable_delta:.6f}")
    print(f"Large-box cells below reliable range: {n_below_reliable:,} ({100.0 * n_below_reliable / delta_large.size:.6f}%)")
    print(f"Large-box cells above reliable range: {n_above_reliable:,} ({100.0 * n_above_reliable / delta_large.size:.6f}%)")

    print()
    print("Reliable calibration statistics")
    print("delta       Ntotal   Nnonzero   p0_raw   p0_smooth   mean_raw   mean_smooth   sigma_raw   sigma_smooth")

    for j, bin_index in enumerate(reliable_bins):
        print(f"{reliable_delta[j]:8.4f}   {n_total_per_bin[bin_index]:6d}   {n_nonzero_per_bin[bin_index]:8d}   {reliable_prob_zero[j]:6.3f}   {smoothed_prob_zero[j]:9.3f}   {reliable_mean_nonzero[j]:8.3f}   {smoothed_mean_nonzero[j]:11.3f}   {reliable_sigma[j]:9.3f}   {smoothed_sigma[j]:12.3f}")

    # Plot raw and smoothed zero probability.
    valid_prob = n_total_per_bin > 0
    plt.figure()
    plt.plot(np.log10(delta_per_bin[valid_prob] + 1.0), prob_zero[valid_prob], "o-", label="Raw bins")
    plt.plot(np.log10(reliable_delta + 1.0), smoothed_prob_zero, "o-", label="Smoothed reliable bins")
    plt.xlabel(r"$\log_{10}(\delta + 1)$")
    plt.ylabel(r"Probability $N_{\rm halo}=0$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(empty_halo_dir / f"nhalo_prob_{args.calibration}_z{redshift:.3f}.png")
    plt.close()

    # Apply the continuously interpolated conditional model in chunks.
    expected_density_weighted_total = 0.0

    for start in range(0, delta_large.size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, delta_large.size)
        delta_chunk = delta_large[start:end]
        delta_eval = np.clip(delta_chunk, min_reliable_delta, max_reliable_delta)
        p0_chunk = np.interp(delta_eval, reliable_delta, smoothed_prob_zero)
        sigma_chunk = np.interp(delta_eval, reliable_delta, smoothed_sigma)
        mean_nonzero_chunk = np.interp(delta_eval, reliable_delta, smoothed_mean_nonzero)

        expected_density_weighted_total += np.sum((1.0 - p0_chunk) * mean_nonzero_chunk)

        random_zero = np.random.random(delta_chunk.size)
        nonzero_chunk_mask = random_zero >= p0_chunk

        if not np.any(nonzero_chunk_mask):
            continue

        sigma_nonzero = sigma_chunk[nonzero_chunk_mask]
        mean_nonzero_target = mean_nonzero_chunk[nonzero_chunk_mask]
        mu_nonzero = np.log(mean_nonzero_target) - 0.5 * sigma_nonzero**2
        sampled_values = np.exp(np.random.normal(mu_nonzero, sigma_nonzero))
        sampled_values = np.maximum(1.0, np.rint(sampled_values))
        chunk_output = np.zeros(delta_chunk.size, dtype=np.float64)
        chunk_output[nonzero_chunk_mask] = sampled_values
        Nhalo[start:end] = chunk_output

    # Save calibration statistics.
    fit_data = np.column_stack((delta_per_bin, sigma_per_bin, n_nonzero_per_bin, min_halo_per_bin, max_halo_per_bin, mean_halo_nonzero_per_bin, prob_zero, total_halos_per_bin, n_total_per_bin))
    np.savetxt(logfile_dir / f"logfile_fits_{args.calibration}_{redshift:.3f}.txt", fit_data, fmt="%.6e", header="delta sigma N_nonzero minimum_halo maximum_halo mean_nonzero_halo prob_zero total_Nhalos counts_total")

    smooth_data = np.column_stack((reliable_delta, reliable_prob_zero, smoothed_prob_zero, reliable_mean_nonzero, smoothed_mean_nonzero, reliable_sigma, smoothed_sigma, reliable_counts, reliable_nonzero_counts))
    np.savetxt(diagnostics_dir / f"smoothed_model_{args.calibration}_z{redshift:.3f}.txt", smooth_data, fmt="%.6e", header="delta prob_zero_raw prob_zero_smooth mean_nonzero_raw mean_nonzero_smooth sigma_raw sigma_smooth Ntotal Nnonzero")

    # Physical abundance checks.
    total_small_all_realizations = np.sum(halo_num)
    total_small_per_physical_box = total_small_all_realizations / N_CALIBRATION_REALIZATIONS
    total_large = np.sum(Nhalo)
    volume_small = SMALL_BOX_SIZE**3
    effective_calibration_volume = volume_small * N_CALIBRATION_REALIZATIONS
    volume_large = LARGE_BOX_SIZE**3
    halo_density_small = total_small_all_realizations / effective_calibration_volume
    halo_density_large = total_large / volume_large
    density_ratio = halo_density_large / halo_density_small if halo_density_small > 0 else np.nan
    expected_large_from_volume = halo_density_small * volume_large
    density_weighted_ratio = total_large / expected_density_weighted_total if expected_density_weighted_total > 0 else np.nan

    print()
    print("Halo abundance check")
    print(f"Calibration mode: {args.calibration}")
    print(f"Calibration realizations: {N_CALIBRATION_REALIZATIONS}")
    print(f"Small-box halos summed over calibration rows: {total_small_all_realizations:.0f}")
    print(f"Small-box halos per physical box: {total_small_per_physical_box:.3f}")
    print(f"Large-box implemented halos: {total_large:.0f}")
    print(f"Large-box halos expected from simple volume scaling: {expected_large_from_volume:.6e}")
    print(f"Large-box halos expected from smoothed density-conditioned model: {expected_density_weighted_total:.6e}")
    print(f"Implemented / density-conditioned expected: {density_weighted_ratio:.6f}")
    print(f"Small-box halo number density: {halo_density_small:.6e} Mpc^-3")
    print(f"Large-box halo number density: {halo_density_large:.6e} Mpc^-3")
    print(f"n_large / n_small: {density_ratio:.6f}")
    print(f"Percentage difference in number density: {100.0 * (density_ratio - 1.0):+.3f}%")

    number_density_ratio_history.append(density_ratio)
    redshift_history.append(redshift)

    # Compare implemented and training Nhalo-delta distributions.
    mock_mask = Nhalo > 0
    training_mask = halo_num > 0

    if np.any(mock_mask) and np.any(training_mask):
        x1 = np.log10(delta_large[mock_mask] + 1.0)
        y1 = np.log10(Nhalo[mock_mask])
        x2 = np.log10(overdense_small[training_mask])
        y2 = np.log10(halo_num[training_mask])
        heatmap1, xedges1, yedges1 = np.histogram2d(x1, y1, bins=100)
        heatmap2, xedges2, yedges2 = np.histogram2d(x2, y2, bins=100)
        positive1 = heatmap1[heatmap1 > 0]
        positive2 = heatmap2[heatmap2 > 0]

        if positive1.size > 0 and positive2.size > 0:
            combined_max = max(np.max(heatmap1), np.max(heatmap2))
            combined_min = max(1.0, min(np.min(positive1), np.min(positive2)))
            norm = LogNorm(vmin=combined_min, vmax=combined_max)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            im1 = ax1.imshow(heatmap1.T, origin="lower", cmap="viridis", aspect="auto", extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]], norm=norm)
            im2 = ax2.imshow(heatmap2.T, origin="lower", cmap="viridis", aspect="auto", extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], norm=norm)
            ax1.set_title("Implemented mock catalogue")
            ax2.set_title("High-resolution simulation")
            ax1.set_xlabel(r"$\log_{10}(\delta + 1)$")
            ax2.set_xlabel(r"$\log_{10}(\delta + 1)$")
            ax1.set_ylabel(r"$\log_{10}(N_{\rm halo})$")
            shared_ymin = min(yedges1[0], yedges2[0])
            shared_ymax = max(yedges1[-1], yedges2[-1])
            shared_xmin = min(xedges1[0], xedges2[0])
            shared_xmax = max(xedges1[-1], xedges2[-1])
            ax1.set_ylim(shared_ymin, shared_ymax)
            ax2.set_ylim(shared_ymin, shared_ymax)
            ax1.set_xlim(shared_xmin, shared_xmax)
            ax2.set_xlim(shared_xmin, shared_xmax)
            fig.colorbar(im2, ax=[ax1, ax2], label="Counts")
            plt.savefig(results_dir / f"scatterImp_N5_8_14_{args.calibration}_z{redshift:.3f}.png")
            plt.close(fig)

    np.save(results_dir / f"halo_num_{args.calibration}_z{redshift:.3f}.npy", Nhalo)

history = np.column_stack((redshift_history, number_density_ratio_history))
np.savetxt(f"number_density_ratio_{args.calibration}.txt", history, fmt="%.6e", header="redshift n_large_over_n_small")

plt.figure()
plt.plot(redshift_history, number_density_ratio_history)
plt.xlabel(r"$z$")
plt.ylabel(r"$n_{\rm halo,large}/n_{\rm halo,small}$")
plt.tight_layout()
plt.savefig(f"number_density_ratio_{args.calibration}.png")
plt.close()