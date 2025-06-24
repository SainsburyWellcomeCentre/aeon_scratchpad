import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from movement.filtering import interpolate_over_time
from movement.io.load_poses import from_numpy
from scipy.signal import welch
from tqdm.notebook import tqdm


def pose_df_to_movement_ds(pose: pd.DataFrame) -> xr.Dataset:
    """
    Convert a DataFrame containing pose data to a `movement` poses dataset.

    Args:
        pose (pandas.DataFrame): DataFrame containing pose data with columns
            ['time', 'part', 'identity', 'x', 'y', 'part_likelihood'].

    Returns:
        xarray.Dataset: movement-compatible xarray.Dataset containing pose estimation data.
    """
    # Convert index to seconds and infer FPS
    pose = pose.copy()
    timestamps = pose.index.drop_duplicates().to_series()
    pose.index = (pose.index - pose.index[0]).total_seconds()
    frames = pose.index.drop_duplicates().to_numpy()
    mean_time_diff = np.diff(frames).mean()
    fps = float(np.ceil(1 / mean_time_diff)) if mean_time_diff > 0 else None
    keypoints = pose["part"].unique()
    individuals = pose["identity"].unique()
    # Create position array
    multi_idx = pd.MultiIndex.from_product(
        [frames, keypoints, individuals], names=["frame", "part", "identity"]
    )
    pose_reset = pose.reset_index().set_index(["time", "part", "identity"])
    pose_full = pose_reset.reindex(multi_idx)
    x_arr = pose_full["x"].values.reshape(len(frames), len(keypoints), len(individuals))
    y_arr = pose_full["y"].values.reshape(len(frames), len(keypoints), len(individuals))
    pose_arr = np.stack([x_arr, y_arr], axis=1).astype(float)
    # Create confidence array
    conf_arr = pose_full["part_likelihood"].values.reshape(
        len(frames), len(keypoints), len(individuals)
    )
    ds = from_numpy(
        position_array=pose_arr,
        confidence_array=conf_arr,
        individual_names=individuals,
        keypoint_names=keypoints,
        fps=fps,
        source_software="SLEAP",
    )
    ds = ds.assign_coords({"time": timestamps, "seconds_elapsed": ("time", frames)})
    ds.attrs["time_unit"] = timestamps.dtype.name
    return ds


def plot_confidence_hist(ds: xr.Dataset, vlines: list = None) -> None:
    """
    Plot histograms of confidence values for each keypoint in the dataset.

    Args:
        ds (xarray.Dataset): The dataset containing keypoint positions and confidence values.
        vlines (list, optional): A list of x-values at which to draw vertical lines (e.g. to
            visualise thresholds). Default is None.
    """
    n_keypoints = ds.sizes["keypoints"]
    # Create subplots for each keypoint
    fig, axes = plt.subplots(
        nrows=2,
        ncols=(n_keypoints + 1) // 2,
        figsize=(n_keypoints * 1.5, n_keypoints * 0.75),
        sharey=True,
        sharex=True,
    )
    for i, kpt in enumerate(ds.keypoints.values):
        ax = axes[i % 2, i // 2]
        for j, ind in enumerate(ds.individuals.values):
            ds.confidence.sel(keypoints=kpt, individuals=ind).plot.hist(
                bins=20,
                ax=ax,
                label=ind,
                histtype="stepfilled",
                fill=False,
                edgecolor=f"C{j}",  # Use matplotlib color cycle for edge
                density=True,
            )
        # Add vertical lines if specified
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color="k", linestyle="--", linewidth=1)
        ax.set_ylabel("Density")
        ax.set_xlabel("")
        ax.set_xlabel("Confidence")
        ax.set_title(kpt)
        ax.legend()
    plt.suptitle("Confidence Histograms by Keypoint and Individual")
    plt.tight_layout()


def plot_speed(da: xr.DataArray) -> None:
    """
    Plot speed over time and a histogram of the speed values.

    This function creates a line plot of speed over time for each individual
    in the dataset, along with a histogram of the speed values.

    Args:
        da (xarray.DataArray): The data array containing speed data.

    """
    individuals = da.individuals.values.tolist() if "individuals" in da.dims else [None]
    n_individuals = len(individuals)
    fig, axes = plt.subplots(
        n_individuals,
        2,
        figsize=(12, 4 * n_individuals),
        squeeze=False,
        gridspec_kw={"width_ratios": [4, 1]},
    )
    for i, ind in enumerate(individuals):
        if ind is not None:
            da_ind = da.sel(individuals=ind)
            label = ind
        else:
            da_ind = da
            label = None
        ax, ax_hist = axes[i]
        keypoints_label = (
            da_ind.keypoints.values if "keypoints" in da_ind.dims else None
        )
        da_ind.plot.line(x="time", lw=0.5, ax=ax, label=keypoints_label)
        ax.set_title(f"Speed vs time: {label}")
        ax.set_ylabel("Speed (cm/s)")
        ax.set_xlabel("Datetime")
        ax.set_xlim(
            da_ind.time.min().values,
            da_ind.time.max().values,
        )
        if keypoints_label is not None:
            ax.legend()
        # Plot histogram of speed
        da_ind.plot.hist(bins=25, orientation="horizontal", ax=ax_hist)
        ax_hist.set_title("Histogram")
        ax_hist.set_xscale("log")
        ax_hist.set_xlabel("log count")
    plt.tight_layout()


def plot_raw_and_smooth_timeseries_and_psd(
    da_raw: xr.DataArray,
    da_smooth: xr.DataArray,
    fps: float,
    individual: str = None,
    keypoint: str = None,
    space: str = "x",
    time_range: slice = None,
) -> None:
    """
    Plot raw and smoothed time series of a keypoint's position
    and their Power Spectral Density (PSD).

    Args:
        da_raw (xarray.DataArray): The data array containing raw keypoint positions.
        da_smooth (xarray.DataArray): The data array containing smoothed keypoint positions.
        fps (float): Frames per second of the data.
        individual (str, optional): The individual to plot. If None, the first individual is used.
        keypoint (str, optional): The keypoint to plot. If None, the first keypoint is used.
        space (str, optional): The spatial dimension to plot (e.g., "x" or "y"). Default is "x".
        time_range (slice, optional): The time range to plot. If None, the entire time series is plotted.
    """

    # If no time range is specified, plot the entire time series
    if time_range is None:
        time_range = slice(0, da_raw.time[-1])
    if individual is None:
        individual = da_raw.individuals.values[0]
    if keypoint is None:
        keypoint = da_raw.keypoints.values[0]
    selection = {
        "time": time_range,
        "individuals": individual,
        "keypoints": keypoint,
        "space": space,
    }
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for da, color, label in zip(
        [da_raw, da_smooth], ["k", "r"], ["raw", "smooth"], strict=False
    ):
        # plot position time series
        pos = da.sel(**selection)
        ax[0].plot(
            pos.time,
            pos,
            color=color,
            lw=2,
            alpha=0.7,
            label=f"{label} {space}",
        )
        # interpolate data to remove NaNs in the PSD calculation
        pos_interp = interpolate_over_time(pos, fill_value="extrapolate")
        # compute and plot the PSD
        freq, psd = welch(pos_interp, fs=fps, nperseg=256)
        ax[1].semilogy(
            freq,
            psd,
            color=color,
            lw=2,
            alpha=0.7,
            label=f"{label} {space}",
        )
    ax[0].set_ylabel(f"{space} position (px)")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_title("Time Domain")
    ax[0].legend()
    ax[1].set_ylabel("PSD (px$^2$/Hz)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_title("Frequency Domain")
    ax[1].legend()
    plt.suptitle(f"Keypoint: {keypoint}, Individual: {individual}, Space: {space}")
    plt.tight_layout()


def plot_polar_histogram(da, bin_width_deg=15, ax=None) -> tuple:
    """
    Plots a polar histogram of the data in the given DataArray.

    Args:
        da (xarray.DataArray): DataArray containing angle data in radians.
        bin_width_deg (int, optional): Width of the bins in degrees. Defaults to 15.
        ax (matplotlib.axes.Axes, optional): Axes on which to plot the histogram.
            If None (default), a new figure and axes are created.

    Returns:
        tuple: (fig, ax) where fig is the matplotlib Figure and ax is the polar Axes.
    """
    n_bins = int(360 / bin_width_deg)
    if ax is None:
        fig, ax = plt.subplots(  # initialise figure with polar projection
            1, 1, figsize=(5, 5), subplot_kw={"projection": "polar"}
        )
    else:
        fig = ax.figure  # or use the provided axes
    # plot histogram using xarray's built-in histogram function
    da.plot.hist(bins=np.linspace(-np.pi, np.pi, n_bins + 1), ax=ax, density=True)
    # axes settings
    ax.set_theta_direction(-1)  # theta increases in clockwise direction
    ax.set_theta_offset(0)  # set zero at the right
    ax.set_xlabel("")  # remove default x-label from xarray's plot.hist()
    # set xticks to match the phi values in degrees
    n_xtick_edges = 9
    ax.set_xticks(np.linspace(0, 2 * np.pi, n_xtick_edges)[:-1])
    xticks_in_deg = list(range(0, 180 + 45, 45)) + list(range(0, -180, -45))[-1:0:-1]
    ax.set_xticklabels([str(t) + "\N{DEGREE SIGN}" for t in xticks_in_deg])
    return fig, ax


def shade_mask_regions(ax, mask, color="red", alpha=0.15):
    """Shades contiguous True regions in a boolean mask on the given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to shade the regions.
        mask (xarray.DataArray): DataArray with a 'time' coordinate,
            indicating regions to shade (True values).
        color (str, optional): Color to use for shading. Defaults to "red".
        alpha (float, optional): Transparency level for shading. Defaults to 0.15.

    """
    times = mask.time.values
    mask_values = mask.values
    in_region = False
    region_start = None
    for idx, val in enumerate(mask_values):
        if val and not in_region:
            in_region = True
            region_start = times[idx]
        elif not val and in_region:
            in_region = False
            region_end = times[idx]
            ax.axvspan(region_start, region_end, color=color, alpha=alpha)
    if in_region:
        ax.axvspan(region_start, times[-1], color=color, alpha=alpha)


def filter_valid_points(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Filter points in the DataFrame that are within the arena or nest region.

    Args:
        df (pandas.DataFrame): DataFrame containing tracking data with columns 'x' and 'y'.
        region_df (pandas.DataFrame): DataFrame containing arena region data with columns:
            - region_name: name of the region (e.g., 'ArenaCenter', 'ArenaOuterRadius', 'NestRegion')
            - region_data: dictionary containing region-specific data

    Returns:
        pandas.DataFrame: DataFrame with only valid (in-arena or in-nest) points

    """

    def get_region_value(region_name):
        mask = region_df.index.get_level_values("region_name") == region_name
        if mask.any():
            return region_df.loc[mask, "region_data"].iloc[0]
        return None

    # --- Extract arena and nest geometry ---
    arena_center = get_region_value("ArenaCenter")
    arena_outer_radius = get_region_value("ArenaOuterRadius")
    nest_region = get_region_value("NestRegion")

    if arena_center is None or arena_outer_radius is None:
        raise ValueError(
            "Could not find ArenaCenter or ArenaOuterRadius in region data"
        )

    center_x = float(arena_center["X"])
    center_y = float(arena_center["Y"])
    outer_radius = float(arena_outer_radius) + 10  # extra padding

    coords = df[["x", "y"]].to_numpy()
    dx = coords[:, 0] - center_x
    dy = coords[:, 1] - center_y
    dist2 = dx**2 + dy**2
    inside_arena = dist2 <= outer_radius**2

    inside_nest = np.zeros(len(df), dtype=bool)
    if nest_region and "ArrayOfPoint" in nest_region:
        nest_coords = np.array(
            [(float(p["X"]), float(p["Y"])) for p in nest_region["ArrayOfPoint"]]
        )
        if len(nest_coords) > 0:
            x_min, x_max = nest_coords[:, 0].min() - 10, nest_coords[:, 0].max() + 10
            y_min, y_max = nest_coords[:, 1].min() - 10, nest_coords[:, 1].max() + 10
            inside_nest = (
                (coords[:, 0] >= x_min)
                & (coords[:, 0] <= x_max)
                & (coords[:, 1] >= y_min)
                & (coords[:, 1] <= y_max)
            )

    return df.loc[inside_arena | inside_nest]


def clean_swaps_refactor(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Swap correction for dual mouse tracking.

    Filters out-of-bounds points first,
    then does identity assignment with majority voting to fix track swaps.

    - Pre-cleaning: removes points outside arena/nest bounds
    - Early frames: filled with raw data before first complete observation
    - Swap detection: uses frame-to-frame distance minimization
    - Identity correction: SLEAP-style majority vote within continuous segments of cleaning
    - Swap marking: sets identity_likelihood=NaN on locally swapped frames

    Args:
        df (pandas.DataFrame): DataFrame containing tracking data with columns 'x' and 'y'.
        region_df (pandas.DataFrame): DataFrame containing arena region data with columns:
            - region_name: name of the region (e.g., 'ArenaCenter', 'ArenaOuterRadius', 'NestRegion')
            - region_data: dictionary containing region-specific data

    Returns:
        pandas.DataFrame: DataFrame with cleaned tracking data having the same structure as
            the input but corrected x,y coords and identity_likelihood=NaN on locally swapped frames

    Examples:
        To retrieve region_df from the database:

        >>> active_region_query = acquisition.EpochConfig.ActiveRegion & (acquisition.Chunk & chunk_key)
        >>> region_df = active_region_query.fetch(format="frame")

    """
    # Select only points within the arena or nest region
    df = filter_valid_points(df, region_df)

    # Swap correction
    # 1) setup data for processing
    df = df.sort_index()
    ids = sorted(df["identity_name"].unique())  # enforce order
    if len(ids) != 2:
        raise ValueError("Expected exactly two identities, found: {}".format(ids))

    # 2) Prep for merge later
    df2 = df.reset_index()
    time_col = df2.columns[0]  # timestamp column name

    # 3) Reshape to 2×T arrays
    wide = df2.pivot(index=time_col, columns="identity_name", values=["x", "y"])
    times = wide.index.values
    T = len(times)
    x_raw = wide["x"][ids].to_numpy().T
    y_raw = wide["y"][ids].to_numpy().T

    # 4) Init cleaned arrays + swap tracking
    x_clean = np.full_like(x_raw, np.nan)
    y_clean = np.full_like(y_raw, np.nan)
    swapped_flags = np.zeros(T, dtype=bool)

    # 5) Find first complete frame and keep raw data before it
    valid = np.isfinite(x_raw).all(axis=0)
    if not valid.any():
        raise RuntimeError("No frame with both subjects present")
    first_i = np.argmax(valid)
    if first_i > 0:
        x_clean[:, :first_i] = x_raw[:, :first_i]
        y_clean[:, :first_i] = y_raw[:, :first_i]

    # 6) Local-segment tracking
    # Initialize on first full-detect frame
    seg_start = first_i
    votes_same = 1
    votes_swap = 0
    last_x = x_raw[:, first_i].copy()
    last_y = y_raw[:, first_i].copy()
    x_clean[:, first_i] = last_x
    y_clean[:, first_i] = last_y

    # --- Helper functions ---
    def _flush_segment(start, end, votes_same, votes_swap):
        """Flush the current segment and apply local vote."""
        if votes_swap > votes_same:
            x_clean[:, start:end] = x_clean[::-1, start:end]
            y_clean[:, start:end] = y_clean[::-1, start:end]
            swapped_flags[start:end] = ~swapped_flags[start:end]

    def _assign_single_detection(t, src_idx, dest_idx):
        """Assign values from src_idx to dest_idx at time t."""
        x_clean[dest_idx, t] = x_raw[src_idx, t]
        y_clean[dest_idx, t] = y_raw[src_idx, t]
        last_x[dest_idx] = x_raw[src_idx, t]
        last_y[dest_idx] = y_raw[src_idx, t]

    def _assign_full_frame(t, x_vals, y_vals):
        """Assign full frame values to both identities at time t."""
        x_clean[:, t] = x_vals
        y_clean[:, t] = y_vals
        last_x[:] = x_vals
        last_y[:] = y_vals

    def _update_votes(t):
        """Determine if a swap occurred at time t and update vote counts accordingly."""
        if np.allclose(x_raw[:, t], x_clean[:, t], equal_nan=True) and np.allclose(
            y_raw[:, t], y_clean[:, t], equal_nan=True
        ):
            nonlocal votes_same
            votes_same += 1
        else:
            nonlocal votes_swap
            votes_swap += 1

    for t in tqdm(range(first_i + 1, T), desc="Cleaning frames"):
        present = np.isfinite(x_raw[:, t])
        n_det = present.sum()

        # zero detections → drop both
        if n_det == 0:
            _flush_segment(seg_start, t, votes_same, votes_swap)
            seg_start = t
            votes_same = votes_swap = 0
            continue

        # 1 detection → assign to closest
        if n_det == 1:
            src_idx = np.where(present)[0][0]
            dist_to_0 = np.hypot(
                x_raw[src_idx, t] - last_x[0], y_raw[src_idx, t] - last_y[0]
            )
            dist_to_1 = np.hypot(
                x_raw[src_idx, t] - last_x[1], y_raw[src_idx, t] - last_y[1]
            )
            if min(dist_to_0, dist_to_1) <= 90:
                dest_idx = 0 if dist_to_0 <= dist_to_1 else 1
                _assign_single_detection(t, src_idx, dest_idx)
                _update_votes(t)
            continue

        # 2 detections
        # compute distances and assignment costs
        inter_d = np.hypot(x_raw[0, t] - x_raw[1, t], y_raw[0, t] - y_raw[1, t])
        dx = x_raw[:, t][:, None] - last_x[None, :]
        dy = y_raw[:, t][:, None] - last_y[None, :]
        dist_mat = np.hypot(dx, dy)
        cost_same = dist_mat[0, 0] + dist_mat[1, 1]
        cost_swap = dist_mat[0, 1] + dist_mat[1, 0]
        min_dist_id0 = dist_mat[0].min()
        min_dist_id1 = dist_mat[1].min()

        # Define discontinuity conditions
        break_too_close = inter_d < 100  # two detections are too close
        break_too_costly = min(cost_same, cost_swap) > 90
        break_both_far = min(min_dist_id0, min_dist_id1) > 90  # both assignments >90px

        if break_too_close or (break_too_costly and break_both_far):
            _flush_segment(seg_start, t, votes_same, votes_swap)
            seg_start = t
            votes_same = votes_swap = 0

        # Re-evaluate after reset
        if break_too_close:  # keep original assignment
            _assign_full_frame(t, x_raw[:, t], y_raw[:, t])
            votes_same += 1
            continue

        # Re-evaluate after reset
        if break_too_costly:
            # If both assignments are too far, skip
            if break_both_far:
                continue
            dest_idx = 0 if min_dist_id0 <= min_dist_id1 else 1
            src_idx = int(np.argmin(dist_mat[dest_idx]))
            _assign_single_detection(t, src_idx, dest_idx)
            _update_votes(t)
            continue

        if cost_same <= cost_swap:
            x_vals = x_raw[:, t]
            y_vals = y_raw[:, t]
            votes_same += 1
        else:
            x_vals = x_raw[::-1, t]
            y_vals = y_raw[::-1, t]
            swapped_flags[t] = True
            votes_swap += 1
        _assign_full_frame(t, x_vals, y_vals)

    # 7) Flush the final segment
    _flush_segment(seg_start, T, votes_same, votes_swap)

    # 8) Build cleaned DataFrame
    cleaned = pd.DataFrame(
        {
            time_col: np.repeat(times, 2),
            "identity_name": np.tile(ids, T),
            "x": x_clean.ravel(order="F"),
            "y": y_clean.ravel(order="F"),
        }
    )

    # 9) Merge back with original data (drop old x, y)
    df2_noxy = df2.drop(columns=["x", "y"])
    result = (
        df2_noxy.merge(cleaned, on=[time_col, "identity_name"], how="right")
        .set_index(time_col)
        .sort_index()
    )

    # 10) Mark swapped frames in likelihood
    if "identity_likelihood" in result.columns:
        mask = result.index.isin(times[swapped_flags])
        result.loc[mask, "identity_likelihood"] = np.nan

    # 11) Final cleanup: drop any rows where x or y is NaN
    result = result.dropna(subset=["x", "y"], how="any")

    return result
