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

    Parameters
    ----------
    pose : pd.DataFrame
        DataFrame containing pose data with columns ['time', 'part', 'identity', 'x', 'y', 'part_likelihood'].

    Returns
    -------
    xarray.Dataset
        movement-compatible xarray.Dataset containing pose estimation data.
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

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing keypoint positions and confidence values.
    vlines : list, optional
        List of x-values at which to draw vertical lines (e.g. to
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

    Parameters
    ----------
    da: xr.DataArray
        The data array containing speed data.

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

    Parameters
    ----------
    da_raw : xr.DataArray
        The data array containing raw keypoint positions.
    da_smooth : xr.DataArray
        The data array containing smoothed keypoint positions.
    fps : float
        The frames per second of the data.
    individual : str, optional
        The individual to plot. If None, the first individual is used.
    keypoint : str, optional
        The keypoint to plot. If None, the first keypoint is used.
    space : str, optional
        The spatial dimension to plot (e.g., "x" or "y"). Default is "x".
    time_range : slice, optional
        The time range to plot. If None, the entire time series is plotted.
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


def plot_polar_histogram(da, bin_width_deg=15, ax=None):
    """Plot a polar histogram of the data in the given DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        A DataArray containing angle data in radians.
    bin_width_deg : int, optional
        Width of the bins in degrees.
    ax : matplotlib.axes.Axes, optional
        The axes on which to plot the histogram.

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


def clean_swaps2(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Swap correction for dual mouse tracking.

    Filters out-of-bounds points first,
    then does identity assignment with majority voting to fix track swaps.

    - Pre-cleaning: removes points outside arena/nest bounds
    - Early frames: filled with raw data before first complete observation
    - Swap detection: uses frame-to-frame distance minimization
    - Identity correction: SLEAP-style majority vote at the end
    - Swap marking: sets identity_likelihood=NaN on locally swapped frames

    Parameters:
    - df: DataFrame with tracking data (x, y, identity_name columns)
    - region_df: DataFrame containing arena region data with columns:
        - region_name: name of the region (e.g., 'ArenaCenter', 'ArenaOuterRadius', 'NestRegion')
        - region_data: dictionary containing region-specific data

    Returns:
    - DataFrame with cleaned tracking data, same structure as input but corrected x,y coords
      and identity_likelihood=NaN on locally swapped frames

    Example:
        To retrieve region_df from the database:
        ```python
        active_region_query = acquisition.EpochConfig.ActiveRegion & (acquisition.Chunk & chunk_key)
        region_df = active_region_query.fetch(format="frame")
        ```
    """

    # Helper to extract region values
    def get_region_value(region_name):
        mask = region_df.index.get_level_values("region_name") == region_name
        if mask.any():
            return region_df.loc[mask, "region_data"].iloc[0]
        return None

    # Parse arena geometry
    arena_center = get_region_value("ArenaCenter")
    arena_outer_radius = get_region_value("ArenaOuterRadius")
    nest_region = get_region_value("NestRegion")

    if arena_center is None or arena_outer_radius is None:
        raise ValueError(
            "Could not find ArenaCenter or ArenaOuterRadius in region data"
        )

    # Coords + radius
    center_x = float(arena_center["X"])
    center_y = float(arena_center["Y"])
    outer_radius = float(arena_outer_radius) + 10

    # Nest boundary points
    nest_x_coords = []
    nest_y_coords = []
    if nest_region and "ArrayOfPoint" in nest_region:
        for point in nest_region["ArrayOfPoint"]:
            nest_x_coords.append(float(point["X"]))
            nest_y_coords.append(float(point["Y"]))

    # Filter out-of-bounds points before cleaning
    if len(nest_x_coords) > 0:
        # Arena circle check
        dist2 = (df["x"] - center_x) ** 2 + (df["y"] - center_y) ** 2
        inside_arena = dist2 <= outer_radius**2

        # Nest bounding box with padding
        nx_min, nx_max = min(nest_x_coords) - 10, max(nest_x_coords) + 10
        ny_min, ny_max = min(nest_y_coords) - 10, max(nest_y_coords) + 10
        inside_nest = df["x"].between(nx_min, nx_max) & df["y"].between(ny_min, ny_max)

        # Keep arena OR nest points
        df = df[inside_arena | inside_nest]
    else:
        # Arena only if no nest
        dist2 = (df["x"] - center_x) ** 2 + (df["y"] - center_y) ** 2
        inside_arena = dist2 <= outer_radius**2
        df = df[inside_arena]

    # Swap correction starts here
    # 1) setup data for processing
    df = df.sort_index()
    ids = df["identity_name"].unique()
    if len(ids) != 2:
        raise ValueError("Expected exactly two identities, found: {}".format(ids))

    # 2) Prep for merge later
    df2 = df.copy().reset_index()
    time_col = df2.columns[0]  # timestamp column name

    # 3) Reshape to 2×T arrays
    times = df.index.unique().values
    T = len(times)
    x_raw = (
        df.set_index("identity_name", append=True)["x"]
        .unstack("identity_name")
        .to_numpy()
        .T
    )
    y_raw = (
        df.set_index("identity_name", append=True)["y"]
        .unstack("identity_name")
        .to_numpy()
        .T
    )

    # 4) Init cleaned arrays + swap tracking
    x_clean = np.full_like(x_raw, np.nan)
    y_clean = np.full_like(y_raw, np.nan)
    swapped_flags = np.zeros(T, dtype=bool)  # Track local swaps

    # 5) Find first complete frame
    valid = np.isfinite(x_raw).all(axis=0)
    first_i = np.argmax(valid)

    # Fill early frames with raw data
    if first_i > 0:
        x_clean[:, :first_i] = x_raw[:, :first_i]
        y_clean[:, :first_i] = y_raw[:, :first_i]

    # Start tracking from first complete frame
    last_x = x_raw[:, first_i].copy()
    last_y = y_raw[:, first_i].copy()
    x_clean[:, first_i] = last_x
    y_clean[:, first_i] = last_y

    # Vote counter for final identity assignment
    track_votes = np.zeros((2, 2), dtype=np.int64)

    # Count first frame if valid
    if valid[first_i]:
        track_votes[0, 0] += 1
        track_votes[1, 1] += 1

    # 6) Main loop: swap correction + voting
    for t in tqdm(range(first_i + 1, T), desc="Cleaning frames"):
        # Missing data: carry forward last known positions
        # if any of the x_raw or y_raw values are NaN
        # fill the corresponding x_clean and y_clean
        # with last known positions, otherwise keep original values
        if np.isnan(x_raw[:, t]).any():
            for k in (0, 1):
                if np.isnan(x_raw[k, t]):
                    x_clean[k, t] = last_x[k]
                    y_clean[k, t] = last_y[k]
                else:
                    x_clean[k, t] = x_raw[k, t]
                    y_clean[k, t] = y_raw[k, t]
            continue

        # Mice too close: keep original assignment
        inter_mouse_dist = np.hypot(
            x_raw[0, t] - x_raw[1, t], y_raw[0, t] - y_raw[1, t]
        )
        if inter_mouse_dist < 115:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            track_votes[0, 0] += 1
            track_votes[1, 1] += 1
            last_x = x_raw[:, t].copy()
            last_y = y_raw[:, t].copy()
            continue

        # Distance-based swap decision
        d_same = np.sum(np.hypot(x_raw[:, t] - last_x, y_raw[:, t] - last_y))
        d_swap = np.sum(np.hypot(x_raw[::-1, t] - last_x, y_raw[::-1, t] - last_y))

        # Too far even with best assignment: carry forward
        if min(d_same, d_swap) > 90:
            x_clean[:, t] = last_x
            y_clean[:, t] = last_y
            continue

        # Assign based on shortest total distance
        if d_same <= d_swap:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            track_votes[0, 0] += 1
            track_votes[1, 1] += 1
        else:
            x_clean[:, t] = x_raw[::-1, t]
            y_clean[:, t] = y_raw[::-1, t]
            track_votes[0, 1] += 1
            track_votes[1, 0] += 1
            swapped_flags[t] = True  # Mark local swap

        # Update reference for next frame
        last_x = x_clean[:, t].copy()
        last_y = y_clean[:, t].copy()

    # 7) Global identity correction via majority vote
    need_swap = track_votes[0, 1] > track_votes[0, 0]
    if need_swap:
        x_clean = x_clean[::-1, :]
        y_clean = y_clean[::-1, :]
        # Flip swap flags too
        swapped_flags = swapped_flags[::-1]

    # 8) Build output dataframe
    cleaned = pd.DataFrame(
        {
            time_col: np.repeat(times, 2),
            "identity_name": np.tile(ids, T),
            "x": x_clean.ravel(order="F"),
            "y": y_clean.ravel(order="F"),
        }
    )

    # 9) Merge back with original data (drop old x,y)
    df2_noxy = df2.drop(columns=["x", "y"])
    result = (
        df2_noxy.merge(cleaned, on=[time_col, "identity_name"], how="left")
        .set_index(time_col)
        .sort_index()
    )

    # 10) Mark swapped frames in likelihood
    if "identity_likelihood" in result.columns:
        mask = result.index.isin(times[swapped_flags])
        result.loc[mask, "identity_likelihood"] = np.nan

    return result


def shade_mask_regions(ax, mask, color="red", alpha=0.15):
    """
    Shade contiguous True regions in mask on the given axis.
    mask: xarray DataArray or 1D boolean array with a 'time' coordinate.
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
