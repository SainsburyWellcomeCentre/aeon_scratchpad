"""Timing checks for ONIX electrophysiology data streams."""

import datetime
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike
from swc.aeon.io.api import chunk, load, to_datetime, to_seconds
from swc.aeon.io.reader import Binary, HarpSyncAlignment

from aeon_qc.reader import NOMINAL_CLOCK_HZ, HarpSync
from aeon_qc.schemas import is_epoch_dir, parse_epoch_timestamp

INTEGRITY_COLS = ("kind", "harp_step", "clock_step_ticks", "deviation_ticks", "device")
DRIFT_COLS = ("residual_seconds", "chunk", "device")
CLOCK_COLS = ("kind", "clock_ticks", "step_ticks", "step_seconds", "file", "index_in_file", "device")
HUB_COLS = (
    "file",
    "n_samples",
    "min_offset_ticks",
    "mean_offset_ticks",
    "max_offset_ticks",
    "length_mismatch",
    "device",
)

OK = "ok"
MIN_SYNC_EVENTS = 2
"""Minimum HarpSync rows for a linear fit."""

HALF_SAMPLE = 0.5
"""A clock step more than half a nominal step away from it is not exactly one sample."""

WORST_N = 10
"""Number of largest deviations kept, with their locations, by each check."""

MAX_CLOCK_EVENTS = 10_000
"""Cap on rows returned by ``onix_clock_sequence`` so a corrupt file cannot exhaust memory."""

BLOCK_SAMPLES = 2**24
"""Samples read per block from a clock file (128 MB of uint64)."""

HISTOGRAM_BUCKETS = 65
"""Power-of-two magnitude buckets: 0, then [2**(k-1), 2**k) ticks for k = 1..64."""


def empty_result(columns: tuple[str, ...], metric: str) -> pd.DataFrame:
    """Return an empty metric frame with the standard UTC ``time`` index and a metric attr."""
    result = pd.DataFrame(
        columns=list(columns),
        index=pd.DatetimeIndex([], name="time", tz=datetime.UTC),
    )
    result.attrs["metric"] = metric
    result.attrs["data_found"] = False
    return result


def magnitude_histogram(values: np.ndarray) -> np.ndarray:
    """Count ``|values|`` in power-of-two buckets: 0, then ``[2**(k-1), 2**k)`` ticks."""
    magnitude = np.abs(values.astype(np.float64))
    bucket = np.zeros(magnitude.shape, dtype=np.int64)
    nonzero = magnitude >= 1
    bucket[nonzero] = np.floor(np.log2(magnitude[nonzero])).astype(np.int64) + 1
    return np.bincount(np.minimum(bucket, HISTOGRAM_BUCKETS - 1), minlength=HISTOGRAM_BUCKETS)


def keep_worst(worst: list[dict], candidates: list[dict], key: str, n: int = WORST_N) -> list[dict]:
    """Return the ``n`` entries of ``worst + candidates`` with the largest ``|entry[key]|``."""
    merged = worst + candidates
    merged.sort(key=lambda entry: abs(entry[key]), reverse=True)
    return merged[:n]


def harp_sync_integrity(
    root: str | PathLike | list[str] | list[PathLike],
    reader: HarpSync,
    start: datetime.datetime,
    end: datetime.datetime | None = None,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Measure every HarpSync step: the Harp second increment and the ONIX clock step.

    Args:
        root: Dataset root path or paths.
        reader: The ``HarpSync`` reader for one ONIX device.
        start: Left bound of the time range.
        end: Optional right bound of the time range.
        data: The HarpSync records, already loaded and sorted. Loaded here when not given.

    Returns:
        A DataFrame with one row per step between consecutive records, indexed by the
        later record's time.

        - kind (str): ``ok``, ``harp_time_gap``, ``harp_time_duplicate`` or ``harp_time_backwards``.
        - harp_step (int): Harp seconds advanced by the step.
        - clock_step_ticks (float): ONIX clock ticks advanced by the step.
        - deviation_ticks (float): Clock step minus the median step scaled by ``harp_step``.
          NaN when Harp time did not advance.
        - device (str): The reader pattern.

        ``attrs`` hold ``n_sync_events``, ``n_faults`` and per-kind counts, ``seconds_offset``
        (median of the ``Seconds`` column minus ``harp_time``: 1 for recordings whose workflow
        added the protocol second a second time, 0 otherwise), ``clock_step_median_ticks``,
        ``clock_step_ppm`` (deviation from the nominal 250 MHz), and the median, 99th
        percentile and maximum of ``|deviation_ticks|``.
    """
    if data is None:
        data = load(root, reader, start=start, end=end)
    result = empty_result(INTEGRITY_COLS, "harp_sync_integrity")
    result.attrs.update(
        {
            "data_found": not data.empty,
            "n_sync_events": len(data),
            "n_faults": 0,
            "seconds_offset": None,
            "clock_step_median_ticks": None,
            "clock_step_ppm": None,
            "deviation_median_abs_ticks": None,
            "deviation_p99_abs_ticks": None,
            "deviation_max_abs_ticks": None,
        }
    )
    if len(data) < MIN_SYNC_EVENTS:
        return result

    harp = data["harp_time"].to_numpy(dtype=np.float64)
    clock = data["clock"].to_numpy(dtype=np.float64)
    seconds = np.asarray(to_seconds(pd.DatetimeIndex(data.index)), dtype=np.float64)
    harp_step = np.diff(harp)
    clock_step = np.diff(clock)
    single = harp_step == 1
    median_step = float(np.median(clock_step[single] if single.any() else clock_step))
    advanced = harp_step > 0
    deviation = np.where(advanced, clock_step - median_step * harp_step, np.nan)
    kind = np.select(
        [harp_step > 1, harp_step == 0, harp_step < 0],
        ["harp_time_gap", "harp_time_duplicate", "harp_time_backwards"],
        default=OK,
    )

    result = pd.DataFrame(
        {
            "kind": kind,
            "harp_step": harp_step.astype(np.int64),
            "clock_step_ticks": clock_step,
            "deviation_ticks": deviation,
            "device": reader.pattern,
        },
        index=pd.DatetimeIndex(data.index[1:], name="time", tz=datetime.UTC),
    )
    magnitude = np.abs(deviation[advanced])
    counts = pd.Series(kind).value_counts()
    result.attrs.update(
        {
            "metric": "harp_sync_integrity",
            "data_found": True,
            "n_sync_events": len(data),
            "n_faults": int((kind != OK).sum()),
            "n_harp_time_gap": int(counts.get("harp_time_gap", 0)),
            "n_harp_time_duplicate": int(counts.get("harp_time_duplicate", 0)),
            "n_harp_time_backwards": int(counts.get("harp_time_backwards", 0)),
            "seconds_offset": float(np.median(seconds - harp)),
            "clock_step_median_ticks": median_step,
            "clock_step_ppm": (median_step / NOMINAL_CLOCK_HZ - 1.0) * 1e6,
            "deviation_median_abs_ticks": float(np.median(magnitude)) if magnitude.size else None,
            "deviation_p99_abs_ticks": float(np.percentile(magnitude, 99)) if magnitude.size else None,
            "deviation_max_abs_ticks": float(magnitude.max()) if magnitude.size else None,
        }
    )
    return result


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return ``(slope, intercept)`` of a mean-centred least-squares line through ``(x, y)``.

    The same fit as ``swc.aeon.io.reader.HarpSyncAlignment``. Centring keeps it well
    conditioned on the large clock and Harp magnitudes.
    """
    dx = x - x.mean()
    dy = y - y.mean()
    slope = float(np.dot(dx, dy) / np.dot(dx, dx))
    return slope, float(y.mean() - slope * x.mean())


def harp_sync_drift(
    root: str | PathLike | list[str] | list[PathLike],
    reader: HarpSync,
    start: datetime.datetime,
    end: datetime.datetime | None = None,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit Harp time on ONIX clock ticks per hourly chunk and keep every residual.

    Ephys data is placed on Harp time with one linear fit per hourly chunk, as
    ``swc.aeon.io.reader.HarpSyncAlignment`` does. The residual of every sync record
    against its chunk's fit is kept. A single fit over the whole window is also returned
    in ``attrs["fit"]`` for placing QC events on Harp time. The fits use ``harp_time``
    (``Value.HarpTime``). That column is the true Harp second in every recording regardless
    of the offset in the ``Seconds`` column (see ``aeon_qc.reader.HarpSync``).

    Args:
        root: Dataset root path or paths.
        reader: The ``HarpSync`` reader for one ONIX device.
        start: Left bound of the time range.
        end: Optional right bound of the time range.
        data: The HarpSync records, already loaded and sorted. Loaded here when not given.

    Returns:
        A DataFrame with one row per sync record, indexed by the record's time.

        - residual_seconds (float): Harp time minus its chunk's fitted time. NaN for a chunk
          with fewer than two records.
        - chunk (Timestamp): The hourly chunk the record was fitted in.
        - device (str): The reader pattern.

        ``attrs`` hold ``chunks`` (a DataFrame with one row per chunk: ``n_sync_events``,
        ``clock_rate_hz``, ``clock_rate_ppm`` and ``max_abs_residual_ms``), ``n_chunks``,
        ``worst_chunk_max_abs_residual_ms``, the whole-window fit's ``clock_rate_hz`` and
        ``clock_rate_ppm``, and ``fit`` as its ``(slope, intercept)`` mapping ticks to Harp
        seconds.
    """
    if data is None:
        data = load(root, reader, start=start, end=end)
    result = empty_result(DRIFT_COLS, "harp_sync_drift")
    result.attrs.update(
        {
            "data_found": not data.empty,
            "n_sync_events": len(data),
            "nominal_clock_hz": NOMINAL_CLOCK_HZ,
            "n_chunks": 0,
            "chunks": None,
            "worst_chunk_max_abs_residual_ms": None,
            "clock_rate_hz": None,
            "clock_rate_ppm": None,
            "fit": None,
        }
    )
    if len(data) < MIN_SYNC_EVENTS:
        return result

    clock = data["clock"].to_numpy(dtype=np.float64)
    harp = data["harp_time"].to_numpy(dtype=np.float64)
    chunks = chunk(pd.DatetimeIndex(data.index))
    residuals = np.full(len(data), np.nan)
    rows = []
    for key in chunks.unique():
        mask = np.asarray(chunks == key)
        row = {"chunk": key, "n_sync_events": int(mask.sum())}
        if mask.sum() >= MIN_SYNC_EVENTS:
            slope, intercept = fit_line(clock[mask], harp[mask])
            residuals[mask] = harp[mask] - (slope * clock[mask] + intercept)
            rate = 1.0 / slope
            row.update(
                clock_rate_hz=rate,
                clock_rate_ppm=(rate / NOMINAL_CLOCK_HZ - 1.0) * 1e6,
                max_abs_residual_ms=float(np.abs(residuals[mask]).max() * 1000),
            )
        rows.append(row)
    chunk_table = pd.DataFrame(rows).set_index("chunk")
    slope, intercept = fit_line(clock, harp)
    rate = 1.0 / slope

    result = pd.DataFrame(
        {"residual_seconds": residuals, "chunk": chunks, "device": reader.pattern},
        index=pd.DatetimeIndex(data.index, name="time", tz=datetime.UTC),
    )
    worst = chunk_table["max_abs_residual_ms"].max() if "max_abs_residual_ms" in chunk_table else None
    result.attrs.update(
        {
            "metric": "harp_sync_drift",
            "data_found": True,
            "n_sync_events": len(data),
            "nominal_clock_hz": NOMINAL_CLOCK_HZ,
            "n_chunks": len(chunk_table),
            "chunks": chunk_table,
            "worst_chunk_max_abs_residual_ms": None if worst is None or pd.isna(worst) else float(worst),
            "clock_rate_hz": float(rate),
            "clock_rate_ppm": float((rate / NOMINAL_CLOCK_HZ - 1.0) * 1e6),
            "fit": (slope, intercept),
        }
    )
    return result


def file_index(path: Path) -> int:
    """Return the integer write-order suffix of a numbered ONIX binary file, or 0."""
    suffix = path.stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def onix_clock_files(
    root: str | PathLike | list[str] | list[PathLike],
    reader: Binary,
    start: datetime.datetime,
    end: datetime.datetime | None = None,
) -> list[Path]:
    """Find the numbered clock files for the epochs overlapping a time window.

    ONIX binary files carry no timestamp and ``swc.aeon.io.api.load`` cannot place them in
    time. Epochs are therefore selected by directory name: every epoch starting in
    ``[start, end)`` and, when none starts exactly at ``start``, the latest epoch that
    started before it (it may still have been recording). A root that is itself an epoch
    directory is used as is. Files are ordered by their integer suffix within each epoch.

    Args:
        root: Dataset root path or paths.
        reader: The ``Binary`` reader whose pattern selects the clock stream.
        start: Left bound of the time range.
        end: Optional right bound of the time range.

    Returns:
        Clock file paths in acquisition order.
    """
    roots = [root] if isinstance(root, str | PathLike) else list(root)
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True) if end is not None else None
    files: list[Path] = []
    for item in roots:
        base = Path(item)
        if is_epoch_dir(base):
            epoch_dirs = [base]
        elif base.is_dir():
            dated = sorted((parse_epoch_timestamp(d), d) for d in base.iterdir() if is_epoch_dir(d))
            before = [d for t, d in dated if t < start]
            within = [d for t, d in dated if t >= start and (end is None or t < end)]
            starts_at_start = any(t == start for t, _ in dated)
            epoch_dirs = ([before[-1]] if before and not starts_at_start else []) + within
        else:
            epoch_dirs = []
        for epoch_dir in epoch_dirs:
            found = epoch_dir.glob(f"*/{reader.pattern}.{reader.extension}")
            files.extend(sorted(found, key=file_index))
    return files


def iter_clock_blocks(
    path: Path, dtype: DTypeLike = np.uint64, block_samples: int = BLOCK_SAMPLES
) -> Iterator[np.ndarray]:
    """Yield the ticks of a clock file as int64 arrays of at most ``block_samples``."""
    with open(path, "rb") as f:
        while True:
            block = np.fromfile(f, dtype=dtype, count=block_samples)
            if block.size == 0:
                return
            yield block.astype(np.int64)


def harp_index(ticks: list[float], sync_fit: tuple[float, float] | None) -> pd.DatetimeIndex:
    """Place clock ticks on Harp time with ``sync_fit``; ``NaT`` without one or for NaN ticks."""
    values = np.asarray(ticks, dtype=np.float64)
    finite = np.isfinite(values)
    if sync_fit is None or not finite.any():
        return pd.DatetimeIndex([pd.NaT] * len(ticks), name="time", tz=datetime.UTC)
    seconds = HarpSyncAlignment.estimate_harp_seconds(values[finite], *sync_fit)
    times = pd.Series(pd.NaT, index=range(len(ticks)), dtype="datetime64[ns, UTC]")
    times[finite] = to_datetime(pd.Index(seconds))
    return pd.DatetimeIndex(times, name="time")


def add_harp_time(worst: list[dict], sync_fit: tuple[float, float] | None) -> list[dict]:
    """Return ``worst`` with a ``time`` entry (Harp time of ``clock_ticks``, or None) on each item."""
    times = harp_index([w["clock_ticks"] for w in worst], sync_fit)
    return [{**w, "time": None if pd.isna(t) else t} for w, t in zip(worst, times, strict=True)]


def onix_clock_sequence(
    root: str | PathLike | list[str] | list[PathLike],
    reader: Binary,
    start: datetime.datetime,
    end: datetime.datetime | None = None,
    clock_rate_hz: float | None = None,
    sync_fit: tuple[float, float] | None = None,
    block_samples: int = BLOCK_SAMPLES,
) -> pd.DataFrame:
    """Check that per-sample ONIX clock ticks advance by exactly one sample step.

    Files are streamed block by block, carrying the last tick across blocks and files.
    Memory use is bounded whatever the recording length. The nominal step is the
    median step of the first block. A step is an event when it goes backwards, repeats,
    or (for fixed-rate streams) is more than half a nominal step from it, that is when it
    is not exactly one sample. Readers tagged ``uniform=False`` (orientation sensor, see
    ``aeon_qc.onix.clock_reader``) have a legitimately irregular interval. Only backwards
    steps and repeats are events for them. Every step's deviation from the
    nominal step is kept as a histogram, per file and overall.

    Args:
        root: Dataset root path or paths.
        reader: The ``Binary`` reader whose pattern selects the clock stream.
        start: Left bound of the time range.
        end: Optional right bound of the time range.
        clock_rate_hz: Ticks per second used to convert steps to seconds. Defaults to
            the fitted rate when ``sync_fit`` is given, else the nominal 250 MHz.
        sync_fit: ``(slope, intercept)`` from ``harp_sync_drift`` mapping ticks to Harp
            seconds. When given, rows are indexed by Harp time; otherwise by ``NaT``.
        block_samples: Samples read per block.

    Returns:
        A DataFrame with one row per event, indexed by Harp time when ``sync_fit`` is
        given and by ``NaT`` otherwise.

        - kind (str): ``backwards``, ``duplicate`` or ``jump``.
        - clock_ticks (int): The offending sample's clock value.
        - step_ticks (int), step_seconds (float): The step onto that sample.
        - file (str), index_in_file (int): Where the sample is.
        - device (str): The reader pattern.

        ``attrs`` hold ``n_samples``, ``n_files``, ``nominal_step_ticks``, ``inferred_rate_hz``,
        per-kind counts, ``truncated`` (more than ``MAX_CLOCK_EVENTS`` events were found),
        ``deviation_histogram`` (counts of ``|step - nominal|`` per power-of-two bucket, see
        ``magnitude_histogram``), ``deviation_max_abs_ticks``, ``worst`` (the largest
        deviations with their locations) and ``files`` (a DataFrame with one row per file).
    """
    files = onix_clock_files(root, reader, start=start, end=end)
    rate = (
        clock_rate_hz
        if clock_rate_hz is not None
        else (1.0 / sync_fit[0] if sync_fit is not None else float(NOMINAL_CLOCK_HZ))
    )
    counts = {"backwards": 0, "duplicate": 0, "jump": 0}
    rows: list[dict] = []
    worst: list[dict] = []
    per_file: list[dict] = []
    histogram = np.zeros(HISTOGRAM_BUCKETS, dtype=np.int64)
    n_samples = 0
    nominal: int | None = None
    last: int | None = None
    truncated = False
    uniform = getattr(reader, "uniform", True)
    for path in files:
        index_offset = 0
        file_stats = {
            "file": path.name,
            "n_samples": 0,
            "first_tick": None,
            "last_tick": None,
            "max_abs_deviation_ticks": 0,
            "n_events": 0,
        }
        for block in iter_clock_blocks(path, reader.dtype, block_samples):
            n_samples += block.size
            file_stats["n_samples"] += block.size
            if file_stats["first_tick"] is None:
                file_stats["first_tick"] = int(block[0])
            file_stats["last_tick"] = int(block[-1])
            seq = block if last is None else np.concatenate(([last], block))
            steps = np.diff(seq)
            # steps[k] lands on seq[k + 1]. With a carried sample that is block[k].
            shift = 1 if last is None else 0
            last = int(block[-1])
            if steps.size == 0:
                index_offset += block.size
                continue
            if nominal is None:
                nominal = int(np.median(steps))
            deviation = steps - nominal
            histogram += magnitude_histogram(deviation)
            top = np.argsort(np.abs(deviation))[-WORST_N:]
            worst = keep_worst(
                worst,
                [
                    {
                        "deviation_ticks": int(deviation[k]),
                        "clock_ticks": int(block[k + shift]),
                        "file": path.name,
                        "index_in_file": int(k + shift + index_offset),
                    }
                    for k in top
                ],
                key="deviation_ticks",
            )
            file_stats["max_abs_deviation_ticks"] = max(
                file_stats["max_abs_deviation_ticks"], int(np.abs(deviation).max())
            )
            bad = steps <= 0
            if uniform:
                bad |= np.abs(deviation) > HALF_SAMPLE * nominal
            for k in np.flatnonzero(bad):
                step = int(steps[k])
                kind = "duplicate" if step == 0 else "backwards" if step < 0 else "jump"
                counts[kind] += 1
                file_stats["n_events"] += 1
                if len(rows) >= MAX_CLOCK_EVENTS:
                    truncated = True
                    continue
                rows.append(
                    {
                        "kind": kind,
                        "clock_ticks": int(block[k + shift]),
                        "step_ticks": step,
                        "step_seconds": step / rate,
                        "file": path.name,
                        "index_in_file": int(k + shift + index_offset),
                        "device": reader.pattern,
                    }
                )
            index_offset += block.size
        per_file.append(file_stats)

    result = empty_result(CLOCK_COLS, "onix_clock_sequence")
    if rows:
        result = pd.DataFrame(rows, index=harp_index([r["clock_ticks"] for r in rows], sync_fit))
    result.attrs.update(
        {
            "metric": "onix_clock_sequence",
            "data_found": n_samples > 0,
            "n_samples": n_samples,
            "n_files": len(files),
            "nominal_step_ticks": nominal,
            "inferred_rate_hz": rate / nominal if nominal else None,
            "clock_rate_hz": rate,
            "truncated": truncated,
            **{f"n_{k}": v for k, v in counts.items()},
            "deviation_histogram": histogram,
            "deviation_max_abs_ticks": abs(worst[0]["deviation_ticks"]) if worst else None,
            "worst": add_harp_time(worst, sync_fit),
            "files": pd.DataFrame(per_file),
        }
    )
    return result


def onix_hub_offset(
    root: str | PathLike | list[str] | list[PathLike],
    clock_reader: Binary,
    hub_reader: Binary,
    start: datetime.datetime,
    end: datetime.datetime | None = None,
    sync_fit: tuple[float, float] | None = None,
    block_samples: int = BLOCK_SAMPLES,
) -> pd.DataFrame:
    """Measure the acquisition clock minus the headstage hub clock of every probe sample.

    Each probe sample carries two clocks: the acquisition clock stamped by the breakout
    board and the headstage hub clock captured when the sample was taken. While the
    headstage link is healthy their difference is nearly constant. A lasting change means
    the link dropped or the hub clock re-locked. Nothing is filtered. The difference is
    summarised for every file. Its deviation from the usual value (the median of the
    first block) is kept as a histogram. The largest deviations are kept with their
    locations. Files are streamed in pairs, block by block.

    Args:
        root: Dataset root path or paths.
        clock_reader: The ``Binary`` reader for the stream's acquisition clock.
        hub_reader: The ``Binary`` reader for the same stream's hub clock.
        start: Left bound of the time range.
        end: Optional right bound of the time range.
        sync_fit: ``(slope, intercept)`` from ``harp_sync_drift`` mapping ticks to Harp
            seconds. When given, rows are indexed by the Harp time of each file's first
            sample and line up with the other QC results. Otherwise they are indexed by
            ``NaT``.
        block_samples: Samples read per block.

    Returns:
        A DataFrame with one row per file, indexed by the Harp time of its first sample
        when ``sync_fit`` is given and by ``NaT`` otherwise.

        - file (str), n_samples (int).
        - min_offset_ticks, mean_offset_ticks, max_offset_ticks (float): Acquisition clock
          minus hub clock over the file.
        - length_mismatch (bool): The two clock files hold different sample counts.
        - device (str): The reader pattern.

        ``attrs`` hold ``n_samples``, ``n_files``, ``nominal_offset_ticks``, the window's
        ``min_offset_ticks`` and ``max_offset_ticks``, ``deviation_histogram`` (counts of
        ``|offset - nominal|`` per power-of-two bucket), ``deviation_max_abs_ticks`` and
        ``worst`` (the largest deviations with their locations).
    """
    clock_files = onix_clock_files(root, clock_reader, start=start, end=end)
    hub_files = onix_clock_files(root, hub_reader, start=start, end=end)
    rows: list[dict] = []
    first_ticks: list[float] = []
    worst: list[dict] = []
    histogram = np.zeros(HISTOGRAM_BUCKETS, dtype=np.int64)
    n_samples = 0
    nominal: int | None = None

    for clock_path, hub_path in zip(clock_files, hub_files, strict=False):
        index_offset = 0
        file_row = {
            "file": clock_path.name,
            "n_samples": 0,
            "min_offset_ticks": None,
            "mean_offset_ticks": None,
            "max_offset_ticks": None,
            "length_mismatch": clock_path.stat().st_size != hub_path.stat().st_size,
            "device": clock_reader.pattern,
        }
        total = 0.0
        first_tick: int | None = None
        blocks = zip(
            iter_clock_blocks(clock_path, clock_reader.dtype, block_samples),
            iter_clock_blocks(hub_path, hub_reader.dtype, block_samples),
            strict=False,
        )
        for clock_block, hub_block in blocks:
            size = min(clock_block.size, hub_block.size)
            clock = clock_block[:size]
            offset = clock - hub_block[:size]
            if size == 0:
                continue
            if first_tick is None:
                first_tick = int(clock[0])
            if nominal is None:
                nominal = int(np.median(offset))
            deviation = offset - nominal
            histogram += magnitude_histogram(deviation)
            top = np.argsort(np.abs(deviation))[-WORST_N:]
            worst = keep_worst(
                worst,
                [
                    {
                        "deviation_ticks": int(deviation[k]),
                        "clock_ticks": int(clock[k]),
                        "file": clock_path.name,
                        "index_in_file": int(k + index_offset),
                    }
                    for k in top
                ],
                key="deviation_ticks",
            )
            low, high = int(offset.min()), int(offset.max())
            file_row["min_offset_ticks"] = (
                low if file_row["min_offset_ticks"] is None else min(file_row["min_offset_ticks"], low)
            )
            file_row["max_offset_ticks"] = (
                high if file_row["max_offset_ticks"] is None else max(file_row["max_offset_ticks"], high)
            )
            total += float(offset.sum())
            file_row["n_samples"] += size
            n_samples += size
            index_offset += size
        if file_row["n_samples"]:
            file_row["mean_offset_ticks"] = total / file_row["n_samples"]
        rows.append(file_row)
        first_ticks.append(float(first_tick) if first_tick is not None else float("nan"))

    result = empty_result(HUB_COLS, "onix_hub_offset")
    if rows:
        result = pd.DataFrame(rows, index=harp_index(first_ticks, sync_fit))
    lows = [r["min_offset_ticks"] for r in rows if r["min_offset_ticks"] is not None]
    highs = [r["max_offset_ticks"] for r in rows if r["max_offset_ticks"] is not None]
    result.attrs.update(
        {
            "metric": "onix_hub_offset",
            "data_found": n_samples > 0,
            "n_samples": n_samples,
            "n_files": len(rows),
            "nominal_offset_ticks": nominal,
            "min_offset_ticks": min(lows) if lows else None,
            "max_offset_ticks": max(highs) if highs else None,
            "deviation_histogram": histogram,
            "deviation_max_abs_ticks": abs(worst[0]["deviation_ticks"]) if worst else None,
            "worst": add_harp_time(worst, sync_fit),
        }
    )
    return result
