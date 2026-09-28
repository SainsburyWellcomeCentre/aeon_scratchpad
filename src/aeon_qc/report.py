"""High-level run_qc orchestration and YAML report generation."""

# pandas-stubs types row attributes from itertuples() and iloc as broad unions, which
# the report helpers below hit in over a hundred places. Suppress those two rules once here.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

import datetime
import pickle
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from swc.aeon.io.api import Reader, load
from swc.aeon.io.reader import Binary, Csv, Encoder, Harp, Heartbeat, Pose, Video

from aeon_qc.environment import environment_state_durations, harp_sync_alerts, message_log_errors
from aeon_qc.ephys import (
    WORST_N,
    harp_sync_drift,
    harp_sync_integrity,
    onix_clock_sequence,
    onix_hub_offset,
)
from aeon_qc.epochs import epoch_gaps
from aeon_qc.harp import harp_gaps
from aeon_qc.heartbeat import heartbeat_duplicates, heartbeat_gaps
from aeon_qc.pellet import pellet_failures
from aeon_qc.reader import HarpSync
from aeon_qc.schemas import is_epoch_dir, normalise_timestamp
from aeon_qc.sequence import timestamp_order
from aeon_qc.sync import MIN_DEVICES, sync_delta
from aeon_qc.video import dropped_frames, frame_rate_stability

DETAIL_ROW_CAP = 100
"""Maximum number of detail rows written to the YAML report for high-volume metrics."""


def iter_readers(schema: Any) -> Iterator[tuple[str, Reader]]:
    """Yield (qualified_name, reader) pairs from a schema DotMap."""
    for device_name, streams in schema.items():
        if isinstance(streams, Reader):
            yield (device_name, streams)
        elif isinstance(streams, dict):
            for stream_name, reader in streams.items():
                if isinstance(reader, Reader):
                    yield (f"{device_name}.{stream_name}", reader)


def run_qc(
    root: str | PathLike | list[str] | list[PathLike],
    schema: Any,
    start: str | datetime.datetime,
    end: str | datetime.datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all applicable QC checks against every stream in a schema DotMap.

    Every Harp and CSV stream is loaded once, in the order its rows were written
    (``load(sort=False)``). The timestamp order check runs on that frame. The frame is
    then sorted and the sorted frame feeds every other metric for the stream. No stream
    is read twice and ordering faults are seen before anything sorts them away.
    SLEAP pose streams are not loaded: they need the model configuration and carry
    several rows per frame by design.

    Args:
        root: Dataset root path, or a list of roots searched together (behaviour root
            first, then for example the ephys root recorded on another machine). Epoch
            gaps are computed on the first root only.
        schema: DotMap of devices to stream readers.
        start: Left bound of the time range.
        end: Optional right bound of the time range.

    Returns:
        Mapping of result key to metric DataFrame.

    """
    start_ts: datetime.datetime = normalise_timestamp(start)
    end_ts: datetime.datetime | None = normalise_timestamp(end) if end is not None else None
    results: dict[str, pd.DataFrame] = {}
    first_root = Path(root[0]) if isinstance(root, list) else Path(root)
    if not is_epoch_dir(first_root):
        results["epoch_gaps"] = epoch_gaps(first_root, start=start_ts, end=end_ts)
    heartbeat_readers: dict[str, Heartbeat] = {}
    device_streams: dict[str, dict[str, Reader]] = {}
    frames: dict[str, pd.DataFrame] = {}
    device_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for qualified_name, reader in iter_readers(schema):
        device, _, stream_name = qualified_name.partition(".")
        device_streams.setdefault(device, {})[stream_name or device] = reader
        if not isinstance(reader, Harp | Csv) or isinstance(reader, Pose):
            continue
        data = load(root, reader, start=start_ts, end=end_ts, sort=False)
        results[f"{qualified_name}.order"] = timestamp_order(
            root, reader, start=start_ts, end=end_ts, data=data
        )
        if not data.index.is_monotonic_increasing:
            data = data.sort_index(kind="stable")
        frames[qualified_name] = data
        device_frames.setdefault(device, {})[stream_name or device] = data
        if isinstance(reader, Heartbeat):
            results[qualified_name] = heartbeat_gaps(root, reader, start=start_ts, end=end_ts, data=data)
            if "rfid" not in device.lower():
                results[f"{qualified_name}.duplicates"] = heartbeat_duplicates(
                    root, reader, start=start_ts, end=end_ts, data=data
                )
            heartbeat_readers[qualified_name] = reader
        elif isinstance(reader, Video):
            results[qualified_name] = dropped_frames(root, reader, start=start_ts, end=end_ts, data=data)
            results[f"{device}.frame_rate"] = frame_rate_stability(
                root, reader, start=start_ts, end=end_ts, data=data
            )
        elif isinstance(reader, Encoder):
            reader.expected_hz = 500.0
            results[qualified_name] = harp_gaps(root, reader, start=start_ts, end=end_ts, data=data)
        elif isinstance(reader, Harp) and hasattr(reader, "expected_hz"):
            results[qualified_name] = harp_gaps(root, reader, start=start_ts, end=end_ts, data=data)
    if len(heartbeat_readers) >= MIN_DEVICES:
        results["sync_delta"] = sync_delta(
            root,
            heartbeat_readers,
            start=start_ts,
            end=end_ts,
            data={name: frames[name] for name in heartbeat_readers},
        )

    for device_name, streams in device_streams.items():
        loaded = device_frames.get(device_name, {})
        if "DeliverPellet" in streams:
            pellet_frames = {
                key: loaded[stream]
                for key, stream in (
                    ("deliver", "DeliverPellet"),
                    ("missed", "MissedPellet"),
                    ("retried", "RetriedDelivery"),
                )
                if stream in loaded
            }
            results[f"{device_name}.pellet_stats"] = pellet_failures(
                root,
                deliver_reader=streams["DeliverPellet"],
                missed_reader=streams.get("MissedPellet"),
                retried_reader=streams.get("RetriedDelivery"),
                start=start_ts,
                end=end_ts,
                data=pellet_frames,
            )
        if "MessageLog" in streams:
            results[f"{device_name}.message_log"] = message_log_errors(
                root, streams["MessageLog"], start=start_ts, end=end_ts, data=loaded.get("MessageLog")
            )
            results[f"{device_name}.harp_sync_alerts"] = harp_sync_alerts(
                root, streams["MessageLog"], start=start_ts, end=end_ts, data=loaded.get("MessageLog")
            )
        if "EnvironmentState" in streams:
            results[f"{device_name}.environment_state"] = environment_state_durations(
                root,
                streams["EnvironmentState"],
                start=start_ts,
                end=end_ts,
                data=loaded.get("EnvironmentState"),
            )
        for stream_name, reader in streams.items():
            if isinstance(reader, HarpSync):
                key = f"{device_name}.{stream_name}"
                data = loaded.get(stream_name)
                results[key] = harp_sync_integrity(root, reader, start=start_ts, end=end_ts, data=data)
                results[f"{key}.drift"] = harp_sync_drift(
                    root, reader, start=start_ts, end=end_ts, data=data
                )
            elif isinstance(reader, Binary) and hasattr(reader, "uniform"):
                results[f"{device_name}.{stream_name}"] = onix_clock_sequence(
                    root, reader, start=start_ts, end=end_ts
                )
                hub_name = stream_name.removesuffix("Clock") + "HubSyncCounter"
                if hub_name in streams:
                    results[f"{device_name}.{hub_name}"] = onix_hub_offset(
                        root, reader, streams[hub_name], start=start_ts, end=end_ts
                    )

    return results


def generate_report(
    root: str | PathLike,
    results: dict[str, pd.DataFrame],
    output_path: str | PathLike,
    start: str | datetime.datetime,
    end: str | datetime.datetime | None = None,
) -> Path:
    """Write a human-readable YAML QC summary from QC metric DataFrames."""
    output_path = Path(output_path)
    start_ts: datetime.datetime = normalise_timestamp(start)
    end_ts: datetime.datetime | None = normalise_timestamp(end) if end is not None else None

    report: dict[str, Any] = {
        "generated_at": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "dataset_root": str(root),
        "time_range": {
            "start": start_ts.isoformat(),
            "end": end_ts.isoformat() if end_ts is not None else None,
        },
        "devices": {},
    }

    for device_name, df in results.items():
        metric = df.attrs.get("metric")
        if metric == "timestamp_order":
            report["devices"][device_name] = timestamp_order_section(df)
        elif metric == "harp_sync_integrity":
            report["devices"][device_name] = harp_sync_integrity_section(df)
        elif metric == "harp_sync_drift":
            report["devices"][device_name] = harp_sync_drift_section(df)
        elif metric == "onix_clock_sequence":
            report["devices"][device_name] = onix_clock_section(df)
        elif metric == "onix_hub_offset":
            report["devices"][device_name] = onix_hub_section(df)
        elif "gap_duration" in df.columns:
            report["devices"][device_name] = epoch_gaps_section(df)
        elif "count" in df.columns and "second" in df.columns:
            report["devices"][device_name] = heartbeat_duplicates_section(df)
        elif "n_dropped" in df.columns:
            report["devices"][device_name] = video_section(df)
        elif "second_before" in df.columns:
            report["devices"][device_name] = heartbeat_section(df)
        elif "n_missed" in df.columns:
            report["devices"][device_name] = harp_gaps_section(df)
        elif "duration" in df.columns and "state" in df.columns:
            report["devices"][device_name] = environment_state_section(df)
        elif "outcome" in df.columns:
            report["devices"][device_name] = pellet_section(df)
        elif "max_difference" in df.columns:
            report["devices"][device_name] = harp_sync_alerts_section(df)
        elif "priority" in df.columns:
            report["devices"][device_name] = message_log_section(df)
        elif "fps_inferred" in df.columns:
            report["devices"][device_name] = frame_rate_section(df)
        elif "delta_seconds" in df.columns:
            report["devices"][device_name] = sync_delta_section(df)

    missing = [name for name, df in results.items() if not df.attrs.get("data_found", True)]
    if missing:
        report["missing_devices"] = sorted(missing)

    with open(output_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return output_path


def save_results(results: dict[str, pd.DataFrame], output_path: str | PathLike) -> Path:
    """Pickle a run_qc results dict to disk for later analysis."""
    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    return output_path


def heartbeat_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a heartbeat_gaps result."""
    data_found = df.attrs.get("data_found", True)
    n_heartbeats = df.attrs.get("n_heartbeats", 0)
    if df.empty:
        summary: dict[str, Any] = {
            "data_found": data_found,
            "n_heartbeats": n_heartbeats,
            "n_gaps": 0,
            "total_dropout_seconds": 0.0,
            "mean_duration_seconds": None,
        }
        detail: list[dict[str, Any]] = []
    else:
        summary = {
            "data_found": data_found,
            "n_heartbeats": n_heartbeats,
            "n_gaps": len(df),
            "total_dropout_seconds": float(df["duration"].dt.total_seconds().sum()),
            "mean_duration_seconds": float(df["duration"].dt.total_seconds().mean()),
        }
        detail = [
            {
                "time": row.Index.isoformat(),
                "duration_seconds": float(row.duration.total_seconds()),
                "second_before": int(row.second_before),
                "second_after": int(row.second_after),
            }
            for row in df.itertuples()
        ]
    return {"metric": "heartbeat_gaps", "summary": summary, "detail": detail}


def video_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a dropped_frames result."""
    data_found = df.attrs.get("data_found", True)
    n_frames = df.attrs.get("n_frames", 0)
    if df.empty:
        summary: dict[str, Any] = {
            "data_found": data_found,
            "n_frames": n_frames,
            "n_drop_events": 0,
            "total_frames_dropped": 0,
            "mean_duration_seconds": None,
        }
        detail: list[dict[str, Any]] = []
    else:
        summary = {
            "data_found": data_found,
            "n_frames": n_frames,
            "n_drop_events": len(df),
            "total_frames_dropped": int(df["n_dropped"].sum()),
            "mean_duration_seconds": float(df["duration"].dt.total_seconds().mean()),
        }
        detail = [
            {
                "time": row.Index.isoformat(),
                "n_dropped": int(row.n_dropped),
                "hw_counter_before": int(row.hw_counter_before),
                "hw_counter_after": int(row.hw_counter_after),
            }
            for row in df.itertuples()
        ]
    return {"metric": "dropped_frames", "summary": summary, "detail": detail}


def sync_delta_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a sync_delta result."""
    if df.empty:
        summary: dict[str, Any] = {
            "n_devices": 0,
            "max_abs_delta_seconds": None,
            "mean_abs_delta_seconds": None,
            "std_delta_seconds": None,
            "worst_device": None,
        }
        detail: list[dict[str, Any]] = []
    else:
        abs_delta = df["delta_seconds"].abs()
        by_device = df.groupby("device")["delta_seconds"]
        worst_device = by_device.apply(lambda s: s.abs().max()).idxmax()
        summary = {
            "n_devices": int(df["device"].nunique()),
            "max_abs_delta_seconds": float(abs_delta.max()),
            "mean_abs_delta_seconds": float(abs_delta.mean()),
            "std_delta_seconds": float(df["delta_seconds"].std()),
            "worst_device": worst_device,
        }
        detail = [
            {
                "device": device,
                "max_delta_seconds": float(grp["delta_seconds"].abs().max()),
                "mean_delta_seconds": float(grp["delta_seconds"].abs().mean()),
                "std_delta_seconds": float(grp["delta_seconds"].std()),
            }
            for device, grp in df.groupby("device")
        ]
    return {"metric": "sync_delta", "summary": summary, "detail": detail}


def epoch_gaps_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for an epoch_gaps result."""
    data_found = df.attrs.get("data_found", True)
    if df.empty:
        summary: dict[str, Any] = {
            "data_found": data_found,
            "n_epochs": 0,
            "total_gap_seconds": 0.0,
            "max_gap_seconds": None,
            "min_gap_seconds": None,
        }
        detail: list[dict[str, Any]] = []
    else:
        durations = df["gap_duration"].dropna().dt.total_seconds()
        summary = {
            "data_found": data_found,
            "n_epochs": len(df),
            "total_gap_seconds": float(durations.sum()),
            "max_gap_seconds": float(durations.max()) if len(durations) else None,
            "min_gap_seconds": float(durations.min()) if len(durations) else None,
        }
        detail = [
            {
                "epoch_start": row.Index.isoformat(),
                "gap_duration_seconds": float(row.gap_duration.total_seconds())
                if pd.notna(row.gap_duration)
                else None,
            }
            for row in df.itertuples()
        ]
    return {"metric": "epoch_gaps", "summary": summary, "detail": detail}


def harp_gaps_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a harp_gaps result on a continuous-rate Harp stream.

    The interval ratios and longest intervals are reported even when there are no gaps,
    so a zero count comes with the evidence behind it.
    """
    summary: dict[str, Any] = {
        "data_found": df.attrs.get("data_found", True),
        "expected_hz": df.attrs.get("expected_hz"),
        "n_samples": int(df.attrs.get("n_samples", 0)),
        "n_gap_events": int(df.attrs.get("n_gap_events", 0)),
        "total_missed_samples": int(df.attrs.get("n_missed_total", 0)),
        "n_irregular_runs": int(df.attrs.get("n_irregular_runs", 0)),
        "n_extra_runs": int(df.attrs.get("n_extra_runs", 0)),
        "interval_ratio_min": optional_float(df.attrs.get("interval_ratio_min"), 4),
        "interval_ratio_median": optional_float(df.attrs.get("interval_ratio_median"), 4),
        "interval_ratio_p99_99": optional_float(df.attrs.get("interval_ratio_p99_99"), 4),
        "interval_ratio_max": optional_float(df.attrs.get("interval_ratio_max"), 4),
    }
    detail = [
        {
            "time": row.Index.isoformat(),
            "kind": row.kind,
            "duration_ms": float(row.duration.total_seconds() * 1000),
            "n_intervals": int(row.n_intervals),
            "n_missed": int(row.n_missed),
        }
        for row in df.head(DETAIL_ROW_CAP).itertuples()
    ]
    if len(df) > DETAIL_ROW_CAP:
        summary["detail_truncated_to"] = DETAIL_ROW_CAP
    return {
        "metric": "harp_gaps",
        "summary": summary,
        "detail": detail,
        "longest_intervals": list(df.attrs.get("longest", [])),
    }


def pellet_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a pellet_failures result."""
    data_found = df.attrs.get("data_found", True)
    n_deliveries = df.attrs.get("n_deliveries", 0)
    n_retried = df.attrs.get("n_retried", 0)
    n_missed = df.attrs.get("n_missed", 0)
    summary: dict[str, Any] = {
        "data_found": data_found,
        "n_deliveries": n_deliveries,
        "n_retried": n_retried,
        "n_missed": n_missed,
    }
    detail: list[dict[str, Any]] = [
        {"time": row.Index.isoformat(), "outcome": row.outcome} for row in df.itertuples()
    ]
    return {"metric": "pellet_failures", "summary": summary, "detail": detail}


def harp_sync_alerts_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a harp_sync_alerts result."""
    data_found = df.attrs.get("data_found", True)
    n_total = df.attrs.get("n_total_messages", 0)
    if df.empty:
        summary: dict[str, Any] = {
            "data_found": data_found,
            "n_total_messages": n_total,
            "n_alerts": 0,
            "max_max_difference": None,
            "min_device_count": None,
        }
        detail: list[dict[str, Any]] = []
    else:
        summary = {
            "data_found": data_found,
            "n_total_messages": n_total,
            "n_alerts": len(df),
            "max_max_difference": float(df["max_difference"].max()),
            "min_device_count": int(df["device_count"].min()),
        }
        detail = [
            {
                "time": row.Index.isoformat(),
                "device_count": int(row.device_count),
                "expected_device_count": int(row.expected_device_count),
                "max_difference": float(row.max_difference),
            }
            for row in df.itertuples()
        ]
    return {"metric": "harp_sync_alerts", "summary": summary, "detail": detail}


def message_log_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a message_log_errors result."""
    data_found = df.attrs.get("data_found", True)
    n_total = df.attrs.get("n_total", 0)
    if df.empty:
        summary: dict[str, Any] = {
            "data_found": data_found,
            "n_total": n_total,
            "n_alerts": 0,
            "n_warnings": 0,
            "n_errors": 0,
        }
        detail: list[dict[str, Any]] = []
    else:
        counts = df["priority"].str.lower().value_counts()
        summary = {
            "data_found": data_found,
            "n_total": n_total,
            "n_alerts": int(counts.get("alert", 0)),
            "n_warnings": int(counts.get("warning", 0)),
            "n_errors": int(counts.get("error", 0)),
        }
        detail = [
            {
                "time": row.Index.isoformat(),
                "priority": row.priority,
                "type": row.type,
                "message": row.message,
            }
            for row in df.itertuples()
        ]
    return {"metric": "message_log_errors", "summary": summary, "detail": detail}


def heartbeat_duplicates_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a heartbeat_duplicates result."""
    data_found = df.attrs.get("data_found", True)
    n_heartbeats = df.attrs.get("n_heartbeats", 0)
    if df.empty:
        summary: dict[str, Any] = {
            "data_found": data_found,
            "n_heartbeats": n_heartbeats,
            "n_affected_seconds": 0,
        }
        detail: list[dict[str, Any]] = []
    else:
        summary = {
            "data_found": data_found,
            "n_heartbeats": n_heartbeats,
            "n_affected_seconds": len(df),
        }
        detail = [
            {"time": row.Index.isoformat(), "second": int(row.second), "count": int(row.count)}
            for row in df.itertuples()
        ]
    return {"metric": "heartbeat_duplicates", "summary": summary, "detail": detail}


def frame_rate_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a frame_rate_stability result."""
    data_found = df.attrs.get("data_found", True)
    fps_source = df.attrs.get("fps_source", "inferred_from_median")
    clock = df.attrs.get("clock", "hw_timestamp_ns")
    row = df.iloc[0]
    fps = row["fps_inferred"]
    if not data_found or pd.isna(fps):
        summary: dict[str, Any] = {
            "data_found": data_found,
            "fps_inferred": None,
            "fps_source": fps_source,
            "clock": clock,
        }
        return {"metric": "frame_rate_stability", "summary": summary}
    summary = {
        "data_found": data_found,
        "n_frames": int(row["n_frames"]),
        "fps_inferred": round(float(fps), 3),
        "fps_source": fps_source,
        "clock": clock,
        "interval_median_ms": round(float(row["interval_median_ms"]), 4),
        "interval_std_ms": round(float(row["interval_std_ms"]), 4),
        "interval_p99_ms": round(float(row["interval_p99_ms"]), 4),
        "interval_max_ms": round(float(row["interval_max_ms"]), 4),
    }
    return {"metric": "frame_rate_stability", "summary": summary}


def timestamp_order_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a timestamp_order result (detail capped at DETAIL_ROW_CAP)."""
    summary: dict[str, Any] = {
        "data_found": df.attrs.get("data_found", True),
        "n_samples": int(df.attrs.get("n_samples", 0)),
        "n_backwards": int(df.attrs.get("n_backwards", 0)),
        "n_duplicates": int(df.attrs.get("n_duplicates", 0)),
        "max_backwards_seconds": float(df.attrs.get("max_backwards_seconds", 0.0)),
    }
    detail = [
        {
            "time": row.Index.isoformat(),
            "kind": row.kind,
            "step_seconds": float(row.step_seconds),
            "index_in_stream": int(row.index_in_stream),
        }
        for row in df.head(DETAIL_ROW_CAP).itertuples()
    ]
    if len(df) > DETAIL_ROW_CAP:
        summary["detail_truncated_to"] = DETAIL_ROW_CAP
    return {"metric": "timestamp_order", "summary": summary, "detail": detail}


def optional_float(value: Any, digits: int | None = None) -> float | None:
    """Return ``value`` as a float rounded to ``digits``, or None when it is missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return round(float(value), digits) if digits is not None else float(value)


def histogram_section(counts: Any) -> dict[str, int]:
    """Render a power-of-two magnitude histogram as ``{"0": n, "<2^k": n}``, non-empty buckets only."""
    if counts is None:
        return {}
    out: dict[str, int] = {}
    for k, n in enumerate(counts):
        if n:
            out["0" if k == 0 else f"<2^{k}"] = int(n)
    return out


def worst_section(worst: Any) -> list[dict[str, Any]]:
    """Render the worst-deviation entries kept by an ONIX check for the YAML report."""
    return [
        {
            "time": None if w.get("time") is None else w["time"].isoformat(),
            "deviation_ticks": int(w["deviation_ticks"]),
            "clock_ticks": int(w["clock_ticks"]),
            "file": w["file"],
            "index_in_file": int(w["index_in_file"]),
        }
        for w in (worst or [])
    ]


def harp_sync_integrity_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a harp_sync_integrity result.

    Lists every Harp-time fault and the steps whose clock step deviates most from the
    median; every step is kept in the saved results.
    """
    summary: dict[str, Any] = {
        "data_found": df.attrs.get("data_found", True),
        "n_sync_events": int(df.attrs.get("n_sync_events", 0)),
        "seconds_offset": optional_float(df.attrs.get("seconds_offset"), 3),
        "clock_step_median_ticks": optional_float(df.attrs.get("clock_step_median_ticks"), 1),
        "clock_step_ppm": optional_float(df.attrs.get("clock_step_ppm"), 2),
        "n_faults": int(df.attrs.get("n_faults", 0)),
        "n_harp_time_gap": int(df.attrs.get("n_harp_time_gap", 0)),
        "n_harp_time_duplicate": int(df.attrs.get("n_harp_time_duplicate", 0)),
        "n_harp_time_backwards": int(df.attrs.get("n_harp_time_backwards", 0)),
        "deviation_median_abs_ticks": optional_float(df.attrs.get("deviation_median_abs_ticks"), 1),
        "deviation_p99_abs_ticks": optional_float(df.attrs.get("deviation_p99_abs_ticks"), 1),
        "deviation_max_abs_ticks": optional_float(df.attrs.get("deviation_max_abs_ticks"), 1),
    }

    def rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "time": row.Index.isoformat(),
                "kind": row.kind,
                "harp_step": int(row.harp_step),
                "clock_step_ticks": int(row.clock_step_ticks),
                "deviation_ticks": None if pd.isna(row.deviation_ticks) else float(row.deviation_ticks),
            }
            for row in frame.itertuples()
        ]

    if df.empty:
        return {"metric": "harp_sync_integrity", "summary": summary, "faults": [], "worst_steps": []}
    faults = df[df["kind"] != "ok"]
    advanced = df.dropna(subset=["deviation_ticks"])
    order = advanced["deviation_ticks"].abs().sort_values(ascending=False).index[:WORST_N]
    if len(faults) > DETAIL_ROW_CAP:
        summary["faults_truncated_to"] = DETAIL_ROW_CAP
    return {
        "metric": "harp_sync_integrity",
        "summary": summary,
        "faults": rows(faults.head(DETAIL_ROW_CAP)),
        "worst_steps": rows(advanced.loc[order]),
    }


def harp_sync_drift_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for a harp_sync_drift result: per-chunk fits, worst first."""
    summary: dict[str, Any] = {
        "data_found": df.attrs.get("data_found", True),
        "n_sync_events": int(df.attrs.get("n_sync_events", 0)),
        "n_chunks": int(df.attrs.get("n_chunks", 0)),
        "n_clock_resets": int(df.attrs.get("n_clock_resets", 0)),
        "worst_chunk_max_abs_residual_ms": optional_float(
            df.attrs.get("worst_chunk_max_abs_residual_ms"), 4
        ),
        "nominal_clock_hz": df.attrs.get("nominal_clock_hz"),
        "clock_rate_hz": optional_float(df.attrs.get("clock_rate_hz"), 1),
        "clock_rate_ppm": optional_float(df.attrs.get("clock_rate_ppm"), 2),
    }
    chunks = df.attrs.get("chunks")
    worst_chunks: list[dict[str, Any]] = []
    if isinstance(chunks, pd.DataFrame) and "max_abs_residual_ms" in chunks:
        ranked = chunks.dropna(subset=["max_abs_residual_ms"])
        ranked = ranked.sort_values("max_abs_residual_ms", ascending=False)
        summary["median_chunk_max_abs_residual_ms"] = optional_float(
            ranked["max_abs_residual_ms"].median(), 4
        )
        summary["chunk_rate_ppm_min"] = optional_float(ranked["clock_rate_ppm"].min(), 2)
        summary["chunk_rate_ppm_max"] = optional_float(ranked["clock_rate_ppm"].max(), 2)
        worst_chunks = [
            {
                "chunk": pd.Timestamp(row.Index).isoformat(),
                "n_sync_events": int(row.n_sync_events),
                "clock_rate_ppm": optional_float(row.clock_rate_ppm, 2),
                "max_abs_residual_ms": optional_float(row.max_abs_residual_ms, 4),
            }
            for row in ranked.head(WORST_N).itertuples()
        ]
    return {"metric": "harp_sync_drift", "summary": summary, "worst_chunks": worst_chunks}


def onix_hub_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for an onix_hub_offset result."""
    summary: dict[str, Any] = {
        "data_found": df.attrs.get("data_found", True),
        "n_samples": int(df.attrs.get("n_samples", 0)),
        "n_files": int(df.attrs.get("n_files", 0)),
        "nominal_offset_ticks": df.attrs.get("nominal_offset_ticks"),
        "min_offset_ticks": df.attrs.get("min_offset_ticks"),
        "max_offset_ticks": df.attrs.get("max_offset_ticks"),
        "deviation_max_abs_ticks": df.attrs.get("deviation_max_abs_ticks"),
        "n_length_mismatch": int(df["length_mismatch"].sum()) if not df.empty else 0,
        "deviation_histogram": histogram_section(df.attrs.get("deviation_histogram")),
    }
    widest_files: list[dict[str, Any]] = []
    if not df.empty:
        spread = (df["max_offset_ticks"] - df["min_offset_ticks"]).astype(float)
        widest = df.iloc[np.argsort(-spread.fillna(-1).to_numpy())[:WORST_N]]
        widest_files = [
            {
                "time": None if pd.isna(row.Index) else row.Index.isoformat(),
                "file": row.file,
                "n_samples": int(row.n_samples),
                "min_offset_ticks": None if pd.isna(row.min_offset_ticks) else int(row.min_offset_ticks),
                "max_offset_ticks": None if pd.isna(row.max_offset_ticks) else int(row.max_offset_ticks),
                "length_mismatch": bool(row.length_mismatch),
            }
            for row in widest.itertuples()
        ]
    return {
        "metric": "onix_hub_offset",
        "summary": summary,
        "worst_samples": worst_section(df.attrs.get("worst")),
        "widest_files": widest_files,
    }


def onix_clock_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for an onix_clock_sequence result."""
    summary: dict[str, Any] = {
        "data_found": df.attrs.get("data_found", True),
        "n_samples": int(df.attrs.get("n_samples", 0)),
        "n_files": int(df.attrs.get("n_files", 0)),
        "nominal_step_ticks": df.attrs.get("nominal_step_ticks"),
        "inferred_rate_hz": optional_float(df.attrs.get("inferred_rate_hz"), 3),
        "clock_rate_hz": optional_float(df.attrs.get("clock_rate_hz"), 1),
        "n_events": int(len(df)),
        "n_backwards": int(df.attrs.get("n_backwards", 0)),
        "n_duplicate": int(df.attrs.get("n_duplicate", 0)),
        "n_jump": int(df.attrs.get("n_jump", 0)),
        "deviation_max_abs_ticks": df.attrs.get("deviation_max_abs_ticks"),
        "deviation_histogram": histogram_section(df.attrs.get("deviation_histogram")),
    }
    if df.attrs.get("truncated"):
        summary["events_truncated"] = True
    events = [
        {
            "time": None if pd.isna(row.Index) else row.Index.isoformat(),
            "kind": row.kind,
            "clock_ticks": int(row.clock_ticks),
            "step_ticks": int(row.step_ticks),
            "step_seconds": float(row.step_seconds),
            "file": row.file,
            "index_in_file": int(row.index_in_file),
        }
        for row in df.head(DETAIL_ROW_CAP).itertuples()
    ]
    if len(df) > DETAIL_ROW_CAP:
        summary["detail_truncated_to"] = DETAIL_ROW_CAP
    return {
        "metric": "onix_clock_sequence",
        "summary": summary,
        "events": events,
        "worst_steps": worst_section(df.attrs.get("worst")),
    }


def environment_state_section(df: pd.DataFrame) -> dict[str, Any]:
    """Build the YAML section for an environment_state_durations result."""
    data_found = df.attrs.get("data_found", True)
    if df.empty:
        summary: dict[str, Any] = {
            "data_found": data_found,
            "n_transitions": 0,
            "state_totals_seconds": {},
        }
        detail: list[dict[str, Any]] = []
    else:
        totals = df.groupby("state")["duration"].sum().dt.total_seconds().round(1).to_dict()
        summary = {
            "data_found": data_found,
            "n_transitions": len(df),
            "state_totals_seconds": {k: float(v) for k, v in totals.items()},
        }
        detail = [
            {
                "time": row.Index.isoformat(),
                "state": row.state,
                "duration_seconds": float(row.duration.total_seconds()),
            }
            for row in df.itertuples()
        ]
    return {"metric": "environment_state", "summary": summary, "detail": detail}
