---
uid: tutorials-run-qc
title: Interactive QC on an Aeon dataset
---

# Interactive QC on an Aeon dataset

`aeon-qc` inspects Project Aeon raw datasets and reports on data quality across all acquisition devices. It reads directly from the dataset directory structure using the `swc-aeon` API. Please note that successful use of this and other tools depend on data being logge din the AEON standard format. 

This tutorial covers interactive use: running QC from a Python script or notebook on a single dataset window. For running QC across many datasets and epochs automatically, see [Batch QC with benchmarks.yaml](batch-qc.md).

## What it checks

| Metric | What it finds |
|---|---|
| Epoch gaps | Gaps between Bonsai session starts (crashes, restarts) |
| Heartbeat gaps | Periods where a Harp device stopped sending heartbeat events |
| Heartbeat duplicates | Seconds where a Harp device emits more than one heartbeat |
| Sync delta | Timestamp drift between Harp devices relative to the clock synchroniser |
| Harp sync alerts | HarpSync alert entries parsed from the Bonsai message log |
| Dropped video frames | Jumps in the camera hardware frame counter |
| Continuous-stream gaps | Missing samples in fixed-rate Harp streams (encoder, photodiode, camera trigger), with the interval distribution |
| Pellet failures | Hardware-reported missed and retried pellet deliveries |
| Message log errors | Warning and Error entries from the Bonsai message log |
| Environment state durations | Time spent in Running vs Maintenance states |

---

## Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (installed automatically by `deploy.cmd` if missing)

---

## Installation

From the repository root:

```cmd
./deploy.cmd
```

This creates a `.venv`, installs all dependencies, and makes the `aeon_qc` package importable within that environment.

---

## Quick start

The fastest way to run all checks on a dataset is to auto-discover the device schema from `Metadata.yml` and pass it to `run_qc`:

```python
from aeon_qc import run_qc, generate_report
from aeon_qc.schemas import schema_from_metadata

root = "/ceph/aeon/aeon/data/raw/AEON3/social0.2"
start = "2024-02-01T22-00-00"   # filesystem format; ISO 8601 strings and pd.Timestamp also accepted
end   = "2024-02-02T10-00-00"

schema  = schema_from_metadata(root)
results = run_qc(root, schema, start=start, end=end)

for name, df in results.items():
    found = df.attrs.get("data_found", True)
    print(f"{name}: {'NO DATA' if not found else f'{len(df)} event(s)'}")

generate_report(root, results, "qc_report.yaml", start=start, end=end)
```

> [!TIP]
> `end` is optional. Omitting it runs QC across all epochs that begin after `start`. That may take a long time for long experiments.

---

## Understanding the schema

Every Aeon epoch directory contains a `Metadata.yml` that lists the devices active during that session. `schema_from_metadata` reads this file and builds a schema automatically.

For the AEON3 social0.2 experiment, the device list includes:

| Device | Type in Metadata.yml | QC applied |
|---|---|---|
| `ClockSynchronizer` | `TimestampGenerator` (has `PortName`) | Heartbeat gaps, sync delta reference |
| `VideoController` | `CameraController` (has `PortName`) | Heartbeat gaps |
| `CameraTop`, `CameraWest`, … | `SpinnakerVideoSource` | Dropped frames |
| `Patch1`, `Patch2`, `Patch3` | `UndergroundFeeder` (has `PortName`) | Heartbeat gaps, encoder gaps, pellet failures |
| `Nest` | `WeightScale` (has `PortName`) | Heartbeat gaps (data usually absent) |
| `NestRfid1`, `NestRfid2`, `GateRfid`, … | `RfidReader` (has `PortName`) | Heartbeat gaps |
| `AudioAmbient` | `AudioSource` (no `PortName`) | Not QC'd |
| `LightCycle` | `EnvironmentCondition` (no `PortName`) | Not QC'd |

The discovery rules are:
- Devices with a `PortName` field → Harp device → `Heartbeat` reader
- Devices with `"Type": "SpinnakerVideoSource"` → camera → `Video` reader

### Using a static schema
Predefined schemas for legacy Aeon experiments is recapitulated from aeon_mecha here. These have been phased out with [aeon-api](https://github.com/SainsburyWellcomeCentre/aeon_api), where data schemas are expected to live with individual experiment schema sets.
To use a pre-defined schema from the registry:

```python
from aeon_qc.schemas import REGISTRY

schema = REGISTRY["social02"]   # covers all streams for the social 0.2 experiment
```

Available registry keys:

| Key | Experiment |
|---|---|
| `exp02` | Foraging (two patches, AEON1/2) |
| `social02` | Social 0.2 (AEON3/4) |
| `social03` | Social 0.3 (AEON3/4) |
| `social04` | Social 0.4 (AEON3/4) |
| `octagon01` | Octagon 0.1 (OCTAGON01) |
| `socialephys01` | ONIX ephys test recording (AEONX1, NeuropixelsV2Beta headstage) |
| `abcephys01` | ForagingABC ephys (NeuropixelsV2 headstage only) |

`schema_from_metadata` automatically selects the matching registry schema if the root path contains a recognisable experiment name (e.g. `social0.2` → `social02`). The auto-discovery fallback (Heartbeat + Video only) is used for unknown experiment types.

---

## Reading the results

`run_qc` returns a `dict[str, pd.DataFrame]`. Each key identifies a device stream or metric. Each value is a tidy DataFrame with a UTC `DatetimeIndex`.

```python
results["epoch_gaps"]                              # one row per Bonsai session start
results["sync_delta"]                              # timestamp drift per device per second
results["ClockSynchronizer.Heartbeat"]             # heartbeat gaps for the clock synchroniser
results["ClockSynchronizer.Heartbeat.duplicates"]  # duplicate heartbeat seconds
results["VideoController.Heartbeat"]               # heartbeat gaps for the video controller
results["CameraTop.Video"]                         # dropped frame events for CameraTop
results["Patch1.Heartbeat"]                        # heartbeat gaps for Patch1
results["Patch1.Encoder"]                          # encoder sample drops for Patch1
results["Patch1.pellet_stats"]                     # pellet delivery failures for Patch1
results["Environment.harp_sync_alerts"]            # HarpSync alert log entries
results["Environment.message_log"]                 # non-Info Bonsai log entries
results["Environment.environment_state"]           # time in Running / Maintenance states
results["Patch1.Heartbeat.order"]                  # timestamps that go backwards or repeat (every Harp and CSV stream gets one)
results["NeuropixelsV2.HarpSync"]                  # ONIX HarpSync integrity (ephys datasets)
results["NeuropixelsV2.HarpSync.drift"]            # ONIX clock vs Harp time linear fit residuals
results["NeuropixelsV2.ProbeAClock"]               # per-sample ONIX clock sequence for probe A
```

`run_qc` accepts a list of roots as well as a single path. Ephys data is recorded on a separate machine and lands under its own rig folder. Pass the behaviour root first and the ephys root second. The ONIX streams are then found in the second root. The ephys samples for the window are selected by time overlap (see the ONIX timing checks below). Every stream is loaded once, in file order. The timestamp order check runs on the frame as read. The sorted frame then feeds the other metrics.

Each DataFrame carries metadata in `.attrs`:

```python
df = results["CameraTop.Video"]
df.attrs["data_found"]   # False if no files were found on disk for this device
df.attrs["n_frames"]     # total frames counted (including dropped)
```

An empty DataFrame with `data_found=False` means the device was in the schema but produced no data files. This is expected for `WeightScale` devices which do not emit heartbeats.

---

## Inspecting specific metrics

### Epoch gaps

```python
df = results["epoch_gaps"]
# columns: gap_duration (Timedelta, NaT for the final epoch)
# index:   UTC timestamp of each Bonsai session start
```

### Heartbeat gaps

```python
df = results["ClockSynchronizer.Heartbeat"]
# columns: duration (Timedelta), n_missed (int), second_before (int), second_after (int), device (str)
# index:   UTC timestamp of gap start

print(f"{len(df)} gap(s), total dropout: {df['duration'].sum()}")
```

### Sync delta

```python
df = results["sync_delta"]
# columns: second (int), device (str), delta_seconds (float)
# index:   UTC reference timestamp (from ClockSynchronizer)
```

### Harp sync alerts

```python
df = results["Environment.harp_sync_alerts"]
# columns: device_count, expected_device_count, max_difference
# index:   UTC timestamp of each alert

print(f"{len(df)} HarpSynch alert(s)")
```

### Dropped video frames

```python
df = results["CameraTop.Video"]
# columns: duration, n_dropped, hw_counter_before, hw_counter_after, device

if not df.empty:
    print(f"{df['n_dropped'].sum()} frames dropped across {len(df)} event(s)")
```

### Pellet failures

```python
df = results["Patch1.pellet_stats"]
# columns: outcome ('missed' or 'retried'), device

n_deliveries = df.attrs["n_deliveries"]
n_retried    = df.attrs["n_retried"]
n_missed     = df.attrs["n_missed"]

print(f"{n_deliveries} deliveries: {n_retried} retried, {n_missed} missed")
```

### Message log errors

```python
df = results["Environment.message_log"]
# columns: priority, type, message
```

### Timestamp order

```python
df = results["Patch1.Heartbeat.order"]
# columns: kind ('backwards' or 'duplicate'), step_seconds, index_in_stream, device
# index:   UTC timestamp of the violating sample
# attrs:   n_samples, n_backwards, n_duplicates, max_backwards_seconds
```

The stream is read in file order without sorting. A sample stamped earlier than its predecessor shows up as a `backwards` row with a negative `step_seconds`.

### ONIX timing checks

The four ONIX checks keep what they measure instead of filtering it through a tolerance. What counts as acceptable can be decided later from the saved results without rerunning QC. Rows are reserved for events defined by a natural rule: a Harp second that is skipped, repeated or goes backwards, or a clock step that is not exactly one sample.

```python
df = results["NeuropixelsV2.HarpSync"]
# one row per step between HarpSync records
# columns: kind ('ok', 'harp_time_gap', 'harp_time_duplicate', 'harp_time_backwards'),
#          harp_step, clock_step_ticks, deviation_ticks, device
# attrs:   n_sync_events, n_faults, seconds_offset, clock_step_median_ticks, clock_step_ppm,
#          deviation_median_abs_ticks, deviation_p99_abs_ticks, deviation_max_abs_ticks

drift = results["NeuropixelsV2.HarpSync.drift"]
# one row per record: residual_seconds against its hourly chunk's fit, chunk, device
# attrs:   chunks (rate and max residual per chunk and clock segment), n_chunks, n_clock_resets,
#          worst_chunk_max_abs_residual_ms, clock_rate_hz, clock_rate_ppm,
#          fit (slope, intercept of the largest clock segment)
```

`deviation_ticks` is how far each second's ONIX clock step sits from the recording's median step. Normal jitter is a few tens of nanoseconds. A late or early pulse shows up directly. The drift fit is done per hourly chunk because that is how ephys data is placed on Harp time (`swc.aeon.io.reader.HarpSyncAlignment`). A single fit over days hides nothing useful and its r² reads 1.0 even when it is hundreds of milliseconds off. An ONIX restart resets the acquisition clock. The records are split into clock segments where the clock steps backwards. Each segment is fitted on its own and `n_clock_resets` counts them. `seconds_offset` is 1 for recordings whose workflow added the protocol second a second time (the `Seconds` column was one second late) and 0 otherwise. `harp_time` is correct in both cases and is what the fits use.

```python
df = results["NeuropixelsV2.ProbeAClock"]
# one row per event: kind ('backwards', 'duplicate', 'jump'), clock_ticks, step_ticks,
#          step_seconds, file, index_in_file, device. The index is Harp time when a fit exists
# attrs:   n_samples, n_files, nominal_step_ticks, inferred_rate_hz, per-kind counts, truncated,
#          deviation_histogram, deviation_max_abs_ticks, worst, files (one row per file)

hub = results["NeuropixelsV2.ProbeAHubSyncCounter"]
# one row per file: min_offset_ticks, mean_offset_ticks, max_offset_ticks, length_mismatch
# attrs:   nominal_offset_ticks, min_offset_ticks, max_offset_ticks, deviation_histogram,
#          deviation_max_abs_ticks, worst (largest deviations with file, index and Harp time)
```

The clock files carry no timestamps of their own. For each ephys epoch the QC window is converted to clock ticks with that epoch's HarpSync fit. A file is kept when its first and last ticks cross the window. Only the samples inside the window are checked. The ephys recording may start before or after the behaviour epoch. Only the overlap is checked and every event is placed on Harp time. An epoch without HarpSync records cannot be placed in time. Its files are checked whole when the epoch directory name falls inside the window, or is the latest one before it.

A clock step is a `jump` when it is more than half a nominal step from it, that is when it is not exactly one sample. The single-sample jump of 2^32 ticks (17.18 s at 250 MHz) reported in aeon_roadmap#78 shows up as a `jump` followed by a `backwards` row. The orientation sensor stream is only checked for backwards steps and repeats: its sample interval varies. The commutator clock is not checked: each turn command copies the clock of the orientation frame that triggered it.

Each probe sample carries the acquisition clock and the headstage hub clock. While the headstage link is healthy their difference stays within a few ticks. A lasting change means the link dropped or the hub clock re-locked. The histograms count deviations in power-of-two buckets (`0`, `<2^1`, `<2^2`, ...). The full distribution is kept for recordings of any length.

---

## Generating a YAML report

`generate_report` writes a structured summary to disk:

```python
from aeon_qc import generate_report

generate_report(root, results, "reports/social02_aeon3.yaml", start=start, end=end)
```

The output format:

```yaml
generated_at: "2026-04-07T10:05:00+00:00"
dataset_root: /ceph/aeon/aeon/data/raw/AEON3/social0.2
time_range:
  start: "2024-02-01T22:00:00+00:00"
  end: "2024-02-02T10:00:00+00:00"
devices:
  ClockSynchronizer.Heartbeat:
    metric: heartbeat_gaps
    summary:
      data_found: true
      n_heartbeats: 43200
      n_gaps: 0
      total_dropout_seconds: 0.0
      mean_duration_seconds: null
    detail: []
  CameraTop.Video:
    metric: dropped_frames
    summary:
      data_found: true
      n_frames: 2160000
      n_drop_events: 3
      total_frames_dropped: 11
      mean_duration_seconds: 0.042
    detail:
      - time: "2024-02-02T01:14:22.400000+00:00"
        n_dropped: 4
        hw_counter_before: 864210
        hw_counter_after: 864215
```

---

## Saving results for later analysis

To avoid re-running QC for downstream analysis, pickle the results dict:

```python
import pickle
from aeon_qc import save_results

save_results(results, "reports/social02_aeon3.pkl")

# Later:
with open("reports/social02_aeon3.pkl", "rb") as f:
    results = pickle.load(f)
```

---

## Next steps

- To run QC systematically across many datasets and epochs, see [Batch QC with benchmarks.yaml](batch-qc.md).
- See the [API reference](xref:aeon_qc) for full function signatures and return value descriptions.
- See [aeon_roadmap#40](https://github.com/SainsburyWellcomeCentre/aeon_roadmap/issues/40) for the full list of requested metrics and their implementation status.
