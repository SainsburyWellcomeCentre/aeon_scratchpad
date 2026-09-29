# aeon-qc

Data quality control (QC) metrics for [Project Aeon](https://github.com/SainsburyWellcomeCentre/aeon_roadmap) datasets.

Developed as a prototype against [aeon_roadmap#40](https://github.com/SainsburyWellcomeCentre/aeon_roadmap/issues/40).
Consider folding back into [`aeon_api`](https://github.com/SainsburyWellcomeCentre/aeon_api).

## QC metrics

| Metric | Result key | What it detects |
|---|---|---|
| Epoch gaps | `epoch_gaps` | Gaps between consecutive Bonsai session starts (crashes, restarts) |
| Heartbeat gaps | `<device>.Heartbeat` | Periods where a Harp device stops emitting heartbeat events (~1 Hz) |
| Heartbeat duplicates | `<device>.Heartbeat.duplicates` | Seconds where a Harp device emits more than one heartbeat (resync events) |
| Sync delta | `sync_delta` | Per-second timestamp drift between Harp devices relative to the ClockSynchronizer |
| Harp sync alerts | `<device>.harp_sync_alerts` | HarpSynch alerts from the Bonsai SynchronizerMonitor (device count or clock misalignment) |
| Dropped frames | `<device>.Video` | Jumps in the camera hardware frame counter indicating lost frames |
| Continuous-stream gaps | `<device>.Encoder`, `Photodiode`, `VideoController` | Missing samples in fixed rate Harp streams (encoder 500 Hz, photodiode 1 kHz, camera trigger 50 Hz). Over each run of irregular intervals, the measured interval count is compared with the count expected for its span, differentiating missing from late samples. The interval distribution and longest intervals are reported |
| Pellet failures | `<device>.pellet_stats` | Hardware-reported missed and retried pellet deliveries |
| Message log errors | `<device>.message_log` | Warning and Error entries from the Bonsai message log |
| Environment state durations | `<device>.environment_state` | Time spent in each environment state (Running, Maintenance) |
| Timestamp order | `<device>.<stream>.order` | Timestamps that go backwards or repeat within any Harp or CSV stream, read in file order without sorting |
| HarpSync integrity | `<device>.HarpSync` | Every ONIX HarpSync step: Harp seconds that skip, repeat or step back, plus each clock step's deviation from the median. Reports the `Seconds` offset convention |
| HarpSync drift | `<device>.HarpSync.drift` | Residuals of a linear fit of Harp time on ONIX clock ticks for each hourly chunk, as analysis aligns the data. Per-chunk clock rate and worst residual |
| ONIX clock sequence | `<device>.<Stream>Clock` | Per-sample ONIX clock ticks that step backwards, repeat or jump |
| ONIX hub clock offset | `<device>.<Probe>HubSyncCounter` | Probe acquisition clock minus headstage hub clock, per file and as a distribution, with the largest deviations. A change means the headstage link dropped |

Ephys datasets are recorded on a separate machine under their own rig folder. Give the behaviour root and the ephys root together, behaviour first. The ONIX streams are discovered from the second root. The ephys samples for the window are selected by time shared with the behavior data, i.e. Harp Time. Each ephys epoch's HarpSync records place its clock on Harp time and only the samples inside the window are checked.

```python
results = run_qc(["/data/raw/<rig>/<behaviour_experiment>", "/data/raw/<ephys_rig>/<ephys_experiment>"], schema, start=start)
```

## Installation

1. Clone the repo.

2. From a terminal in the root of this repo:

```cmd
./deploy.cmd
```

This installs `uv` if needed, creates `.venv`, and installs all dependencies including dev tools.

## Quick start

```python
import pandas as pd
from aeon_qc import run_qc, generate_report
from aeon_qc.schemas import REGISTRY

root = "Z:/aeon/data/raw/AEON4/social0.4"
schema = REGISTRY["social04"]

results = run_qc(root, schema, start=pd.Timestamp("2024-09-01T00:00:00+00:00"))
generate_report(root, results, "qc_report.yaml", start=pd.Timestamp("2024-09-01T00:00:00+00:00"))
```

See the [run-qc tutorial](docs/tutorials/run-qc.md) for a full walkthrough.

## Batch runs

To run QC across all datasets defined in `benchmarks.yaml`:

```bash
uv run python scripts/run_benchmarks.py --benchmarks benchmarks.yaml --output benchmarks_output/
```

See the [batch-qc tutorial](docs/tutorials/batch-qc.md) for manifest format and options.

## Development

```bash
uv run ruff check src/         # lint
uv run pyright src/            # type check
```

## Citation

Sainsbury Wellcome Centre Foraging Behaviour Working Group. (2023). Aeon: An open-source platform to study the neural basis of ethological behaviours over naturalistic timescales. https://doi.org/10.5281/zenodo.8411157

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8411157.svg)](https://zenodo.org/doi/10.5281/zenodo.8411157)
