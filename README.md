# aeon_scratchpad
Scratchpad repo for Project Aeon

## Citation Policy

If you use this software, please cite it as below:

Sainsbury Wellcome Centre Foraging Behaviour Working Group. (2023). Aeon: An open-source platform to study the neural basis of ethological behaviours over naturalistic timescales,  https://doi.org/10.5281/zenodo.8411157

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8411157.svg)](https://zenodo.org/doi/10.5281/zenodo.8411157)

## Loom Escape Analysis (JF1×CAST)

Scripts for analysing mouse behavioural responses to looming stimuli across
JF1, CAST, and JF1×CAST F1/BC1 cohorts (~55 sessions, 40 Hz tracking).

### Pipeline

```
escape_kinematics_hmm.py   →   escape_score.py
```

**`escape_kinematics_hmm.py`** — Two-pass Kalman + HMM pipeline.

- Pass 1 (per session): fits a Kalman filter to the pre-loom centroid baseline,
  smooths all trials, extracts speed and acceleration.
- Pass 2 (pooled): trains a Gaussian HMM on speed/accel features from all mice,
  decodes per-trial state sequences and posterior probabilities (7 speed-sorted
  states, 0 = near-stationary, 6 = sprint).

Outputs per session: `trials_escape_kinematics_kalman.pkl`,
`trials_escape_kinematics_hmm_global.pkl`.

```bash
conda run -n escape_swc python escape_kinematics_hmm.py
conda run -n escape_swc python escape_kinematics_hmm.py --n_states 6 --hmm_scope strain
```

**`escape_score.py`** — Per-trial behavioural response score (0–1).

Combines six features into a continuous score ranking response quality from
indifference → brief acknowledgment → freezing → partial escape → full escape.
Features: peak speed ratio (F1), early burst speed (F2), directional efficiency
(F3), HMM state divergence (F4), response latency to sprint state (F5), shelter
approach (F6). Train/test split is at the mouse level, stratified by strain.

```bash
conda run -n escape_swc python escape_score.py --out_dir results/
conda run -n escape_swc python escape_score.py --no_hmm --out_dir results/  # kinematics only
```

### Data

Base path: `/ceph/branco/Dario/Escape_SWC/JF1xCAST/`  
Mouse metadata and ground-truth `P_escape_loom`: `JF1xCAST Mice.xlsx` (sheet: Behaviour, `RT_tagging == 1`).

### Environment

```bash
conda activate escape_swc
```
