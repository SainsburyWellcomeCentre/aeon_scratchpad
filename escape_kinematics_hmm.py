import os
# Must be set before numpy/OpenBLAS loads. Direct assignment (not setdefault) so
# it always overrides, preventing segfaults on high-core-count servers where
# OpenBLAS tries to allocate memory for more threads than its compiled-in limit.
os.environ["OMP_NUM_THREADS"] = "16"

"""
Kalman filtering + HMM behavioural state inference for loom-trial centroid data.

Adapted from zimoli02/2023-2024-SURFiN-Foraging-Project (MIT / BSD licence).
Core functions (filterLDS_SS_withMissingValues_np, smoothLDS_SS) are copied verbatim
from SURFiN Functions/inference.py.

Key adaptations vs the SURFiN pipeline:
  - dt = 1/40  (40 Hz native; SURFiN downsampled 50 Hz → 10 Hz so dt = 0.1)
  - 2 HMM features (speed + acceleration) — no skeleton body features
  - HMM state-count scan: np.arange(3, 15) instead of np.arange(3, 50)
  - Kalman parameters fitted via scipy L-BFGS-B (mirrors SURFiN's
    torch_lbfgs_optimize_SS_tracking_diagV0) on the pre-loom baseline (frames
    0:200) per session (analogous to SURFiN's 10-11 am calibration window)
  - Kalman filter run per trial; HMM trained on pooled trials then decoded
    back per trial

Pipeline structure (two passes):
  Pass 1 — per session: load centroid pkl → (optionally) piebald pre-filter →
            fit Kalman via EM → smooth trials → compute speed/accel features
  Pass 2 — per HMM group: pool features → model selection → fit one HMM →
            decode per session → save pkls + figures

Input  (per session): trials_top_visual_loom.bonsai_centroid.pkl
Output (per session): trials_escape_kinematics_kalman[_piebald].pkl        (Kalman cache)
                      trials_escape_kinematics_hmm_<scope>[_piebald].pkl   (HMM states)
Figures: /ceph/branco/Dario/Escape_SWC/JF1xCAST/HMM_results/<scope[_piebald]>/<scope>_model_selection.png
         /ceph/branco/Dario/Escape_SWC/JF1xCAST/HMM_results/<scope[_piebald]>/<mouse_id>_<session>_*.png

Usage:
  python escape_kinematics_hmm.py                          # all mice, one global HMM
  python escape_kinematics_hmm.py --hmm_scope strain       # one HMM per strain
  python escape_kinematics_hmm.py --piebald_filter         # k=3 median pre-filter for piebald mice
  python escape_kinematics_hmm.py --n_states 6             # skip model selection, use N states
  python escape_kinematics_hmm.py --refit_kalman           # ignore Kalman cache, recompute
"""

import argparse
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.optimize import minimize
import ssm
from kneed import KneeLocator

# ---------------------------------------------------------------------------
# Config (mirrors loom_trials_bonsai_centroid_overlay.py)
# ---------------------------------------------------------------------------
EXCEL_PATH        = Path(__file__).parent / "JF1xCAST Mice.xlsx"
DATA_ROOT         = Path("/ceph/branco/Dario/Escape_SWC/JF1xCAST")
PKL_IN_NAME       = "trials_top_visual_loom.bonsai_centroid.pkl"
PRE_FRAMES        = 200   # frames before loom onset
TRIAL_LEN         = 600   # total frames per trial
HMM_RESULTS_DIR   = DATA_ROOT / "HMM_results"


def kalman_cache_name(piebald_filter: bool) -> str:
    """Filename for the per-session Kalman cache pkl."""
    suffix = "_piebald" if piebald_filter else ""
    return f"trials_escape_kinematics_kalman{suffix}.pkl"


def hmm_out_name(scope_label: str, piebald_filter: bool) -> str:
    """Filename for the per-session HMM output pkl."""
    suffix = "_piebald" if piebald_filter else ""
    return f"trials_escape_kinematics_hmm_{scope_label}{suffix}.pkl"

# ---------------------------------------------------------------------------
# ============================================================
# Kalman filter + RTS smoother
# Copied verbatim from SURFiN Functions/inference.py
# ============================================================

def filterLDS_SS_withMissingValues_np(y, B, Q, m0, V0, Z, R):
    """Kalman filter (Shumway & Stoffer 2006). Handles NaN observations."""
    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")
    N = y.shape[1]
    M = B.shape[0]
    P = y.shape[0]
    xnn1 = np.empty(shape=[M, 1, N])
    Vnn1 = np.empty(shape=[M, M, N])
    xnn  = np.empty(shape=[M, 1, N])
    Vnn  = np.empty(shape=[M, M, N])
    innov = np.empty(shape=[P, 1, N])
    Sn    = np.empty(shape=[P, P, N])

    # k == 0
    xnn1[:, 0, 0] = B @ m0
    Vnn1[:, :, 0] = B @ V0 @ B.T + Q
    Stmp = Z @ Vnn1[:, :, 0] @ Z.T + R
    Sn[:, :, 0] = (Stmp + Stmp.T) / 2
    Sinv = np.linalg.inv(Sn[:, :, 0])
    K    = Vnn1[:, :, 0] @ Z.T @ Sinv
    innov[:, 0, 0] = y[:, 0] - (Z @ xnn1[:, :, 0]).squeeze()
    xnn[:, :, 0]   = xnn1[:, :, 0] + K @ innov[:, :, 0]
    Vnn[:, :, 0]   = Vnn1[:, :, 0] - K @ Z @ Vnn1[:, :, 0]
    logLike = (-N * P * np.log(2 * np.pi)
               - np.linalg.slogdet(Sn[:, :, 0])[1]
               - innov[:, :, 0].T @ Sinv @ innov[:, :, 0])

    for k in range(1, N):
        xnn1[:, :, k] = B @ xnn[:, :, k - 1]
        Vnn1[:, :, k] = B @ Vnn[:, :, k - 1] @ B.T + Q
        if np.any(np.isnan(y[:, k])):
            xnn[:, :, k]  = xnn1[:, :, k]
            Vnn[:, :, k]  = Vnn1[:, :, k]
        else:
            Stmp = Z @ Vnn1[:, :, k] @ Z.T + R
            Sn[:, :, k] = (Stmp + Stmp.T) / 2
            Sinv = np.linalg.inv(Sn[:, :, k])
            K    = Vnn1[:, :, k] @ Z.T @ Sinv
            innov[:, 0, k] = y[:, k] - (Z @ xnn1[:, :, k]).squeeze()
            xnn[:, :, k]   = xnn1[:, :, k] + K @ innov[:, :, k]
            Vnn[:, :, k]   = Vnn1[:, :, k] - K @ Z @ Vnn1[:, :, k]
        logLike = (logLike
                   - np.linalg.slogdet(Sn[:, :, k])[1]
                   - innov[:, :, k].T @ Sinv @ innov[:, :, k])

    logLike = 0.5 * logLike
    return {"xnn1": xnn1, "Vnn1": Vnn1, "xnn": xnn, "Vnn": Vnn,
            "innov": innov, "KN": K, "Sn": Sn, "logLike": logLike}


def smoothLDS_SS(B, xnn, Vnn, xnn1, Vnn1, m0, V0):
    """RTS smoother (backward pass). Copied verbatim from SURFiN."""
    if m0.ndim != 1:
        raise ValueError("mean must be 1 dimensional")
    N = xnn.shape[2]
    M = B.shape[0]
    xnN = np.empty(shape=[M, 1, N])
    VnN = np.empty(shape=[M, M, N])
    Jn  = np.empty(shape=[M, M, N])

    xnN[:, :, -1] = xnn[:, :, -1]
    VnN[:, :, -1] = Vnn[:, :, -1]

    epsilon = 1e-5
    for n in reversed(range(1, N)):
        try:
            Jn[:, :, n - 1] = Vnn[:, :, n - 1] @ B.T @ np.linalg.inv(Vnn1[:, :, n])
        except np.linalg.LinAlgError:
            Vnn1_reg = Vnn1[:, :, n] + epsilon * np.eye(Vnn1.shape[1])
            Jn[:, :, n - 1] = Vnn[:, :, n - 1] @ B.T @ np.linalg.inv(Vnn1_reg)
        xnN[:, :, n - 1] = (xnn[:, :, n - 1]
                             + Jn[:, :, n - 1] @ (xnN[:, :, n] - xnn1[:, :, n]))
        VnN[:, :, n - 1] = (Vnn[:, :, n - 1]
                             + Jn[:, :, n - 1]
                             @ (VnN[:, :, n] - Vnn1[:, :, n])
                             @ Jn[:, :, n - 1].T)

    Vnn1_reg = Vnn1[:, :, n] + epsilon * np.eye(Vnn1.shape[1])
    Jn[:, :, n - 1] = Vnn[:, :, n - 1] @ B.T @ np.linalg.inv(Vnn1_reg)
    J0  = V0 @ B.T @ np.linalg.inv(Vnn1[:, :, 0])
    x0N = np.expand_dims(m0, 1) + J0 @ (xnN[:, :, 0] - xnn1[:, :, 0])
    V0N = V0 + J0 @ (VnN[:, :, 0] - Vnn1[:, :, 0]) @ J0.T

    return {"xnN": xnN, "VnN": VnN, "Jn": Jn, "x0N": x0N, "V0N": V0N, "J0": J0}


# ---------------------------------------------------------------------------
# ============================================================
# L-BFGS parameter fitting
# Mirrors SURFiN's torch_lbfgs_optimize_SS_tracking_diagV0 from
# Functions/learning.py, using scipy L-BFGS-B in place of PyTorch.
# Optimises the same parameters (sigma_a, sqrt_diag_R, sqrt_diag_V0)
# against the same log-likelihood (filterLDS_SS_withMissingValues_np).
# ============================================================

def lbfgs_optimize_SS_tracking(y, B, sigma_a0, Qe, Z,
                                sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                vars_to_estimate=None,
                                max_iter=200, tol=1e-9):
    """
    Fit Kalman parameters by maximising the log-likelihood with L-BFGS-B.

    Parameters optimised (matching SURFiN's torch variant):
      sigma_a      — scalar process-noise scale
      sqrt_diag_R  — sqrt of observation-noise diagonal (length 2)
      sqrt_diag_V0 — sqrt of initial-covariance diagonal (length 6)
    m0 is fixed (not optimised).

    Returns a dict with keys: estimates, log_like, termination_info.
    """
    if vars_to_estimate is None:
        vars_to_estimate = {"sigma_a": True, "sqrt_diag_R": True,
                            "m0": False, "sqrt_diag_V0": True}

    sqrt_diag_R_0  = np.asarray(sqrt_diag_R_0,  dtype=np.float64)
    sqrt_diag_V0_0 = np.asarray(sqrt_diag_V0_0, dtype=np.float64)
    m0             = np.asarray(m0_0,            dtype=np.float64)

    # Pack free parameters into a single vector for scipy
    x0_parts = []
    if vars_to_estimate["sigma_a"]:      x0_parts.append(np.array([sigma_a0]))
    if vars_to_estimate["sqrt_diag_R"]:  x0_parts.append(sqrt_diag_R_0)
    if vars_to_estimate["sqrt_diag_V0"]: x0_parts.append(sqrt_diag_V0_0)
    x0_vec = np.concatenate(x0_parts)

    n_sigma = 1                    if vars_to_estimate["sigma_a"]      else 0
    n_R     = len(sqrt_diag_R_0)   if vars_to_estimate["sqrt_diag_R"]  else 0
    n_V0    = len(sqrt_diag_V0_0)  if vars_to_estimate["sqrt_diag_V0"] else 0

    log_like_history = []

    def neg_log_like(x):
        idx = 0
        sigma_a = x[idx] if vars_to_estimate["sigma_a"] else sigma_a0
        if vars_to_estimate["sigma_a"]: idx += n_sigma
        sqrt_diag_R = x[idx:idx + n_R] if vars_to_estimate["sqrt_diag_R"] else sqrt_diag_R_0
        if vars_to_estimate["sqrt_diag_R"]: idx += n_R
        sqrt_diag_V0 = x[idx:idx + n_V0] if vars_to_estimate["sqrt_diag_V0"] else sqrt_diag_V0_0

        Q  = Qe * sigma_a ** 2
        R  = np.diag(sqrt_diag_R ** 2)
        V0 = np.diag(sqrt_diag_V0 ** 2)

        kf = filterLDS_SS_withMissingValues_np(y=y, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
        ll = kf["logLike"].item()
        log_like_history.append(ll)
        return -ll

    result = minimize(neg_log_like, x0_vec, method="L-BFGS-B",
                      options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-7})

    x = result.x
    idx = 0
    sigma_a_fit = x[idx] if vars_to_estimate["sigma_a"] else sigma_a0
    if vars_to_estimate["sigma_a"]: idx += n_sigma
    sqrt_diag_R_fit = x[idx:idx + n_R] if vars_to_estimate["sqrt_diag_R"] else sqrt_diag_R_0
    if vars_to_estimate["sqrt_diag_R"]: idx += n_R
    sqrt_diag_V0_fit = x[idx:idx + n_V0] if vars_to_estimate["sqrt_diag_V0"] else sqrt_diag_V0_0

    estimates = {}
    if vars_to_estimate["sigma_a"]:      estimates["sigma_a"]      = float(sigma_a_fit)
    if vars_to_estimate["sqrt_diag_R"]:  estimates["sqrt_diag_R"]  = sqrt_diag_R_fit
    if vars_to_estimate["sqrt_diag_V0"]: estimates["sqrt_diag_V0"] = sqrt_diag_V0_fit

    return {"estimates": estimates, "log_like": log_like_history,
            "termination_info": result.message}


# ---------------------------------------------------------------------------
# ============================================================
# Parameter initialisation
# Adapted from SURFiN Functions/mouse.py  Kinematics.Get_Manual_Parameters
# ONLY change: dt = 1/40 (40 Hz native) vs dt = 0.1 (SURFiN, 10 Hz)
# ============================================================

def get_initial_parameters(x0=0.0, y0=0.0):
    dt = 1 / 40        # ← only adaptation: 40 Hz (SURFiN used 0.1 = 10 Hz)

    sigma_a          = 20
    sigma_x          = 1
    sigma_y          = 1
    sqrt_diag_V0     = 1e-3

    m0 = np.array([x0, 0.0, 0.0, y0, 0.0, 0.0], dtype=np.double)
    V0 = np.diag(np.ones(6) * sqrt_diag_V0 ** 2)

    B = np.array([[1, dt, dt**2/2, 0,  0,       0      ],
                  [0,  1, dt,      0,  0,       0      ],
                  [0,  0,  1,      0,  0,       0      ],
                  [0,  0,  0,      1,  dt,      dt**2/2],
                  [0,  0,  0,      0,  1,       dt     ],
                  [0,  0,  0,      0,  0,       1      ]], dtype=np.double)

    Qe = np.array([[dt**4/4, dt**3/2, dt**2/2, 0,       0,       0      ],
                   [dt**3/2, dt**2,   dt,       0,       0,       0      ],
                   [dt**2/2, dt,      1,        0,       0,       0      ],
                   [0,       0,       0,        dt**4/4, dt**3/2, dt**2/2],
                   [0,       0,       0,        dt**3/2, dt**2,   dt     ],
                   [0,       0,       0,        dt**2/2, dt,      1      ]], dtype=np.double)
    Q = sigma_a**2 * Qe

    Z = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=np.double)
    R = np.diag([sigma_x**2, sigma_y**2])

    return {"sigma_a": sigma_a, "Qe": Qe, "Q": Q,
            "m0": m0, "V0": V0, "B": B, "Z": Z, "R": R}


# ---------------------------------------------------------------------------
# ============================================================
# Kalman parameter fitting (EM on pre-loom baseline)
# ============================================================

def fit_kalman_parameters(x_all, y_all):
    """
    Fit sigma_a and R via EM using the pre-loom baseline (frames 0:PRE_FRAMES)
    from all trials.  m0 is fixed to the mean initial position; V0 is also
    fitted.  Analogous to SURFiN's 10-11 am calibration window.

    y_cal shape: (2, n_trials * PRE_FRAMES) — concatenated x/y rows
    """
    # Use only pre-loom frames from all trials, drop rows with any NaN
    x_pre = x_all[:, :PRE_FRAMES]   # (n_trials, PRE_FRAMES)
    y_pre = y_all[:, :PRE_FRAMES]

    # Identify trials whose entire baseline is valid
    valid = ~(np.isnan(x_pre).any(axis=1) | np.isnan(y_pre).any(axis=1))
    x_pre = x_pre[valid]
    y_pre = y_pre[valid]

    if x_pre.shape[0] == 0:
        print("  WARNING: no fully-valid pre-loom windows; using manual parameters")
        return get_initial_parameters()

    # Concatenate into one long sequence: shape (2, n_valid * PRE_FRAMES)
    y_cal = np.vstack([x_pre.ravel(), y_pre.ravel()])  # (2, T)

    # Initialise with manual parameters, m0 from mean starting position
    x0 = float(np.nanmean(x_pre[:, 0]))
    y0 = float(np.nanmean(y_pre[:, 0]))
    p  = get_initial_parameters(x0=x0, y0=y0)

    print(f"  Fitting Kalman parameters via L-BFGS on {x_pre.shape[0]} pre-loom baseline windows…")
    result = lbfgs_optimize_SS_tracking(
        y=y_cal, B=p["B"], sigma_a0=p["sigma_a"], Qe=p["Qe"], Z=p["Z"],
        sqrt_diag_R_0=np.sqrt(np.diag(p["R"])),
        m0_0=p["m0"],
        sqrt_diag_V0_0=np.sqrt(np.diag(p["V0"])),
        vars_to_estimate={"sigma_a": True, "sqrt_diag_R": True,
                          "m0": False, "sqrt_diag_V0": True})

    print(f"  L-BFGS termination: {result['termination_info']}")
    est = result["estimates"]
    sigma_a      = est["sigma_a"]
    sqrt_diag_R  = est["sqrt_diag_R"]
    sqrt_diag_V0 = est["sqrt_diag_V0"]
    print(f"  Fitted sigma_a={sigma_a:.4f}")
    print(f"  Fitted sqrt_diag_R={sqrt_diag_R}")

    # Build full parameter dict with fitted values
    fitted = get_initial_parameters(x0=x0, y0=y0)
    fitted["sigma_a"] = sigma_a
    fitted["Q"]       = sigma_a**2 * fitted["Qe"]
    fitted["R"]       = np.diag(sqrt_diag_R ** 2)
    fitted["V0"]      = np.diag(sqrt_diag_V0 ** 2)
    fitted["log_like"] = result["log_like"]
    return fitted


# ---------------------------------------------------------------------------
# ============================================================
# Per-trial Kalman smoothing
# ============================================================

def smooth_trial(x_trial, y_trial, params):
    """
    Run Kalman filter + RTS smoother on a single trial.
    x_trial, y_trial: 1-D arrays of length TRIAL_LEN (may contain NaN)
    Returns smoothed state array xnN of shape (6, 1, TRIAL_LEN).
    """
    y_obs = np.vstack([x_trial, y_trial])   # (2, T)

    # Set m0 from first valid observation
    first_valid = np.where(np.isfinite(x_trial) & np.isfinite(y_trial))[0]
    if len(first_valid) == 0:
        return None
    m0 = params["m0"].copy()
    m0[0] = x_trial[first_valid[0]]
    m0[3] = y_trial[first_valid[0]]

    kf = filterLDS_SS_withMissingValues_np(
        y=y_obs, B=params["B"], Q=params["Q"],
        m0=m0, V0=params["V0"], Z=params["Z"], R=params["R"])
    ks = smoothLDS_SS(B=params["B"], xnn=kf["xnn"], Vnn=kf["Vnn"],
                      xnn1=kf["xnn1"], Vnn1=kf["Vnn1"], m0=m0, V0=params["V0"])
    return ks["xnN"]   # (6, 1, T)


# ---------------------------------------------------------------------------
# ============================================================
# Feature extraction
# Matches SURFiN 'Kinematics' mode: speed + acceleration
# ============================================================

def compute_features(xnN):
    """
    xnN: (6, 1, T) smoothed state.
    State layout: [x, vx, ax, y, vy, ay]
    Returns speed and acceleration arrays of length T.
    """
    vx    = xnN[1, 0, :]
    vy    = xnN[4, 0, :]
    ax    = xnN[2, 0, :]
    ay    = xnN[5, 0, :]
    speed = np.sqrt(vx**2 + vy**2)
    accel = np.sqrt(ax**2 + ay**2)
    return speed, accel


# ---------------------------------------------------------------------------
# ============================================================
# Piebald pre-filter (k=3 median filter on x/y before Kalman)
# ============================================================

def is_piebald(coat):
    """Return True if coat value indicates a Piebald Black mouse."""
    if not isinstance(coat, str):
        return False
    return "piebald black" in coat.strip().lower()


def apply_median_filter(x, k):
    """
    3-frame median filter along the time axis (axis=1) of a (n_trials, T) array.
    NaN frames are linearly interpolated before filtering then restored, so the
    filter does not propagate NaNs into valid data.
    """
    out = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        row   = x[i]
        valid = np.isfinite(row)
        if valid.sum() < k:
            continue
        tmp = row.copy()
        tmp[~valid] = np.interp(np.where(~valid)[0],
                                np.where(valid)[0], row[valid])
        out[i]         = scipy_median_filter(tmp, size=k)
        out[i, ~valid] = np.nan
    return out


# ---------------------------------------------------------------------------
# ============================================================
# HMM fitting — adapted from SURFiN Functions/mouse.py HMM class
# ============================================================

def fit_model_without_saving(features, n_state):
    """
    Fit a Gaussian HMM and return the EM log-likelihoods (one per iteration).
    Copied from SURFiN HMM.Fit_Model_without_Saving.
    features: (T, 2) array of [speed, accel]
    """
    model = ssm.HMM(n_state, features.shape[1], observations="gaussian")
    lls   = model.fit(features, method="em", num_iters=500, tolerance=1,
                      init_method="kmeans")
    return lls   # shape (num_iters,)


def fit_final_model(features, n_state):
    """
    Fit Gaussian HMM, sort states by ascending mean speed, return model +
    sorted parameters.  Adapted from SURFiN HMM.Fit_Model.
    features: (T, 2) array of [speed, accel]
    """
    model = ssm.HMM(n_state, features.shape[1], observations="gaussian")
    lls   = model.fit(features, method="em", num_iters=500, tolerance=1,
                      init_method="kmeans")

    # Sort states by ascending mean speed (first feature, index 0)
    state_mean_speed = model.observations.params[0].T[0]
    index = np.argsort(state_mean_speed)

    params_mean   = model.observations.params[0][index].T            # (2, n_state)
    params_covar  = model.observations.params[1][index]              # (n_state, 2, 2)
    params_var    = np.array([[params_covar[i][j][j]
                               for j in range(features.shape[1])]
                              for i in range(n_state)]).T            # (2, n_state)
    TransM = model.transitions.transition_matrix[index].T[index].T  # (n_state, n_state)

    return {
        "model":       model,
        "index":       index,
        "params_mean": params_mean,
        "params_var":  params_var,
        "params_covar": params_covar,
        "TransM":      TransM,
        "lls":         lls,
    }


def decode_states(model, index, features):
    """Decode most-likely state sequence and remap to speed-sorted labels."""
    raw_states = model.most_likely_states(features)
    new_states = np.empty_like(raw_states)
    for i, val in enumerate(index):
        new_states[raw_states == val] = i
    return new_states


def decode_posteriors(model, index, features):
    """
    Compute smoothed posterior state probabilities and remap to speed-sorted order.

    Returns Ez of shape (T, n_states) where Ez[t, s] = P(state=s at frame t | all observations).
    Column order matches the speed-sorted state labels (0=slowest).
    """
    Ez, _, _ = model.expected_states(features)   # (T, K), rows sum to 1
    return Ez[:, index]                           # reorder columns to speed-sorted order


# ---------------------------------------------------------------------------
# ============================================================
# Model selection — adapted from SURFiN Scripts/SingleMouse.py
# Display_Model_Selection
# ============================================================

def display_model_selection(features, N, out_path):
    """
    Fit HMMs for each state count in N, plot log-likelihood + delta.
    Adapted from SURFiN Display_Model_Selection.
    N: 1-D array of state counts to test.
    """
    loglikelihoods = []
    for n in N:
        print(f"  Model selection: fitting n={n} states…")
        lls = fit_model_without_saving(features, n)
        loglikelihoods.append(lls)
        print(f"    final ll={lls[-1]:.4f}")

    # Extract final ll from each run (lengths differ now models converge early)
    final_lls = np.array([lls[-1] for lls in loglikelihoods])

    # Normalise by total number of training frames (cf. SURFiN: /= points)
    points = features.shape[0]
    ll_per_point = final_lls / points
    delta_ll     = ll_per_point[1:] - ll_per_point[:-1]

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    axs[0].scatter(N, ll_per_point)
    axs[0].plot(N, ll_per_point)
    axs[0].set_xticks(np.arange(N[0], N[-1] + 1, 2))

    axs[1].scatter(N[1:], delta_ll)
    axs[1].plot(N[1:], delta_ll)
    axs[1].set_xticks(np.arange(N[0], N[-1] + 1, 2))

    for i in range(2):
        axs[i].set_xlabel("State Number", fontsize=30)
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].tick_params(axis="both", which="major", labelsize=20)
    axs[0].set_ylabel("Log Likelihood per Frame", fontsize=30)
    axs[1].set_ylabel(r"$\Delta$ Log Likelihood per Frame", fontsize=30)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"  Model selection figure saved: {out_path}")

    return ll_per_point, delta_ll


# ---------------------------------------------------------------------------
# ============================================================
# Visualisations — adapted from SURFiN Functions/result.py
# ============================================================

def plot_state_features(params_mean, params_var, TransM, n_state, out_path):
    """
    Bar charts of mean speed/acceleration per state + transition matrix heatmap.
    Adapted from SURFiN result.py HMM.Model_Features.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(n_state)
    feature_names = ["speed (px/s)", "accel (px/s²)"]
    colors = ["steelblue", "darkorange"]

    for feat_i in range(2):
        ax = axs[feat_i]
        ax.bar(x, params_mean[feat_i], yerr=np.sqrt(params_var[feat_i]),
               color=colors[feat_i], capsize=4, alpha=0.85)
        ax.set_xlabel("State (sorted by speed)", fontsize=12)
        ax.set_ylabel(feature_names[feat_i], fontsize=12)
        ax.set_xticks(x)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    im = axs[2].imshow(TransM, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axs[2].set_xlabel("To state", fontsize=12)
    axs[2].set_ylabel("From state", fontsize=12)
    axs[2].set_title("Transition matrix", fontsize=12)
    plt.colorbar(im, ax=axs[2])

    plt.suptitle(f"HMM state features  (n={n_state})", fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  State features figure saved: {out_path}")


def plot_event_heatmap(states_all, n_state, out_path):
    """
    Heatmap of HMM state sequences across trials, aligned to loom onset.
    Adapted from SURFiN result.py HMM.Characterize_Timepoints.EventHeatmap.
    Rows = trials, columns = time frames (loom onset at PRE_FRAMES).
    Layout: heatmap on top, mean state probability below.
    """
    t = (np.arange(TRIAL_LEN) - PRE_FRAMES) / 40   # seconds relative to loom

    heatmap_h = max(3, len(states_all) * 0.15 + 2)
    prob_h    = 3
    fig = plt.figure(figsize=(14, heatmap_h + prob_h + 1))
    gs  = fig.add_gridspec(2, 1,
                           height_ratios=[heatmap_h, prob_h],
                           hspace=0.35)
    ax_heat = fig.add_subplot(gs[0])
    ax_prob = fig.add_subplot(gs[1])

    # Top: state sequence heatmap
    im = ax_heat.imshow(states_all, aspect="auto", interpolation="nearest",
                        cmap=_state_cmap(n_state), vmin=0, vmax=n_state - 1,
                        extent=[t[0], t[-1], len(states_all), 0])
    ax_heat.axvline(0, color="red", lw=1.5, ls="--", label="loom onset")
    ax_heat.set_xlabel("Time relative to loom onset (s)", fontsize=12)
    ax_heat.set_ylabel("Trial", fontsize=12)
    ax_heat.set_title("HMM state sequence per trial", fontsize=12)
    ax_heat.legend(fontsize=9, loc="upper left")
    plt.colorbar(im, ax=ax_heat, label="State (0=slowest)")

    # Bottom: mean state probability across trials over time
    state_probs = np.zeros((n_state, TRIAL_LEN))
    for s in range(n_state):
        state_probs[s] = (states_all == s).mean(axis=0)

    for s in range(n_state):
        ax_prob.plot(t, state_probs[s], label=f"s{s}", lw=1.5,
                     color=_state_color(s))
    ax_prob.axvline(0, color="red", lw=1.5, ls="--")
    ax_prob.set_xlabel("Time (s)", fontsize=12)
    ax_prob.set_ylabel("Fraction of trials", fontsize=12)
    ax_prob.set_title("Mean state probability", fontsize=12)
    ax_prob.legend(fontsize=7, frameon=False)
    ax_prob.spines["top"].set_visible(False)
    ax_prob.spines["right"].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Event heatmap figure saved: {out_path}")


def plot_event_positions(smoothed_x, smoothed_y, states_all, n_state, out_path,
                         n_examples=8):
    """
    x,y trajectories coloured by HMM state.
    Adapted from SURFiN result.py HMM.Characterize_Timepoints.EventPosition.
    """
    valid = np.where(~np.isnan(smoothed_x[:, 0]))[0]
    indices = valid[:n_examples]

    ncols = 4
    nrows = math.ceil(len(indices) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = np.array(axs).reshape(-1)

    for col, i in enumerate(indices):
        ax = axs[col]
        x  = smoothed_x[i]
        y  = smoothed_y[i]
        st = states_all[i]

        # Plot trajectory segments coloured by state
        for t in range(len(x) - 1):
            if np.isfinite(x[t]) and np.isfinite(x[t + 1]):
                ax.plot([x[t], x[t + 1]], [y[t], y[t + 1]],
                        color=_state_color(st[t]), lw=0.8, alpha=0.8)

        # Mark loom onset
        if np.isfinite(x[PRE_FRAMES]):
            ax.scatter(x[PRE_FRAMES], y[PRE_FRAMES],
                       color="red", s=50, zorder=5, label="loom")
        ax.set_title(f"Trial {i}", fontsize=10)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for ax in axs[len(indices):]:
        ax.set_visible(False)

    # Shared colour legend
    handles = [plt.Line2D([0], [0], color=_state_color(s), lw=2, label=f"s{s}")
               for s in range(n_state)]
    fig.legend(handles=handles, loc="lower right", fontsize=9, ncol=n_state,
               title="State (0=slowest)")
    plt.suptitle("Trajectories coloured by HMM state  (red = loom onset)",
                 fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Event positions figure saved: {out_path}")


# ---------------------------------------------------------------------------
# ============================================================
# Sequence distance helpers — adapted from SURFiN result.py
# ============================================================

_STRAIN_PALETTE = {"CAST": "#2196F3", "JF1": "#F44336", "_JF1xCAST": "#9C27B0"}
_STRAIN_ORDER   = ["CAST", "JF1", "_JF1xCAST"]


def _state_color(s: int):
    """Return a consistent tab10 color for HMM state s (0-indexed)."""
    return plt.get_cmap("tab10")(s / 10)


def _state_cmap(n_states: int):
    """ListedColormap with consistent tab10 colors for use in imshow."""
    return matplotlib.colors.ListedColormap([_state_color(s) for s in range(n_states)])


def _kl_div_gaussian(p_means, p_variances, q_means, q_variances):
    """KL divergence between two multivariate Gaussians. Copied from SURFiN."""
    q_inv      = np.linalg.inv(q_variances)
    trace_term = np.trace(q_inv @ p_variances)
    diff       = q_means - p_means
    mean_term  = float(diff @ q_inv @ diff)
    det_term   = np.log(np.linalg.det(q_variances) / np.linalg.det(p_variances))
    return max(0.0, 0.5 * (trace_term + mean_term - len(p_means) + det_term))


def _build_kl_matrix(params_mean, params_var, n_states):
    """
    n_states × n_states matrix of pairwise KL divergences between HMM state
    Gaussians. params_mean/params_var shape: (n_features, n_states).
    """
    kl = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            kl[i, j] = _kl_div_gaussian(
                params_mean[:, i], np.diag(params_var[:, i]),
                params_mean[:, j], np.diag(params_var[:, j]))
    return kl


def _find_dominant_sequence(states_matrix):
    """
    For each timeframe (column) return the modal state across rows.
    states_matrix: (n_trials, T); −1 values (invalid frames) are excluded.
    Adapted from SURFiN Find_Event_Sequence.
    """
    seq = []
    for col in states_matrix.T:
        valid = col[col >= 0]
        if valid.size == 0:
            seq.append(0)
        else:
            vals, counts = np.unique(valid, return_counts=True)
            seq.append(int(vals[np.argmax(counts)]))
    return np.array(seq)


def _compare_sequences(seq1, seq2, kl_matrix):
    """
    KL-weighted edit distance between two state sequences via DP.
    Adapted verbatim from SURFiN Compare_Sequence.
    """
    nx, ny = len(seq1), len(seq2)
    cost = np.zeros((nx + 1, ny + 1))
    cost[0, 1:] = np.inf
    cost[1:, 0] = np.inf
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            cost[i, j] = kl_matrix[int(seq1[i - 1]), int(seq2[j - 1])]
            cost[i, j] += min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[-1, -1]


def plot_group_ethogram(decoded_sessions, n_states, out_path):
    """
    Cross-session ethogram: all valid trials pooled, sorted by strain then
    mouse. Left colour strip indicates strain; white horizontal lines separate
    strains.
    Adapted from SURFiN EventHeatmap, extended to pool across sessions.

    decoded_sessions: list of dicts with keys strain_folder, mouse_id, states_all.
    states_all: (n_trials, TRIAL_LEN) int array, −1 for undecodable trials.
    """
    def _sort_key(d):
        sf = d["strain_folder"]
        return (_STRAIN_ORDER.index(sf) if sf in _STRAIN_ORDER else 3, d["mouse_id"])

    sorted_sess = sorted(decoded_sessions, key=_sort_key)

    all_states, trial_strains = [], []
    for d in sorted_sess:
        sa = d["states_all"]
        for i in range(sa.shape[0]):
            if (sa[i] >= 0).any():
                all_states.append(sa[i])
                trial_strains.append(d["strain_folder"])

    if not all_states:
        print("  [group ethogram] No valid trials — skipping")
        return

    all_states = np.array(all_states)   # (N, TRIAL_LEN)
    t       = (np.arange(TRIAL_LEN) - PRE_FRAMES) / 40
    n_total = len(all_states)

    heatmap_h = max(4, n_total * 0.10 + 2)
    prob_h    = 3
    fig = plt.figure(figsize=(16, heatmap_h + prob_h + 1))
    gs  = fig.add_gridspec(2, 2,
                           width_ratios=[0.02, 1],
                           height_ratios=[heatmap_h, prob_h],
                           hspace=0.35, wspace=0.05)
    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_prob = fig.add_subplot(gs[1, :])

    # Left strain colour strip
    strip = np.array([[matplotlib.colors.to_rgba(
        _STRAIN_PALETTE.get(s, "#888888"))] for s in trial_strains])
    ax_bar.imshow(strip, aspect="auto", interpolation="nearest")
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])

    # Heatmap
    im = ax_heat.imshow(all_states, aspect="auto", interpolation="nearest",
                        cmap=_state_cmap(n_states), vmin=0, vmax=n_states - 1,
                        extent=[t[0], t[-1], n_total, 0])
    ax_heat.axvline(0, color="white", lw=1.0, ls="--")
    ax_heat.set_xlabel("Time relative to loom onset (s)", fontsize=12)
    ax_heat.set_ylabel("Trial (sorted by strain)", fontsize=12)
    ax_heat.set_title("HMM state sequences — all sessions", fontsize=12)
    plt.colorbar(im, ax=ax_heat, label="State (0=slowest)", fraction=0.02)

    # Strain boundary lines + legend
    legend_handles, prev_strain = [], None
    for row_i, s in enumerate(trial_strains):
        if s != prev_strain:
            if row_i > 0:
                ax_heat.axhline(row_i, color="white", lw=1.5)
            legend_handles.append(
                mpatches.Rectangle((0, 0), 1, 1,
                               fc=_STRAIN_PALETTE.get(s, "#888888"),
                               label=s.lstrip("_")))
            prev_strain = s
    ax_heat.legend(handles=legend_handles, loc="upper left",
                   fontsize=9, framealpha=0.7)

    # Mean state probability (all trials pooled)
    for s in range(n_states):
        ax_prob.plot(t, (all_states == s).mean(axis=0), lw=1.5, label=f"s{s}",
                     color=_state_color(s))
    ax_prob.axvline(0, color="red", lw=1.5, ls="--")
    ax_prob.set_xlabel("Time (s)", fontsize=12)
    ax_prob.set_ylabel("Fraction of trials", fontsize=12)
    ax_prob.set_title("Mean state probability", fontsize=12)
    ax_prob.legend(fontsize=7, frameon=False)
    ax_prob.spines["top"].set_visible(False)
    ax_prob.spines["right"].set_visible(False)

    plt.suptitle("Group ethogram  (sorted by strain)", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Group ethogram saved: {out_path}")


def plot_sequence_distance_matrix(decoded_sessions, hmm_result, n_states, out_path):
    """
    Mouse × mouse matrix of dominant escape-sequence distances.
    For each mouse, trials are pooled across sessions and the dominant sequence
    is the modal state at each frame in a ±1.5 s window around loom onset
    (frames [PRE_FRAMES−60 : PRE_FRAMES+60] at 40 Hz).
    Distance = KL-weighted DP edit distance (SURFiN Compare_Sequence).
    Adapted from SURFiN Summary.Event_Sequence.
    """
    kl_mat = _build_kl_matrix(
        hmm_result["params_mean"], hmm_result["params_var"], n_states)
    kl_mat = np.log10(kl_mat + 1)   # log-scale as in SURFiN

    win_start = PRE_FRAMES - 60     # ±1.5 s at 40 Hz
    win_end   = PRE_FRAMES + 60

    # Aggregate trials per mouse across sessions
    mouse_data: dict = {}
    for d in decoded_sessions:
        mid = d["mouse_id"]
        if mid not in mouse_data:
            mouse_data[mid] = {"strain_folder": d["strain_folder"], "rows": []}
        sa = d["states_all"]
        for i in range(sa.shape[0]):
            if (sa[i] >= 0).any():
                mouse_data[mid]["rows"].append(sa[i, win_start:win_end])

    mouse_data = {m: v for m, v in mouse_data.items() if v["rows"]}
    if len(mouse_data) < 2:
        print("  [sequence distance] Fewer than 2 mice with valid trials — skipping")
        return

    def _sort_key(mid):
        sf = mouse_data[mid]["strain_folder"]
        return (_STRAIN_ORDER.index(sf) if sf in _STRAIN_ORDER else 3, mid)

    mice_sorted = sorted(mouse_data.keys(), key=_sort_key)

    # Dominant sequence per mouse (modal state at each frame)
    sequences = {mid: _find_dominant_sequence(np.array(mouse_data[mid]["rows"]))
                 for mid in mice_sorted}

    # Pairwise distance matrix
    n_mice = len(mice_sorted)
    dist_matrix = np.zeros((n_mice, n_mice))
    for i, mi in enumerate(mice_sorted):
        for j, mj in enumerate(mice_sorted):
            if i != j:
                dist_matrix[i, j] = _compare_sequences(
                    sequences[mi], sequences[mj], kl_mat)

    strains_sorted = [mouse_data[m]["strain_folder"] for m in mice_sorted]
    tick_colors    = [_STRAIN_PALETTE.get(s, "#888888") for s in strains_sorted]

    fig, ax = plt.subplots(figsize=(max(8, n_mice * 0.7 + 2),
                                    max(7, n_mice * 0.7 + 1)))
    pos_vals = dist_matrix[dist_matrix > 0]
    vmax = float(np.percentile(pos_vals, 95)) if pos_vals.size > 0 else 1.0
    im = ax.imshow(dist_matrix, cmap="RdBu_r", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="KL-weighted edit distance (log-scaled)")

    ax.set_xticks(np.arange(n_mice))
    ax.set_yticks(np.arange(n_mice))
    ax.set_xticklabels(mice_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mice_sorted, fontsize=8)

    for tick, col in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(col)
    for tick, col in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(col)

    # Numerical annotations
    for i in range(n_mice):
        for j in range(n_mice):
            text_col = "white" if dist_matrix[i, j] > vmax * 0.6 else "black"
            ax.text(j, i, f"{dist_matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=6, color=text_col)

    # Strain boundary lines
    prev = None
    for k, s in enumerate(strains_sorted):
        if s != prev and k > 0:
            ax.axhline(k - 0.5, color="white", lw=1.5)
            ax.axvline(k - 0.5, color="white", lw=1.5)
        prev = s

    # Strain legend
    seen, handles = set(), []
    for s in strains_sorted:
        if s not in seen:
            handles.append(mpatches.Rectangle((0, 0), 1, 1,
                                          fc=_STRAIN_PALETTE.get(s, "#888888"),
                                          label=s.lstrip("_")))
            seen.add(s)
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.8)
    ax.set_title("Dominant escape sequence distance  (mouse × mouse)", fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sequence distance matrix saved: {out_path}")


# ---------------------------------------------------------------------------
# ============================================================
# Pass 1 — per-session Kalman
# ============================================================

def run_kalman_session(pkl_path, mouse_id, strain, coat, piebald_filter,
                       refit_kalman=False):
    """
    Load centroid pkl, optionally apply piebald median pre-filter, fit Kalman
    parameters via EM, smooth all trials and compute speed/accel features.

    Results are cached to a pkl alongside the input. On subsequent calls with
    the same piebald_filter setting the cache is loaded directly, skipping the
    EM fitting and smoothing. Pass refit_kalman=True to force recomputation.

    Returns a dict with all data needed for HMM fitting and saving, or None
    on failure.
    """
    sync_dir   = pkl_path.parent
    cache_path = sync_dir / kalman_cache_name(piebald_filter)

    # Load from cache if available and not forcing refit
    if cache_path.exists() and not refit_kalman:
        print(f"  Loading Kalman cache: {cache_path.name}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        cached["pkl_path"]      = pkl_path
        cached["strain_folder"] = strain_to_folder(strain)
        cached["fig_prefix"]    = f"{mouse_id}_{pkl_path.parent.parent.name}"
        return cached

    print(f"  Loading {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    x_all = np.array(data["x_position_trace"], dtype=np.float64)  # (n_trials, 600)
    y_all = np.array(data["y_position_trace"],  dtype=np.float64)
    n_trials = x_all.shape[0]
    print(f"  {n_trials} trials loaded")

    # Optionally apply k=3 median pre-filter for piebald mice
    if piebald_filter and is_piebald(coat):
        print(f"  Applying k=3 median pre-filter (piebald coat: {coat})")
        x_all = apply_median_filter(x_all, k=3)
        y_all = apply_median_filter(y_all, k=3)

    # Fit Kalman parameters on this session's pre-loom baseline
    params = fit_kalman_parameters(x_all, y_all)

    # Smooth each trial
    smoothed_x = np.full_like(x_all, np.nan)
    smoothed_y = np.full_like(y_all, np.nan)
    speed_all  = np.full_like(x_all, np.nan)
    accel_all  = np.full_like(x_all, np.nan)

    valid_trials = []
    for i in range(n_trials):
        xnN = smooth_trial(x_all[i], y_all[i], params)
        if xnN is None:
            continue
        smoothed_x[i] = xnN[0, 0, :]
        smoothed_y[i] = xnN[3, 0, :]
        spd, acc       = compute_features(xnN)
        speed_all[i]   = spd
        accel_all[i]   = acc
        valid_trials.append(i)

    print(f"  {len(valid_trials)} trials smoothed successfully")

    sr = {
        "pkl_path":      pkl_path,
        "mouse_id":      mouse_id,
        "strain":        strain,
        "strain_folder": strain_to_folder(strain),
        "fig_prefix":    f"{mouse_id}_{pkl_path.parent.parent.name}",
        "x_all":         x_all,
        "y_all":         y_all,
        "smoothed_x":    smoothed_x,
        "smoothed_y":    smoothed_y,
        "speed_all":     speed_all,
        "accel_all":     accel_all,
        "valid_trials":  valid_trials,
        "kalman_params": {k: v for k, v in params.items() if k not in ("Q", "Qe")},
        "n_trials":      n_trials,
    }

    # Save cache (exclude runtime-only fields that are re-set on load)
    cache_data = {k: v for k, v in sr.items()
                  if k not in ("pkl_path", "strain_folder", "fig_prefix")}
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Kalman cache saved: {cache_path.name}")

    return sr


# ---------------------------------------------------------------------------
# ============================================================
# Pass 2 — pooled HMM for a group of sessions
# ============================================================

def run_hmm_group(session_results, scope_label, n_states_arg, piebald_filter):
    """
    Pool speed/accel features across all sessions in session_results, fit one
    HMM, decode back per session, save output pkls and figures.

    scope_label:    string used for figure and pkl filenames
                    (e.g. 'global', 'JF1', 'CAST', '_JF1xCAST')
    piebald_filter: used to construct the output pkl filename so runs with
                    different pre-processing don't overwrite each other.

    Output pkl contains only HMM-derived fields (states, model params).
    Kinematics (smoothed traces, speed, accel) live in the Kalman cache pkl.
    """
    run_label = f"{scope_label}{'_piebald' if piebald_filter else ''}"
    fig_dir   = HMM_RESULTS_DIR / run_label
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Pool features from all sessions in this group
    feat_list = []
    for sr in session_results:
        for i in sr["valid_trials"]:
            sp = sr["speed_all"][i]
            ac = sr["accel_all"][i]
            mask = np.isfinite(sp) & np.isfinite(ac)
            feat_list.append(np.stack([sp[mask], ac[mask]], axis=1))

    if not feat_list:
        print(f"  [{scope_label}] No valid features — skipping HMM")
        return

    features_cat = np.vstack(feat_list)   # (T_total, 2)
    n_sessions   = len(session_results)
    print(f"\n[{scope_label}] Training HMM on {features_cat.shape[0]} frames "
          f"from {n_sessions} sessions")

    # Model selection or fixed n_states
    if n_states_arg is None:
        N_scan = np.arange(3, 15)
        print(f"  Running model selection over n_states = {N_scan} …")
        _, delta_ll = display_model_selection(
            features_cat, N_scan,
            out_path=fig_dir / f"{scope_label}_model_selection.png")
        knee = KneeLocator(N_scan[1:], delta_ll, curve="convex",
                           direction="decreasing", interp_method="polynomial")
        if knee.knee is None:
            raise RuntimeError(
                "Kneedle could not find a clear elbow in the delta-LL curve. "
                f"Inspect the model selection figure at {fig_dir / f'{scope_label}_model_selection.png'} "
                "and rerun with --n_states to set the number of states manually.")
        n_states = int(knee.knee)
        print(f"  Auto-selected n_states = {n_states} (kneedle elbow detection)")
    else:
        n_states = n_states_arg
        print(f"  Using fixed n_states = {n_states}")

    # Fit final HMM on pooled features
    print(f"  Fitting final HMM (n={n_states}) …")
    hmm_result = fit_final_model(features_cat, n_states)

    # Decode per session and save
    out_fname        = hmm_out_name(scope_label, piebald_filter)
    decoded_sessions = []
    for sr in session_results:
        print(f"  Decoding + saving: {sr['fig_prefix']}")

        states_all    = np.full((sr["n_trials"], TRIAL_LEN), -1, dtype=np.int32)
        posteriors_all = np.full((sr["n_trials"], TRIAL_LEN, n_states), np.nan, dtype=np.float32)
        for i in sr["valid_trials"]:
            sp   = sr["speed_all"][i]
            ac   = sr["accel_all"][i]
            mask = np.isfinite(sp) & np.isfinite(ac)
            feat = np.stack([sp[mask], ac[mask]], axis=1)
            seq  = decode_states(hmm_result["model"], hmm_result["index"], feat)
            ez   = decode_posteriors(hmm_result["model"], hmm_result["index"], feat)
            states_all[i, mask]         = seq
            posteriors_all[i, mask, :]  = ez

        decoded_sessions.append({
            "strain_folder": sr["strain_folder"],
            "mouse_id":      sr["mouse_id"],
            "states_all":    states_all,
        })

        # HMM output pkl — kinematics live in the Kalman cache pkl
        out_pkl  = sr["pkl_path"].parent / out_fname
        out_data = {
            "states":      states_all,
            "posteriors":  posteriors_all,   # (n_trials, TRIAL_LEN, n_states) float32, NaN = invalid
            "params_mean": hmm_result["params_mean"],
            "params_var":  hmm_result["params_var"],
            "TransM":      hmm_result["TransM"],
            "lls":         hmm_result["lls"],
            "n_states":    n_states,
            "hmm_scope":   scope_label,
            "kalman_source": kalman_cache_name(piebald_filter),
        }
        with open(out_pkl, "wb") as f:
            pickle.dump(out_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  HMM results saved: {out_pkl}")

        # Figures
        prefix = sr["fig_prefix"]
        plot_state_features(
            hmm_result["params_mean"], hmm_result["params_var"],
            hmm_result["TransM"], n_states,
            out_path=fig_dir / f"{prefix}_state_features.png")

        decoded_mask = (states_all >= 0).any(axis=1)
        if decoded_mask.sum() > 0:
            plot_event_heatmap(
                states_all[decoded_mask], n_states,
                out_path=fig_dir / f"{prefix}_event_heatmap.png")
            plot_event_positions(
                sr["smoothed_x"], sr["smoothed_y"], states_all, n_states,
                out_path=fig_dir / f"{prefix}_event_positions.png")

    # Group-level figures (one per HMM run, pools all sessions in this group)
    plot_group_ethogram(
        decoded_sessions, n_states,
        out_path=fig_dir / f"{scope_label}_group_ethogram.png")
    plot_sequence_distance_matrix(
        decoded_sessions, hmm_result, n_states,
        out_path=fig_dir / f"{scope_label}_sequence_distances.png")


# ---------------------------------------------------------------------------
# ============================================================
# Entry point
# ============================================================

def strain_to_folder(strain: str) -> str:
    if strain == "JF1":   return "JF1"
    if strain == "CAST":  return "CAST"
    return "_JF1xCAST"


def valid_sessions(mouse_dir: Path) -> list[Path]:
    return sorted(
        [d for d in mouse_dir.iterdir()
         if d.is_dir()
         and (d / "test" / "behaviour_and_sync" / PKL_IN_NAME).exists()],
        key=lambda d: d.name)


def main(n_states_arg, hmm_scope: str, piebald_filter: bool,
         refit_kalman: bool = False):
    df   = pd.read_excel(EXCEL_PATH, sheet_name="Behaviour")
    mice = df[df["RT_tagging"] == 1][["PyRat_ID", "Strain", "Coat"]].copy()
    mice["mouse_id"] = mice["PyRat_ID"].str.replace("BAA-", "", regex=False)

    print(f"HMM scope: {hmm_scope}")
    print(f"Piebald pre-filter: {'on' if piebald_filter else 'off'}")
    print(f"Refit Kalman: {'yes' if refit_kalman else 'no (use cache if available)'}\n")

    # ------------------------------------------------------------------
    # Pass 1: Kalman per session
    # ------------------------------------------------------------------
    all_session_results = []

    for _, row in mice.iterrows():
        mouse_id  = str(row["mouse_id"])
        strain    = str(row["Strain"])
        coat      = row.get("Coat", None)
        mouse_dir = DATA_ROOT / strain_to_folder(strain) / mouse_id

        print(f"\nMouse {mouse_id} ({strain})")
        if not mouse_dir.exists():
            print(f"  Folder not found: {mouse_dir}")
            continue

        sessions = valid_sessions(mouse_dir)
        if not sessions:
            print(f"  No sessions with {PKL_IN_NAME} found.")
            continue

        for session_dir in sessions:
            print(f"  Session: {session_dir.name}")
            pkl_path = session_dir / "test" / "behaviour_and_sync" / PKL_IN_NAME
            try:
                sr = run_kalman_session(pkl_path, mouse_id, strain, coat,
                                        piebald_filter, refit_kalman)
                if sr is not None:
                    all_session_results.append(sr)
            except Exception as e:
                print(f"  ERROR in Kalman pass: {e}")
                import traceback; traceback.print_exc()

    if not all_session_results:
        print("No sessions processed — exiting.")
        return

    # ------------------------------------------------------------------
    # Pass 2: HMM per scope group
    # ------------------------------------------------------------------
    if hmm_scope == "all":
        groups = {"global": all_session_results}
    else:  # "strain"
        groups = {}
        for sr in all_session_results:
            key = sr["strain_folder"]
            groups.setdefault(key, []).append(sr)

    for scope_label, session_results in groups.items():
        try:
            run_hmm_group(session_results, scope_label, n_states_arg, piebald_filter)
        except Exception as e:
            print(f"  ERROR in HMM pass [{scope_label}]: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kalman + HMM pipeline for loom-trial escape kinematics")
    parser.add_argument("--n_states", type=int, default=None,
                        help="Fix number of HMM states (skips model selection)")
    parser.add_argument("--hmm_scope", choices=["all", "strain"], default="all",
                        help="'all' = one global HMM across all mice; "
                             "'strain' = one HMM per strain (JF1/CAST/JF1xCAST)")
    parser.add_argument("--piebald_filter", action="store_true",
                        help="Apply k=3 median pre-filter to piebald mice "
                             "before Kalman filtering to suppress coat-patch artefacts")
    parser.add_argument("--refit_kalman", action="store_true",
                        help="Recompute Kalman smoothing even if a cache pkl exists")
    args = parser.parse_args()
    main(n_states_arg=args.n_states, hmm_scope=args.hmm_scope,
         piebald_filter=args.piebald_filter, refit_kalman=args.refit_kalman)
