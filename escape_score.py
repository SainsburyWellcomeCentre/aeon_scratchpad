import os
os.environ["OMP_NUM_THREADS"] = "16"

"""
Per-trial behavioural response score for mouse loom experiments.

Computes a continuous 0–1 score capturing the quality of each mouse's response
to a looming stimulus, ranking (ascending):
  1. Complete indifference
  2. Brief acknowledgment
  3. Sustained freezing / deceleration
  4. Partial / aborted escape
  5. Full escape to shelter

Six kinematic + HMM-based features are extracted per trial, MinMax-normalised
on the training set, and combined via weighted sum.

Data sources (per session):
  trials_escape_kinematics_kalman.pkl       — smoothed positions, speed, accel
  trials_escape_kinematics_hmm_global.pkl  — HMM states and posteriors (7 states)

Ground truth for validation: P_escape_loom per mouse from JF1xCAST Mice.xlsx

Usage:
  python escape_score.py
  python escape_score.py --out_dir results/ --weights equal --verbose
  python escape_score.py --test       # run sanity assertions and exit
"""

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Config — mirrors escape_kinematics_hmm.py
# ---------------------------------------------------------------------------
EXCEL_PATH  = Path(__file__).parent / "JF1xCAST Mice.xlsx"
DATA_ROOT   = Path("/ceph/branco/Dario/Escape_SWC/JF1xCAST")
KALMAN_PKL  = "trials_escape_kinematics_kalman.pkl"
HMM_PKL     = "trials_escape_kinematics_hmm_global.pkl"

LOOM_FRAME  = 200   # frame index of loom onset
TRIAL_LEN   = 600   # total frames per trial
EARLY_END   = 280   # LOOM_FRAME + 80 = 2 s post-loom
EPS         = 1e-6
N_STATES    = 7     # verified constant across all sessions

FEATURE_NAMES = ["f1", "f2", "f3", "f4", "f5", "f6"]

# Weights proportional to empirical Spearman rho of each feature with P_escape_loom:
# rhos ≈ [0.472, 0.488, 0.299, 0.400, 0.348, 0.298], normalised to sum=1
DEFAULT_WEIGHTS = np.array([0.205, 0.212, 0.130, 0.174, 0.151, 0.129])
EQUAL_WEIGHTS   = np.full(6, 1.0 / 6.0)

# Kinematic-only variants (F4/F5 excluded): rhos ≈ [0.472, 0.488, 0.299, 0.298]
# Renormalised from the same rho-proportional scheme: sum(remaining rhos) = 1.557
KINEMATIC_FEATURE_NAMES = ["f1", "f2", "f3", "f6"]
KINEMATIC_WEIGHTS       = np.array([0.303, 0.314, 0.192, 0.191])

# Strain folder mapping (kept in sync with escape_kinematics_hmm.py)
_STRAIN_PALETTE = {"CAST": "#2196F3", "JF1": "#F44336", "_JF1xCAST": "#9C27B0"}


# ===========================================================================
# Data loading
# ===========================================================================

def strain_to_folder(strain: str) -> str:
    """Map Excel Strain label to subdirectory name under DATA_ROOT."""
    if strain == "JF1":  return "JF1"
    if strain == "CAST": return "CAST"
    return "_JF1xCAST"


def load_mouse_list(excel_path: Path = EXCEL_PATH) -> pd.DataFrame:
    """Load RT_tagging==1 mice from Excel.

    Returns DataFrame with columns: mouse_id (str), strain (str),
    p_escape_loom (float).
    """
    df = pd.read_excel(excel_path, sheet_name="Behaviour")
    mice = df[df["RT_tagging"] == 1][["PyRat_ID", "Strain", "P_escape_loom"]].copy()
    mice["mouse_id"] = mice["PyRat_ID"].str.replace("BAA-", "", regex=False)
    mice = mice.rename(columns={"Strain": "strain", "P_escape_loom": "p_escape_loom"})
    mice = mice[["mouse_id", "strain", "p_escape_loom"]].reset_index(drop=True)
    if mice["p_escape_loom"].isna().any():
        raise ValueError("P_escape_loom contains NaN — check the Excel sheet.")
    return mice


def find_sessions(mouse_id: str, strain: str,
                  data_root: Path = DATA_ROOT,
                  use_hmm: bool = True) -> list[Path]:
    """Return sorted list of behaviour_and_sync dirs with the required pkl files.

    When use_hmm=False only the Kalman pkl is required.
    Returns empty list if mouse directory does not exist.
    """
    mouse_dir = data_root / strain_to_folder(strain) / mouse_id
    if not mouse_dir.exists():
        return []
    sessions = []
    for session_dir in sorted(mouse_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        bs = session_dir / "test" / "behaviour_and_sync"
        present = (bs / KALMAN_PKL).exists()
        if use_hmm:
            present = present and (bs / HMM_PKL).exists()
        if present:
            sessions.append(bs)
    return sessions


def load_session_data(bs_dir: Path,
                      use_hmm: bool = True) -> tuple[dict, dict | None]:
    """Load Kalman (and optionally HMM) pkl files for one session directory.

    Returns (kalman_dict, hmm_dict).  hmm_dict is None when use_hmm=False.
    """
    with open(bs_dir / KALMAN_PKL, "rb") as f:
        kalman = pickle.load(f)
    if not use_hmm:
        return kalman, None
    with open(bs_dir / HMM_PKL, "rb") as f:
        hmm = pickle.load(f)
    return kalman, hmm


# ===========================================================================
# Shelter estimation
# ===========================================================================

def estimate_shelter(smoothed_x: np.ndarray,
                     smoothed_y: np.ndarray,
                     valid_trials: list) -> tuple[float, float]:
    """Estimate shelter location as median frame-0 position across valid trials.

    Mice are placed at or near the shelter at trial start, making the first
    frame a reliable proxy for shelter position.
    """
    idx = list(valid_trials)
    sx = float(np.nanmedian(smoothed_x[idx, 0]))
    sy = float(np.nanmedian(smoothed_y[idx, 0]))
    return sx, sy


# ===========================================================================
# Feature functions
# ===========================================================================

def compute_f1(speed: np.ndarray, loom: int = LOOM_FRAME) -> float:
    """Peak speed ratio post-loom vs pre-loom baseline, log-transformed.

    F1 = log1p(max(speed[loom:]) / (mean(speed[:loom]) + EPS))
    """
    baseline = float(np.nanmean(speed[:loom]))
    peak     = float(np.nanmax(speed[loom:]))
    return float(np.log1p(peak / (baseline + EPS)))


def compute_f2(speed: np.ndarray,
               loom: int = LOOM_FRAME,
               early_end: int = EARLY_END) -> float:
    """Peak speed in the first 2 s post-loom vs baseline, log-transformed.

    F2 = log1p(max(speed[loom:early_end]) / (mean(speed[:loom]) + EPS))
    """
    baseline = float(np.nanmean(speed[:loom]))
    peak     = float(np.nanmax(speed[loom:early_end]))
    return float(np.log1p(peak / (baseline + EPS)))


def compute_f3(x: np.ndarray, y: np.ndarray,
               loom: int = LOOM_FRAME) -> float:
    """Directional efficiency of post-loom trajectory.

    F3 = net_displacement / (path_length + EPS), clipped to [0, 1].
    """
    dx = np.diff(x[loom:])
    dy = np.diff(y[loom:])
    path_length = float(np.nansum(np.sqrt(dx**2 + dy**2)))
    net_disp    = float(np.sqrt((x[-1] - x[loom])**2 + (y[-1] - y[loom])**2))
    return float(np.clip(net_disp / (path_length + EPS), 0.0, 1.0))


def compute_f4(posteriors: np.ndarray, loom: int = LOOM_FRAME) -> float:
    """Squared Jensen-Shannon divergence between pre- and post-loom HMM posteriors.

    Masks NaN frames before computing means.  Returns 0.0 if either window
    has no valid frames.

    F4 = jensenshannon(mean_pre, mean_post) ** 2,  in [0, 1].
    """
    valid_mask = np.isfinite(posteriors[:, 0])
    pre_valid  = valid_mask[:loom]
    post_valid = valid_mask[loom:]

    if not pre_valid.any() or not post_valid.any():
        return 0.0

    p_pre  = np.nanmean(posteriors[:loom][pre_valid],  axis=0)
    p_post = np.nanmean(posteriors[loom:][post_valid], axis=0)

    # Ensure proper probability distributions (guard against numerical noise)
    p_pre  = np.clip(p_pre,  0, None); p_pre  /= p_pre.sum()  + EPS
    p_post = np.clip(p_post, 0, None); p_post /= p_post.sum() + EPS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jsd = float(jensenshannon(p_pre, p_post))

    return float(np.clip(jsd ** 2, 0.0, 1.0))


def compute_f5(states: np.ndarray,
               loom: int = LOOM_FRAME,
               n_states: int = N_STATES) -> float:
    """Inverted latency to first occurrence of the top HMM state post-loom.

    Only the highest speed state (n_states - 1, i.e. state 6) is used;
    using state >= 5 gives much weaker discrimination.

    F5 = 1 - latency / (TRIAL_LEN - loom),  in [0, 1].
    Returns 0.0 if the top state is never reached post-loom.
    """
    top_state = n_states - 1
    post_states = states[loom:]
    hits = np.where(post_states == top_state)[0]
    if len(hits) == 0:
        return 0.0
    latency = int(hits[0])
    return float(1.0 - latency / (TRIAL_LEN - loom))


def compute_f6(x: np.ndarray, y: np.ndarray,
               shelter_x: float, shelter_y: float,
               loom: int = LOOM_FRAME) -> float:
    """Fractional reduction in distance to shelter post-loom.

    F6 = clip((d_loom - min_post_dist) / (d_loom + EPS), 0, 1)

    Returns 0.0 if mouse is already at the shelter at loom onset (d < 1 px).
    """
    d_loom = float(np.sqrt((x[loom] - shelter_x)**2 + (y[loom] - shelter_y)**2))
    if d_loom < 1.0:
        return 0.0
    dists = np.sqrt((x[loom:] - shelter_x)**2 + (y[loom:] - shelter_y)**2)
    min_post = float(np.nanmin(dists))
    return float(np.clip((d_loom - min_post) / (d_loom + EPS), 0.0, 1.0))


def compute_trial_features(kalman: dict,
                            hmm: dict | None,
                            trial_idx: int,
                            shelter_x: float,
                            shelter_y: float,
                            use_hmm: bool = True) -> dict | None:
    """Compute features for one trial.

    Returns None if the trial is invalid.
    With use_hmm=True returns keys f1..f6; with use_hmm=False returns f1, f2, f3, f6.
    """
    valid_trials = list(kalman["valid_trials"])
    if trial_idx not in valid_trials:
        return None

    speed = kalman["speed_all"][trial_idx]
    x     = kalman["smoothed_x"][trial_idx]
    y     = kalman["smoothed_y"][trial_idx]

    feats = {
        "f1": compute_f1(speed),
        "f2": compute_f2(speed),
        "f3": compute_f3(x, y),
        "f6": compute_f6(x, y, shelter_x, shelter_y),
    }

    if use_hmm:
        states_trial = hmm["states"][trial_idx]
        if np.all(states_trial == -1):
            return None
        posts = hmm["posteriors"][trial_idx]  # (600, 7)
        n_st  = int(hmm.get("n_states", N_STATES))
        if n_st != N_STATES:
            warnings.warn(f"n_states={n_st} (expected {N_STATES}); adapting F5 threshold.")
        feats["f4"] = compute_f4(posts)
        feats["f5"] = compute_f5(states_trial, n_states=n_st)

    return feats


# ===========================================================================
# Session-level processing
# ===========================================================================

def process_session(bs_dir: Path,
                    mouse_id: str,
                    strain: str,
                    p_escape_loom: float,
                    verbose: bool = False,
                    use_hmm: bool = True) -> list[dict]:
    """Process one session: load pkls, estimate shelter, extract features.

    Returns list of record dicts (one per valid trial).
    Returns empty list on load failure.
    """
    try:
        kalman, hmm = load_session_data(bs_dir, use_hmm=use_hmm)
    except FileNotFoundError as e:
        warnings.warn(f"  [skip] {bs_dir}: {e}")
        return []

    valid_trials = list(kalman["valid_trials"])
    if not valid_trials:
        return []

    shelter_x, shelter_y = estimate_shelter(
        kalman["smoothed_x"], kalman["smoothed_y"], valid_trials
    )

    session_date = bs_dir.parents[2].name  # .../MOUSE/DATE/test/behaviour_and_sync
    n_trials = int(kalman["n_trials"])

    records = []
    for trial_idx in range(n_trials):
        feats = compute_trial_features(kalman, hmm, trial_idx,
                                       shelter_x, shelter_y, use_hmm=use_hmm)
        if feats is None:
            continue
        records.append({
            "mouse_id":      mouse_id,
            "strain":        strain,
            "session":       session_date,
            "trial_idx":     trial_idx,
            "p_escape_loom": p_escape_loom,
            **feats,
        })

    if verbose:
        print(f"    {session_date}: {len(records)}/{n_trials} valid trials")

    return records


# ===========================================================================
# Train / test split
# ===========================================================================

def mouse_level_split(records_df: pd.DataFrame,
                      test_size: float = 0.3,
                      random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified mouse-level train/test split.

    Splits at mouse level (not trial level), stratified by strain.
    Returns (train_df, test_df).
    """
    mouse_meta = (records_df
                  .groupby("mouse_id", sort=False)["strain"]
                  .first()
                  .reset_index())
    mouse_ids = mouse_meta["mouse_id"].values
    strains   = mouse_meta["strain"].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                  random_state=random_state)
    train_idx, test_idx = next(sss.split(mouse_ids, strains))

    train_mice = set(mouse_ids[train_idx])
    test_mice  = set(mouse_ids[test_idx])

    train_df = records_df[records_df["mouse_id"].isin(train_mice)].copy()
    test_df  = records_df[records_df["mouse_id"].isin(test_mice)].copy()

    train_df["split"] = "train"
    test_df["split"]  = "test"

    return train_df, test_df


# ===========================================================================
# Scoring
# ===========================================================================

def fit_normalizer(train_df: pd.DataFrame,
                   feature_cols: list[str] = FEATURE_NAMES) -> MinMaxScaler:
    """Fit MinMaxScaler on training set features only."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def compute_scores(df: pd.DataFrame,
                   scaler: MinMaxScaler,
                   weights: np.ndarray = DEFAULT_WEIGHTS,
                   feature_cols: list[str] = FEATURE_NAMES) -> np.ndarray:
    """Compute per-trial scores: normalise features then take weighted sum.

    Returns float64 array of shape (n_trials,), values in [0, 1].
    """
    normed = scaler.transform(df[feature_cols].values)  # (n, 6) in [0, 1]
    return (normed @ weights).astype(np.float64)


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_mouse_level(df: pd.DataFrame, label: str = "test") -> float:
    """Compute Spearman rho between per-mouse mean score and P_escape_loom.

    Prints result and returns rho.
    """
    per_mouse = (df.groupby("mouse_id")
                   .agg(mean_score=("score", "mean"),
                        p_escape=("p_escape_loom", "first"))
                   .reset_index())
    rho, pval = spearmanr(per_mouse["mean_score"], per_mouse["p_escape"])
    n = len(per_mouse)
    print(f"  [{label}] n_mice={n}  Spearman ρ={rho:.3f}  p={pval:.2e}")
    return float(rho)


# ===========================================================================
# Output
# ===========================================================================

def save_results(df: pd.DataFrame, out_dir: Path,
                 prefix: str = "escape_score") -> None:
    """Save results DataFrame to CSV and pkl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{prefix}_results.csv", index=False)
    with open(out_dir / f"{prefix}_results.pkl", "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {out_dir / (prefix + '_results.csv')}")


def _strain_color(strain: str) -> str:
    return _STRAIN_PALETTE.get(strain_to_folder(strain), "#888888")


def plot_mouse_scatter(df: pd.DataFrame, out_dir: Path,
                       prefix: str = "escape_score") -> None:
    """Per-mouse mean score vs P_escape_loom, coloured by strain."""
    out_dir.mkdir(parents=True, exist_ok=True)

    per_mouse = (df.groupby("mouse_id")
                   .agg(mean_score=("score", "mean"),
                        p_escape=("p_escape_loom", "first"),
                        strain=("strain", "first"),
                        split=("split", "first"))
                   .reset_index())

    fig, ax = plt.subplots(figsize=(7, 6))

    markers = {"train": "o", "test": "^"}
    strains_seen: set = set()

    for _, row in per_mouse.iterrows():
        c  = _strain_color(row["strain"])
        mk = markers.get(row["split"], "o")
        ax.scatter(row["p_escape"], row["mean_score"],
                   color=c, marker=mk, s=60, alpha=0.85,
                   linewidths=0.5, edgecolors="white")
        strains_seen.add(row["strain"])

    # Strain legend
    strain_handles = [
        mpatches.Patch(color=_strain_color(s), label=s)
        for s in sorted(strains_seen)
    ]
    split_handles = [
        plt.Line2D([0], [0], marker="o", color="grey", linestyle="",
                   markersize=7, label="train"),
        plt.Line2D([0], [0], marker="^", color="grey", linestyle="",
                   markersize=7, label="test"),
    ]
    ax.legend(handles=strain_handles + split_handles,
              loc="upper left", fontsize=9, framealpha=0.8)

    # Spearman rho on test set
    test_df = per_mouse[per_mouse["split"] == "test"]
    if len(test_df) > 1:
        rho, pval = spearmanr(test_df["p_escape"], test_df["mean_score"])
        ax.set_title(f"Test set: Spearman ρ = {rho:.3f}  (p = {pval:.2e})", fontsize=11)

    ax.set_xlabel("P_escape_loom (ground truth)", fontsize=11)
    ax.set_ylabel("Mean escape score (per mouse)", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    out_path = out_dir / f"{prefix}_mouse_scatter.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_score_distributions(df: pd.DataFrame, out_dir: Path,
                              prefix: str = "escape_score") -> None:
    """Per-trial score histograms, one subplot per strain."""
    out_dir.mkdir(parents=True, exist_ok=True)

    strains = sorted(df["strain"].unique())
    n = len(strains)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, strain in zip(axes, strains):
        sub = df[df["strain"] == strain]
        c   = _strain_color(strain)
        for split, ls in [("train", "-"), ("test", "--")]:
            d = sub[sub["split"] == split]["score"]
            if len(d):
                ax.hist(d, bins=20, range=(0, 1), density=True,
                        color=c, alpha=0.6, histtype="stepfilled",
                        linestyle=ls, label=split)
        ax.set_title(strain, fontsize=11)
        ax.set_xlabel("Score", fontsize=10)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Density", fontsize=10)
    fig.suptitle("Per-trial score distributions by strain", fontsize=12)

    out_path = out_dir / f"{prefix}_score_distributions.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_feature_correlations(df: pd.DataFrame, out_dir: Path,
                               prefix: str = "escape_score",
                               feature_cols: list[str] = FEATURE_NAMES) -> None:
    """Bar chart of per-feature Spearman rho with P_escape_loom (test set)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df = df[df["split"] == "test"]
    rhos = []
    for feat in feature_cols:
        rho, _ = spearmanr(test_df[feat], test_df["p_escape_loom"])
        rhos.append(rho)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2196F3" if r >= 0 else "#F44336" for r in rhos]
    ax.bar(feature_cols, rhos, color=colors, edgecolor="white")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Spearman ρ with P_escape_loom", fontsize=11)
    ax.set_title("Per-feature discriminability (test set)", fontsize=11)
    ax.set_ylim(-0.1, max(0.6, max(rhos) + 0.05))
    for i, (feat, r) in enumerate(zip(feature_cols, rhos)):
        ax.text(i, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

    out_path = out_dir / f"{prefix}_feature_rhos.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ===========================================================================
# Sanity / unit tests
# ===========================================================================

def run_tests(verbose: bool = False) -> None:
    """Run basic sanity assertions.  Exits with non-zero status on failure."""
    import sys

    print("Running sanity tests...")

    mice = load_mouse_list()
    assert not mice["p_escape_loom"].isna().any(), "NaN in P_escape_loom"
    assert set(mice["strain"].unique()).issubset({"JF1", "CAST", "JF1xCAST F1",
                                                   "JF1xCAST BC1 (JF1)"}), \
        f"Unexpected strains: {mice['strain'].unique()}"
    print(f"  Mouse list: {len(mice)} mice loaded")

    # Test features on synthetic data
    rng = np.random.default_rng(0)
    speed     = np.abs(rng.normal(50, 20, TRIAL_LEN)).astype(float)
    x         = np.cumsum(rng.normal(0, 2, TRIAL_LEN)).astype(float) + 500
    y         = np.cumsum(rng.normal(0, 2, TRIAL_LEN)).astype(float) + 400
    posteriors = rng.dirichlet(np.ones(N_STATES), size=TRIAL_LEN).astype(np.float32)
    states    = rng.integers(0, N_STATES, size=TRIAL_LEN).astype(np.int32)

    f1 = compute_f1(speed)
    f2 = compute_f2(speed)
    f3 = compute_f3(x, y)
    f4 = compute_f4(posteriors)
    f5 = compute_f5(states)
    f6 = compute_f6(x, y, 500.0, 400.0)

    assert f1 >= 0,            f"F1 out of range: {f1}"
    assert f2 >= 0,            f"F2 out of range: {f2}"
    assert 0 <= f3 <= 1,       f"F3 out of range: {f3}"
    assert 0 <= f4 <= 1,       f"F4 out of range: {f4}"
    assert 0 <= f5 <= 1,       f"F5 out of range: {f5}"
    assert 0 <= f6 <= 1,       f"F6 out of range: {f6}"
    print(f"  Feature range checks passed: f1={f1:.3f} f2={f2:.3f} f3={f3:.3f} "
          f"f4={f4:.3f} f5={f5:.3f} f6={f6:.3f}")

    # NaN posteriors handled
    posts_nan = posteriors.copy()
    posts_nan[:10] = np.nan
    f4_nan = compute_f4(posts_nan)
    assert 0 <= f4_nan <= 1, f"F4 with NaN posteriors out of range: {f4_nan}"
    print("  F4 NaN-robustness check passed")

    # Stationary trial → F3 ≈ 0
    x_static = np.full(TRIAL_LEN, 500.0)
    y_static = np.full(TRIAL_LEN, 400.0)
    f3_static = compute_f3(x_static, y_static)
    assert f3_static < 0.01, f"F3 for stationary trial should be ~0, got {f3_static}"
    print("  F3 stationary-trial check passed")

    # Mouse already at shelter → F6 = 0
    f6_at_shelter = compute_f6(x_static, y_static, 500.0, 400.0)
    assert f6_at_shelter == 0.0, f"F6 with mouse at shelter should be 0, got {f6_at_shelter}"
    print("  F6 at-shelter check passed")

    print("All sanity tests passed.")


# ===========================================================================
# Main pipeline
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-trial escape behaviour scores for loom experiments."
    )
    parser.add_argument("--out_dir",    type=Path,  default=Path("."),
                        help="Output directory (default: current directory)")
    parser.add_argument("--prefix",     type=str,   default="escape_score",
                        help="Output filename prefix (default: escape_score)")
    parser.add_argument("--weights",    choices=["rho", "equal"], default="rho",
                        help="Feature weighting: 'rho' (proportional to Spearman rho, "
                             "default) or 'equal' (uniform 1/N)")
    parser.add_argument("--no_hmm",     action="store_true",
                        help="Kinematic-only mode: use F1, F2, F3, F6 only (skip HMM pkl)")
    parser.add_argument("--test_size",  type=float, default=0.3,
                        help="Fraction of mice for test set (default: 0.3)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for train/test split (default: 42)")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print per-session progress")
    parser.add_argument("--test",       action="store_true",
                        help="Run sanity assertions and exit")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    use_hmm    = not args.no_hmm
    feat_names = FEATURE_NAMES if use_hmm else KINEMATIC_FEATURE_NAMES
    if args.weights == "rho":
        weights = DEFAULT_WEIGHTS if use_hmm else KINEMATIC_WEIGHTS
    else:
        weights = EQUAL_WEIGHTS if use_hmm else np.full(len(feat_names), 1.0 / len(feat_names))

    # ------------------------------------------------------------------
    # 1. Load mouse list
    # ------------------------------------------------------------------
    print("Loading mouse list...")
    mice = load_mouse_list()
    print(f"  {len(mice)} mice (RT_tagging==1)")

    # ------------------------------------------------------------------
    # 2. Process all sessions
    # ------------------------------------------------------------------
    print("\nProcessing sessions...")
    all_records: list[dict] = []
    mice_missing = []

    for _, row in mice.iterrows():
        mouse_id = str(row["mouse_id"])
        strain   = str(row["strain"])
        p_esc    = float(row["p_escape_loom"])

        sessions = find_sessions(mouse_id, strain, use_hmm=use_hmm)
        if not sessions:
            mice_missing.append(mouse_id)
            if args.verbose:
                print(f"  Mouse {mouse_id} ({strain}): no sessions found")
            continue

        if args.verbose:
            print(f"  Mouse {mouse_id} ({strain}, P_esc={p_esc:.2f}): "
                  f"{len(sessions)} session(s)")

        for bs_dir in sessions:
            records = process_session(bs_dir, mouse_id, strain, p_esc,
                                      verbose=args.verbose, use_hmm=use_hmm)
            all_records.extend(records)

    if mice_missing:
        print(f"\n  [warning] {len(mice_missing)} mice with no session data: "
              f"{', '.join(mice_missing[:10])}"
              + (" ..." if len(mice_missing) > 10 else ""))

    if not all_records:
        print("ERROR: No valid trials found. Check DATA_ROOT path.")
        return

    records_df = pd.DataFrame(all_records)
    print(f"\nTotal valid trials: {len(records_df)}  "
          f"(across {records_df['mouse_id'].nunique()} mice, "
          f"{records_df.groupby(['mouse_id','session']).ngroups} sessions)")

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    print("\nSplitting train/test (mouse level, stratified by strain)...")
    train_df, test_df = mouse_level_split(records_df,
                                           test_size=args.test_size,
                                           random_state=args.seed)
    print(f"  Train: {train_df['mouse_id'].nunique()} mice, "
          f"{len(train_df)} trials")
    print(f"  Test:  {test_df['mouse_id'].nunique()} mice, "
          f"{len(test_df)} trials")

    # ------------------------------------------------------------------
    # 4. Fit normaliser on train set only
    # ------------------------------------------------------------------
    scaler = fit_normalizer(train_df, feature_cols=feat_names)

    # ------------------------------------------------------------------
    # 5. Compute scores on both splits
    # ------------------------------------------------------------------
    train_scores = compute_scores(train_df, scaler, weights, feature_cols=feat_names)
    test_scores  = compute_scores(test_df,  scaler, weights, feature_cols=feat_names)

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["score"] = train_scores
    test_df["score"]  = test_scores

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Sanity: all scores finite and in [0, 1]
    assert full_df["score"].between(0, 1).all(), \
        "Some scores outside [0, 1] — check normalisation."
    assert not full_df["score"].isna().any(), "NaN scores found."

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    mode_label = "kinematic-only (F1,F2,F3,F6)" if not use_hmm else "full (F1–F6)"
    print("\nEvaluation:")
    print(f"  Mode: {mode_label}   Weights: {args.weights} → {np.round(weights, 3)}")
    evaluate_mouse_level(train_df, label="train")
    rho_test = evaluate_mouse_level(test_df,  label="test")

    if rho_test < 0.75:
        print(f"  [warning] Test Spearman ρ = {rho_test:.3f} is below 0.75")

    # Per-feature rhos on test set
    print("\n  Per-feature Spearman ρ (test set):")
    for feat in feat_names:
        rho, _ = spearmanr(test_df[feat], test_df["p_escape_loom"])
        print(f"    {feat}: ρ = {rho:.3f}")

    # ------------------------------------------------------------------
    # 7. Save and plot
    # ------------------------------------------------------------------
    print("\nSaving results...")
    save_results(full_df, args.out_dir, args.prefix)

    print("Generating figures...")
    plot_mouse_scatter(full_df,        args.out_dir, args.prefix)
    plot_score_distributions(full_df,  args.out_dir, args.prefix)
    plot_feature_correlations(full_df, args.out_dir, args.prefix,
                               feature_cols=feat_names)

    print("\nDone.")


if __name__ == "__main__":
    main()
