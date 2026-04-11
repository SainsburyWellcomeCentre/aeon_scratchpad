"""
Crop Bonsai centroid data to loom trial windows and generate overlay videos.

For each RT_tagging==1 mouse, loads pre-computed pickle files from
test/behaviour_and_sync/ to align the full-session centroid trace with each
top_visual_loom trial window (5 s pre + 10 s post onset at 40 Hz = 600 frames).
Produces a cropped centroid array and an overlay video for each session.

Output per session (default — saves to /ceph alongside existing pipeline outputs):
  test/behaviour_and_sync/trials_top_visual_loom.bonsai_centroid.pkl         (dict: x_position_trace, y_position_trace, cam_frames)
  test/behaviour_and_sync/trials_top_visual_loom.bonsai_centroid_overlay.avi

Output per session (--test — saves locally for inspection before committing to /ceph):
  loom_tracking_results/{mouse_id}_{session}/trials_top_visual_loom.bonsai_centroid.pkl
  loom_tracking_results/{mouse_id}_{session}/{mouse_id}_{session}_overlay.avi

Usage:
  python loom_trials_bonsai_centroid_overlay.py           # all RT_tagging==1 mice
  python loom_trials_bonsai_centroid_overlay.py --test    # 3 mice per strain, saves locally
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# --- Config ---------------------------------------------------------------
EXCEL_PATH        = Path(__file__).parent / "JF1xCAST Mice.xlsx"
DATA_ROOT         = Path("/ceph/branco/Dario/Escape_SWC/JF1xCAST")
LOCAL_SAVE_DIR    = Path(__file__).parent / "loom_tracking_results"
PKL_NAME          = "trials_top_visual_loom.bonsai_centroid.pkl"
OVERLAY_NAME      = "trials_top_visual_loom.bonsai_centroid_overlay.avi"
PRE_FRAMES        = 200   # 5 s * 40 Hz — matches cutVideo_duration_params['top_visual_loom'][0] in extract_behaviour_and_sync_yl_AF.py
POST_FRAMES       = 400   # 10 s * 40 Hz — matches cutVideo_duration_params['top_visual_loom'][1]
TRIAL_LEN         = PRE_FRAMES + POST_FRAMES  # 600 — must equal trials_top_visual_loom.avi total_frames / n_trials
N_TEST_PER_STRAIN = 3
DOT_COLOUR        = (0, 255, 0)   # BGR green
DOT_RADIUS        = 8
DOT_THICK         = -1            # filled
# --------------------------------------------------------------------------


def strain_to_folder(strain: str) -> str:
    if strain == "JF1":
        return "JF1"
    if strain == "CAST":
        return "CAST"
    return "_JF1xCAST"


def valid_sessions(mouse_dir: Path) -> list[Path]:
    """Return all YYMMDD session folders that have the required pkls and loom avi, oldest first."""
    return sorted(
        [
            d for d in mouse_dir.iterdir()
            if d.is_dir()
            and (d / "test" / "behaviour_and_sync" / "events_dev3time.pkl").exists()
            and (d / "test" / "behaviour_and_sync" / "events_camera.pkl").exists()
            and (d / "test" / "behaviour_and_sync" / "trials_top_visual_loom.avi").exists()
        ],
        key=lambda d: d.name,
    )


def extract_trials(session_dir: Path):
    """Load pkls and return (trials array, cam_frame list) or (None, None)."""
    sync_dir = session_dir / "test" / "behaviour_and_sync"

    with open(sync_dir / "events_dev3time.pkl", "rb") as f:
        ev_dev3 = pickle.load(f)
    with open(sync_dir / "events_camera.pkl", "rb") as f:
        ev_cam = pickle.load(f)

    try:
        cam_trig    = ev_dev3["dev3"]["cam_trig"][0]
        loom_onsets = ev_dev3["dev3"]["visual_loom"][0]
    except KeyError as e:
        print(f"  Missing key in events_dev3time: {e}")
        return None, None

    try:
        x = np.array(ev_cam["x_position_trace"])
        y = np.array(ev_cam["y_position_trace"])
    except KeyError as e:
        print(f"  Missing key in events_camera: {e}")
        return None, None

    # Replicate find_nearest_camera_frame() from the original pipeline
    # (cam_trig and loom_onsets are both in dev3 sample units)
    loom_cam_frames = [int(np.argmin(np.abs(cam_trig - t))) for t in loom_onsets]

    trials = np.full((len(loom_onsets), TRIAL_LEN, 2), np.nan, dtype=np.float32)
    for i, f in enumerate(loom_cam_frames):
        start, end = f - PRE_FRAMES, f + POST_FRAMES
        if start >= 0 and end <= len(x):
            trials[i, :, 0] = x[start:end]
            trials[i, :, 1] = y[start:end]

    return trials, loom_cam_frames


def make_overlay(trials: np.ndarray, avi_path: Path, out_path: Path):
    cap    = cv2.VideoCapture(str(avi_path))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_idx < total:
                print(f"  Read failed at frame {frame_idx}/{total}, retrying...")
                cap.release()
                cap = cv2.VideoCapture(str(avi_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    print(f"  Retry failed, stopping at frame {frame_idx}")
                    break
            else:
                break

        trial_i  = frame_idx // TRIAL_LEN
        offset_j = frame_idx % TRIAL_LEN
        if trial_i < len(trials):
            x, y = trials[trial_i, offset_j, :]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x), int(y)), DOT_RADIUS, DOT_COLOUR, DOT_THICK)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  Overlay saved: {out_path}  ({frame_idx} frames)")


def process_session(mouse_id: str, strain: str, session_dir: Path, test_run: bool):
    print(f"  Session: {session_dir.name}")

    trials, cam_frames = extract_trials(session_dir)
    if trials is None:
        return

    sync_dir = session_dir / "test" / "behaviour_and_sync"
    avi_path = sync_dir / "trials_top_visual_loom.avi"

    if test_run:
        out_dir = LOCAL_SAVE_DIR / f"{mouse_id}_{session_dir.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl_path     = out_dir / PKL_NAME
        overlay_path = out_dir / f"{mouse_id}_{session_dir.name}_overlay.avi"
    else:
        pkl_path     = sync_dir / PKL_NAME
        overlay_path = sync_dir / OVERLAY_NAME

    centroid_data = {
        "x_position_trace": trials[:, :, 0],
        "y_position_trace": trials[:, :, 1],
        "cam_frames":       np.array(cam_frames),
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(centroid_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Centroids saved: {pkl_path}  ({len(trials)} trials)")

    make_overlay(trials, avi_path, overlay_path)


def main(test_run: bool):
    df   = pd.read_excel(EXCEL_PATH, sheet_name="Behaviour")
    mice = df[df["RT_tagging"] == 1][["PyRat_ID", "Strain"]].copy()
    mice["mouse_id"] = mice["PyRat_ID"].str.replace("BAA-", "", regex=False)

    if test_run:
        mice["strain_group"] = mice["Strain"].apply(strain_to_folder)
        mice = (
            mice.groupby("strain_group", group_keys=False)
            .apply(lambda g: g.head(N_TEST_PER_STRAIN), include_groups=False)
            .reset_index(drop=True)
        )
        print(f"Test run: {len(mice)} mice ({N_TEST_PER_STRAIN} per strain)\n")
        LOCAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in mice.iterrows():
        mouse_id  = str(row["mouse_id"])
        strain    = str(row["Strain"])
        mouse_dir = DATA_ROOT / strain_to_folder(strain) / mouse_id

        print(f"Mouse {mouse_id} ({strain})")

        if not mouse_dir.exists():
            print(f"  Folder not found: {mouse_dir}")
            continue

        sessions = valid_sessions(mouse_dir)
        if not sessions:
            print(f"  No valid session found.")
            continue

        for session_dir in sessions:
            process_session(mouse_id, strain, session_dir, test_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Test run: 3 mice per strain, saves locally")
    args = parser.parse_args()
    main(test_run=args.test)
