import os
os.environ["OMP_NUM_THREADS"] = "16"

"""
Generate validation reels from scored top-loom trials.

Supports two grouping modes:
  - strain: one reel per strain (default, preserves the existing workflow)
  - score_bin: one reel per fixed score-threshold bin

Usage:
  python score_validation_reels.py
  python score_validation_reels.py --out_dir score_validation_reels/
  python score_validation_reels.py --clips_per_strain 12
  python score_validation_reels.py --group_by score_bin --clips_per_bin 20 --n_bins 5
  python score_validation_reels.py --group_by score_bin --clips_per_bin 20 --pre_seconds 2 --post_seconds 5 --ext mp4
  python score_validation_reels.py --test
"""

import argparse
from functools import lru_cache
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd


DATA_ROOT = Path("/ceph/branco/Dario/Escape_SWC/JF1xCAST")
CSV_PATH = Path(__file__).parent / "escape_score_results" / "escape_score_results.csv"
TRIAL_VIDEO_NAME = "trials_top_visual_loom.avi"
DEFAULT_OUT_DIR = Path(__file__).parent / "score_validation_reels"
TRIAL_LEN = 600
LOOM_FRAME = 200
FPS_FALLBACK = 40.0
DEFAULT_PRE_SECONDS = 0.0
DEFAULT_POST_SECONDS = 3.0
DEFAULT_EXT = "avi"
DEFAULT_GROUP_BY = "strain"
DEFAULT_N_BINS = 5
DEFAULT_SCORE_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEPARATOR_FRAMES = 12
_WARNED_AMBIGUITIES: set[tuple[str, str]] = set()

# ── Annotation reel ──────────────────────────────────────────────────────────
ANNOTATION_REEL_TOTAL = 200
ANNOTATION_REEL_SEED = 42
ANNOTATION_SCORE_MIN = 0.05   # exclude likely video/tracking errors below this
ANNOTATION_SCORE_MAX = 0.80   # exclude outliers above this
ANNOTATION_STRAIN_TARGETS: dict[str, int] = {
    "CAST": 35,
    "JF1": 35,
    "JF1xCAST BC1 (JF1)": 55,
    "JF1xCAST F1": 75,
}
ANNOTATION_SEPARATOR_FRAMES = 80    # ~2 s at 40 fps — human-readable pause
ANNOTATION_XLSX_COLUMNS = [
    "clip_number", "strain", "mouse_id", "session", "trial_idx",
    "score", "annotated_score",
]


def strain_to_folder(strain: str) -> str:
    if strain == "JF1":
        return "JF1"
    if strain == "CAST":
        return "CAST"
    return "_JF1xCAST"


def slugify_label(label: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")


def edge_label(value: float) -> str:
    return f"{int(round(value * 10)):02d}"


def load_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"mouse_id", "strain", "session", "trial_idx", "score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    df = df.copy()
    df["mouse_id"] = df["mouse_id"].astype(str)
    df["strain"] = df["strain"].astype(str)
    df["session"] = df["session"].astype(str)
    df["trial_idx"] = df["trial_idx"].astype(int)
    df["score"] = df["score"].astype(float)
    return df


def sort_rows(df: pd.DataFrame, include_strain: bool = False) -> pd.DataFrame:
    cols = ["score"]
    if include_strain:
        cols.append("strain")
    cols += ["mouse_id", "session", "trial_idx"]
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def sample_examples(df: pd.DataFrame, n_clips: int, include_strain: bool = False) -> pd.DataFrame:
    ordered = sort_rows(df, include_strain=include_strain)
    if len(ordered) <= n_clips:
        return ordered

    idx = np.linspace(0, len(ordered) - 1, n_clips, dtype=int)
    sampled = ordered.iloc[idx].copy()
    return sort_rows(sampled, include_strain=include_strain)


def resolve_video_path(row: pd.Series) -> Path:
    return (
        DATA_ROOT
        / strain_to_folder(row["strain"])
        / row["mouse_id"]
        / row["session"]
        / "test"
        / "behaviour_and_sync"
        / TRIAL_VIDEO_NAME
    )


@lru_cache(maxsize=None)
def valid_trial_videos(mouse_id: str, strain: str) -> tuple[Path, ...]:
    mouse_dir = DATA_ROOT / strain_to_folder(strain) / mouse_id
    if not mouse_dir.exists():
        return tuple()

    videos = []
    for session_dir in sorted(d for d in mouse_dir.iterdir() if d.is_dir()):
        video_path = session_dir / "test" / "behaviour_and_sync" / TRIAL_VIDEO_NAME
        if video_path.exists():
            videos.append(video_path)
    return tuple(videos)


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")
    return cap


def get_video_props(path: Path) -> tuple[float, int, int]:
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions for {path}")
    return fps, width, height


def resolve_existing_video(row: pd.Series, pre_frames: int, post_frames: int) -> Path:
    direct = resolve_video_path(row)
    if direct.exists():
        return direct

    candidates = valid_trial_videos(str(row["mouse_id"]), str(row["strain"]))
    if not candidates:
        raise FileNotFoundError(
            f"No {TRIAL_VIDEO_NAME} files found for mouse {row['mouse_id']} ({row['strain']})"
        )

    requested_start = int(row["trial_idx"]) * TRIAL_LEN + LOOM_FRAME - pre_frames
    requested_end = int(row["trial_idx"]) * TRIAL_LEN + LOOM_FRAME + post_frames
    valid = []
    for path in candidates:
        cap = open_video(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if requested_start >= 0 and requested_end <= total:
            valid.append(path)

    if not valid:
        raise FileNotFoundError(
            f"No candidate session for mouse {row['mouse_id']} can satisfy trial_idx={row['trial_idx']}"
        )
    if len(valid) > 1:
        key = (str(row["mouse_id"]), str(row["strain"]))
        sessions = ", ".join(str(p.parent.parent.parent.name) for p in valid)
        if key not in _WARNED_AMBIGUITIES:
            _WARNED_AMBIGUITIES.add(key)
            raise ValueError(
                f"Ambiguous session for mouse {row['mouse_id']} trial_idx={row['trial_idx']}: {sessions}"
            )
        raise ValueError("Ambiguous session due to multiple matching session folders")
    return valid[0]


def draw_label(frame: np.ndarray, row: pd.Series) -> np.ndarray:
    text = (
        f"mouse_id={row['mouse_id']} | strain={row['strain']} | "
        f"session={row['session']} | trial={int(row['trial_idx'])} | "
        f"score={row['score']:.3f}"
    )
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    x, y = 18, 32

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(out, (10, 10), (10 + text_w + 16, 10 + text_h + baseline + 16), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def normalize_frame_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def read_trial_clip(video_path: Path, trial_idx: int, pre_frames: int, post_frames: int) -> list[np.ndarray]:
    start_frame = trial_idx * TRIAL_LEN + LOOM_FRAME - pre_frames
    end_frame = trial_idx * TRIAL_LEN + LOOM_FRAME + post_frames

    cap = open_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 0 or end_frame > total:
        cap.release()
        raise ValueError(
            f"Requested frames [{start_frame}, {end_frame}) exceed video length {total} for {video_path}"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(pre_frames + post_frames):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Failed reading frame from {video_path} at trial {trial_idx}")
        frames.append(frame)
    cap.release()
    return frames


def write_separator(writer: cv2.VideoWriter, width: int, height: int, label: str, score: float) -> None:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    text = f"{label} | next score {score:.3f}"
    cv2.putText(frame, text, (18, max(36, height // 2)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    for _ in range(SEPARATOR_FRAMES):
        writer.write(frame)


def collect_resolvable_rows(
    df: pd.DataFrame,
    pre_frames: int,
    post_frames: int,
) -> list[tuple[pd.Series, Path]]:
    resolved_rows: list[tuple[pd.Series, Path]] = []
    for _, row in df.iterrows():
        try:
            resolved_rows.append((row, resolve_existing_video(row, pre_frames, post_frames)))
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(f"Skipping {row['mouse_id']} trial {row['trial_idx']}: {exc}")
    return resolved_rows


def output_filename(label: str, pre_seconds: float, post_seconds: float, ext: str) -> str:
    pre_label = f"{pre_seconds:g}".replace(".", "p")
    post_label = f"{post_seconds:g}".replace(".", "p")
    return f"escape_score_validation_{slugify_label(label)}_{pre_label}s_pre_{post_label}s_post.{ext}"


def video_fourcc(ext: str) -> int:
    if ext == "mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter_fourcc(*"XVID")


def select_resolved_subset(
    resolved_rows: list[tuple[pd.Series, Path]],
    n_clips: int,
    include_strain: bool = False,
) -> list[tuple[pd.Series, Path]]:
    if not resolved_rows:
        return []

    resolved_df = pd.DataFrame([row.to_dict() for row, _ in resolved_rows])
    sampled_df = sample_examples(resolved_df, n_clips, include_strain=include_strain)
    sampled_key = {
        (row["mouse_id"], row["session"], int(row["trial_idx"]), float(row["score"]))
        for _, row in sampled_df.iterrows()
    }
    sampled_resolved = [
        (row, path)
        for row, path in resolved_rows
        if (row["mouse_id"], row["session"], int(row["trial_idx"]), float(row["score"])) in sampled_key
    ]
    sampled_resolved.sort(key=lambda item: (
        float(item[0]["score"]),
        str(item[0]["strain"]) if include_strain else "",
        str(item[0]["mouse_id"]),
        str(item[0]["session"]),
        int(item[0]["trial_idx"]),
    ))
    return sampled_resolved


def write_reel(
    label: str,
    resolved_rows: list[tuple[pd.Series, Path]],
    out_dir: Path,
    pre_frames: int,
    post_frames: int,
    pre_seconds: float,
    post_seconds: float,
    ext: str,
) -> tuple[Path, int]:
    if not resolved_rows:
        raise RuntimeError(f"No resolvable clips found for group {label}")

    first_video = resolved_rows[0][1]
    fps, width, height = get_video_props(first_video)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename(label, pre_seconds, post_seconds, ext)
    writer = cv2.VideoWriter(str(out_path), video_fourcc(ext), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for {out_path}")

    try:
        for i, (row, video_path) in enumerate(resolved_rows):
            clip_frames = read_trial_clip(video_path, int(row["trial_idx"]), pre_frames, post_frames)
            for frame in clip_frames:
                frame = normalize_frame_size(frame, width, height)
                writer.write(draw_label(frame, row))
            if i < len(resolved_rows) - 1:
                next_score = float(resolved_rows[i + 1][0]["score"])
                write_separator(writer, width, height, label, next_score)
    finally:
        writer.release()

    return out_path, len(resolved_rows)


def build_strain_reels(
    df: pd.DataFrame,
    out_dir: Path,
    clips_per_strain: int,
    pre_frames: int,
    post_frames: int,
    pre_seconds: float,
    post_seconds: float,
    ext: str,
) -> list[tuple[str, Path, int]]:
    outputs = []
    for strain in sorted(df["strain"].unique()):
        strain_df = df[df["strain"] == strain].copy()
        resolvable = collect_resolvable_rows(strain_df, pre_frames, post_frames)
        if not resolvable:
            warnings.warn(f"Skipping strain {strain}: no resolvable source videos")
            continue

        sampled_resolved = select_resolved_subset(resolvable, clips_per_strain, include_strain=False)
        out_path, n_written = write_reel(
            strain,
            sampled_resolved,
            out_dir,
            pre_frames,
            post_frames,
            pre_seconds,
            post_seconds,
            ext,
        )
        outputs.append((strain, out_path, n_written))
    return outputs


def build_score_bin_groups(df: pd.DataFrame, n_bins: int) -> list[tuple[str, pd.DataFrame]]:
    ordered = sort_rows(df, include_strain=True)
    if n_bins != 5:
        raise ValueError("score_bin mode currently supports only 5 fixed score bins")

    groups: list[tuple[str, pd.DataFrame]] = []
    last_bin_idx = len(DEFAULT_SCORE_BIN_EDGES) - 2
    for bin_idx, (low, high) in enumerate(zip(DEFAULT_SCORE_BIN_EDGES[:-1], DEFAULT_SCORE_BIN_EDGES[1:])):
        if bin_idx == len(DEFAULT_SCORE_BIN_EDGES) - 2:
            mask = (ordered["score"] >= low) & (ordered["score"] <= high)
        else:
            mask = (ordered["score"] >= low) & (ordered["score"] < high)
        bin_df = ordered[mask].copy().reset_index(drop=True)
        if bin_df.empty:
            continue
        low_label = edge_label(low)
        high_label = edge_label(high)
        label = f"bin_{low_label}_to_{high_label}"
        groups.append((label, bin_df))
    return groups


def build_score_bin_reels(
    df: pd.DataFrame,
    out_dir: Path,
    clips_per_bin: int,
    n_bins: int,
    pre_frames: int,
    post_frames: int,
    pre_seconds: float,
    post_seconds: float,
    ext: str,
    max_groups: int | None = None,
) -> list[tuple[str, Path, int]]:
    outputs = []
    groups = build_score_bin_groups(df, n_bins)
    if max_groups is not None:
        groups = groups[:max_groups]

    for label, bin_df in groups:
        resolvable = collect_resolvable_rows(bin_df, pre_frames, post_frames)
        if not resolvable:
            warnings.warn(f"Skipping score bin {label}: no resolvable source videos")
            continue

        sampled_resolved = select_resolved_subset(resolvable, clips_per_bin, include_strain=True)
        out_path, n_written = write_reel(
            label,
            sampled_resolved,
            out_dir,
            pre_frames,
            post_frames,
            pre_seconds,
            post_seconds,
            ext,
        )
        outputs.append((label, out_path, n_written))
    return outputs


def sample_annotation_reel(
    df: pd.DataFrame,
    targets: dict[str, int] = ANNOTATION_STRAIN_TARGETS,
    seed: int = ANNOTATION_REEL_SEED,
) -> pd.DataFrame:
    """Sample trials for the annotation reel.

    Filters extreme outlier scores, then draws a linspace sample from each
    strain so that the reel covers each strain's full score range.  Rows are
    shuffled with a fixed seed so playback order is random (no score gradient
    visible to the annotator).  A 1-indexed ``clip_number`` column is added
    after shuffling — it doubles as the Excel row number shown in the overlay.
    """
    filtered = df[
        (df["score"] >= ANNOTATION_SCORE_MIN) & (df["score"] <= ANNOTATION_SCORE_MAX)
    ].copy()

    parts: list[pd.DataFrame] = []
    for strain, n_target in targets.items():
        s_df = filtered[filtered["strain"] == strain].sort_values("score").reset_index(drop=True)
        if s_df.empty:
            warnings.warn(f"annotation_reel: no trials found for strain '{strain}' after filtering")
            continue
        n = min(n_target, len(s_df))
        parts.append(sample_examples(s_df, n))

    combined = pd.concat(parts, ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled["clip_number"] = range(1, len(shuffled) + 1)
    return shuffled


def _join_resolved_paths(
    sampled_df: pd.DataFrame,
    resolvable: list[tuple[pd.Series, Path]],
) -> list[tuple[pd.Series, Path]]:
    """Join sampled rows (with clip_number) to their resolved video paths.

    Preserves clip_number and returns pairs sorted by clip_number so the
    video writer produces clips in shuffle/playback order.  Bypasses
    select_resolved_subset to avoid re-sampling.
    """
    key_to_path: dict[tuple, Path] = {
        (str(row["mouse_id"]), str(row["session"]), int(row["trial_idx"])): path
        for row, path in resolvable
    }

    joined: list[tuple[pd.Series, Path]] = []
    for _, row in sampled_df.iterrows():
        key = (str(row["mouse_id"]), str(row["session"]), int(row["trial_idx"]))
        if key in key_to_path:
            joined.append((row, key_to_path[key]))
        else:
            warnings.warn(
                f"annotation_reel: no resolved video for "
                f"mouse {row['mouse_id']} session {row['session']} trial {row['trial_idx']}"
            )

    joined.sort(key=lambda item: int(item[0]["clip_number"]))
    return joined


def draw_clip_number(frame: np.ndarray, clip_number: int, total: int) -> np.ndarray:
    """Draw the clip number prominently in the bottom-right corner."""
    text = f"{clip_number}/{total}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 3.0
    thickness = 4
    margin = 15

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    x = w - tw - margin - 8
    y = h - margin

    out = frame.copy()
    cv2.rectangle(
        out,
        (x - 8, y - th - baseline - 8),
        (w - margin + 8, h - margin + baseline),
        (0, 0, 0),
        -1,
    )
    cv2.putText(out, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def write_annotation_xlsx(
    video_path: Path,
    resolved_rows: list[tuple[pd.Series, Path]],
) -> Path:
    """Write companion annotation Excel sheet alongside the video."""
    rows = [row.to_dict() for row, _ in resolved_rows]
    annotation_df = pd.DataFrame(rows)[
        [c for c in ANNOTATION_XLSX_COLUMNS if c != "annotated_score"]
    ].copy()
    annotation_df["annotated_score"] = annotation_df["score"]
    annotation_df = annotation_df[ANNOTATION_XLSX_COLUMNS]

    xlsx_path = video_path.with_suffix(".xlsx")
    annotation_df.to_excel(str(xlsx_path), index=False, engine="openpyxl")
    return xlsx_path


def write_reel_annotated(
    label: str,
    resolved_rows: list[tuple[pd.Series, Path]],
    out_dir: Path,
    pre_frames: int,
    post_frames: int,
    pre_seconds: float,
    post_seconds: float,
    ext: str,
) -> tuple[Path, int]:
    """Write the annotation reel video.

    Like write_reel but with:
      - large clip-number overlay in the bottom-right of every frame
      - longer separator frames (ANNOTATION_SEPARATOR_FRAMES) for readability
    """
    if not resolved_rows:
        raise RuntimeError(f"No resolvable clips found for {label}")

    total = len(resolved_rows)
    first_video = resolved_rows[0][1]
    fps, width, height = get_video_props(first_video)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename(label, pre_seconds, post_seconds, ext)
    writer = cv2.VideoWriter(str(out_path), video_fourcc(ext), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for {out_path}")

    try:
        for i, (row, video_path) in enumerate(resolved_rows):
            clip_number = int(row["clip_number"])
            clip_frames = read_trial_clip(video_path, int(row["trial_idx"]), pre_frames, post_frames)
            for frame in clip_frames:
                frame = normalize_frame_size(frame, width, height)
                frame = draw_label(frame, row)
                frame = draw_clip_number(frame, clip_number, total)
                writer.write(frame)

            if i < len(resolved_rows) - 1:
                next_row = resolved_rows[i + 1][0]
                next_clip = int(next_row["clip_number"])
                next_score = float(next_row["score"])
                sep_text = f"Next: {next_clip}/{total} | score {next_score:.3f}"
                sep_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(
                    sep_frame, sep_text,
                    (18, max(36, height // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
                )
                for _ in range(ANNOTATION_SEPARATOR_FRAMES):
                    writer.write(sep_frame)
    finally:
        writer.release()

    return out_path, total


def build_annotation_reel(
    df: pd.DataFrame,
    out_dir: Path,
    pre_frames: int,
    post_frames: int,
    pre_seconds: float,
    post_seconds: float,
    ext: str,
    targets: dict[str, int] | None = None,
) -> tuple[Path, Path, int]:
    """Build the single annotation reel: one video + one Excel sheet."""
    sampled_df = sample_annotation_reel(df, targets=targets or ANNOTATION_STRAIN_TARGETS)
    if sampled_df.empty:
        raise RuntimeError("annotation_reel: no trials selected after filtering and sampling")

    resolvable = collect_resolvable_rows(sampled_df, pre_frames, post_frames)
    if not resolvable:
        raise RuntimeError("annotation_reel: no source videos could be resolved")

    joined = _join_resolved_paths(sampled_df, resolvable)
    if len(joined) < len(sampled_df):
        warnings.warn(
            f"annotation_reel: {len(sampled_df) - len(joined)} clips dropped "
            f"(missing video files); writing {len(joined)} clips"
        )

    label = "annotation_reel"
    video_path, n_written = write_reel_annotated(
        label, joined, out_dir, pre_frames, post_frames, pre_seconds, post_seconds, ext
    )
    xlsx_path = write_annotation_xlsx(video_path, joined)
    return video_path, xlsx_path, n_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH,
                        help="Path to escape_score_results.csv")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Output directory for validation reels")
    parser.add_argument("--group_by", choices=["strain", "score_bin"], default=DEFAULT_GROUP_BY,
                        help="How to group clips into reels")
    parser.add_argument("--clips_per_strain", type=int, default=12,
                        help="Representative clips to sample per strain")
    parser.add_argument("--clips_per_bin", type=int, default=20,
                        help="Representative clips to sample per score bin")
    parser.add_argument("--n_bins", type=int, default=DEFAULT_N_BINS,
                        help="Number of global score bins for score_bin mode")
    parser.add_argument("--pre_seconds", type=float, default=DEFAULT_PRE_SECONDS,
                        help="Seconds to keep before loom onset")
    parser.add_argument("--post_seconds", type=float, default=DEFAULT_POST_SECONDS,
                        help="Seconds to keep after loom onset")
    parser.add_argument("--ext", choices=["avi", "mp4"], default=DEFAULT_EXT,
                        help="Output video container/codec preset")
    parser.add_argument("--test", action="store_true",
                        help="Write reels for a small subset of groups only")
    parser.add_argument("--annotation_reel", action="store_true",
                        help="Generate a single 200-trial annotation reel with companion Excel sheet")
    args = parser.parse_args()

    if args.clips_per_strain <= 0:
        raise ValueError("--clips_per_strain must be > 0")
    if args.clips_per_bin <= 0:
        raise ValueError("--clips_per_bin must be > 0")
    if args.n_bins <= 0:
        raise ValueError("--n_bins must be > 0")
    if args.pre_seconds < 0:
        raise ValueError("--pre_seconds must be >= 0")
    if args.post_seconds <= 0:
        raise ValueError("--post_seconds must be > 0")

    pre_frames = int(round(args.pre_seconds * FPS_FALLBACK))
    post_frames = int(round(args.post_seconds * FPS_FALLBACK))
    if pre_frames < 0 or post_frames <= 0:
        raise ValueError("Computed frame counts are invalid")
    if LOOM_FRAME - pre_frames < 0 or LOOM_FRAME + post_frames > TRIAL_LEN:
        raise ValueError("Requested clip exceeds the 600-frame trial window")

    df = load_scores(args.csv)

    if args.annotation_reel:
        # Scale down targets for a quick test run (~20 clips)
        targets = ANNOTATION_STRAIN_TARGETS
        if args.test:
            scale = 0.10
            targets = {s: max(1, round(n * scale)) for s, n in ANNOTATION_STRAIN_TARGETS.items()}

        video_path, xlsx_path, n_written = build_annotation_reel(
            df,
            args.out_dir,
            pre_frames,
            post_frames,
            args.pre_seconds,
            args.post_seconds,
            args.ext,
            targets=targets,
        )
        print(f"Generated annotation reel: {n_written} clips")
        print(f"  Video: {video_path}")
        print(f"  Excel: {xlsx_path}")
        return

    score_bin_max_groups = None
    if args.test:
        if args.group_by == "strain":
            keep = sorted(df["strain"].unique())[:2]
            df = df[df["strain"].isin(keep)].copy()
        else:
            if not build_score_bin_groups(df, args.n_bins):
                raise RuntimeError("No score bins available for test mode")
            score_bin_max_groups = 2

    if args.group_by == "strain":
        outputs = build_strain_reels(
            df,
            args.out_dir,
            args.clips_per_strain,
            pre_frames,
            post_frames,
            args.pre_seconds,
            args.post_seconds,
            args.ext,
        )
    else:
        outputs = build_score_bin_reels(
            df,
            args.out_dir,
            args.clips_per_bin,
            args.n_bins,
            pre_frames,
            post_frames,
            args.pre_seconds,
            args.post_seconds,
            args.ext,
            max_groups=score_bin_max_groups,
        )

    print("Generated validation reels:")
    for label, path, n_clips in outputs:
        print(f"  {label}: {n_clips} clips -> {path}")


if __name__ == "__main__":
    main()
