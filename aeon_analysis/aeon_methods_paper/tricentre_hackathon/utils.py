import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from aeon.dj_pipeline.analysis.block_analysis import get_foraging_bouts
from aeon.dj_pipeline.analysis.block_analysis import *
from tqdm import tqdm


def ensure_ts_arr_datetime(array: Any) -> np.ndarray:
    if len(array) == 0:
        return np.array([], dtype='datetime64[ns]')
    return np.array(array, dtype='datetime64[ns]')


def load_subject_patch_data(
    key: dict[str, str],
    period_start: str,
    period_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patch_info = (
        BlockAnalysis.Patch()
        & key
        & f"block_start >= '{period_start}'"
        & f"block_start <= '{period_end}'"
    ).fetch(
        "block_start", "patch_name", "patch_rate", "patch_offset", "wheel_timestamps", as_dict=True
    )

    block_subject_patch_data = (
        BlockSubjectAnalysis.Patch()
        & key
        & f"block_start >= '{period_start}'"
        & f"block_start <= '{period_end}'"
    ).fetch(format="frame")

    block_subject_patch_pref = (
        BlockSubjectAnalysis.Preference()
        & key
        & f"block_start >= '{period_start}'"
        & f"block_start <= '{period_end}'"
    ).fetch(format="frame")

    if patch_info:
        patch_info = pd.DataFrame(patch_info)

    if not block_subject_patch_data.empty:
        block_subject_patch_data.reset_index(inplace=True)

    if not block_subject_patch_pref.empty:
        block_subject_patch_pref.reset_index(inplace=True)

    return patch_info, block_subject_patch_data, block_subject_patch_pref


def load_all_patch_data(experiments: list[dict[str, str]]) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, pd.DataFrame]]
]:
    patch_info_dict = {}
    subject_patch_data_dict = {}
    subject_patch_pref_dict = {}

    for exp in experiments:
        key = {"experiment_name": exp["name"]}

        periods = {}
        if "presocial_start" in exp and "presocial_end" in exp:
            periods["presocial"] = (exp["presocial_start"], exp["presocial_end"])
        if "social_start" in exp and "social_end" in exp:
            periods["social"] = (exp["social_start"], exp["social_end"])
        if "postsocial_start" in exp and "postsocial_end" in exp:
            periods["postsocial"] = (exp["postsocial_start"], exp["postsocial_end"])

        patch_info_dict[exp["name"]] = {}
        subject_patch_data_dict[exp["name"]] = {}
        subject_patch_pref_dict[exp["name"]] = {}

        for period_name, (period_start, period_end) in tqdm(
            periods.items(),
            desc=f"Loading periods for {exp['name']}",
            leave=False
        ):
            period_start = datetime.strptime(period_start, "%Y-%m-%d %H:%M:%S")
            period_end = datetime.strptime(period_end, "%Y-%m-%d %H:%M:%S")

            patch_info, block_subject_patch_data, block_subject_patch_pref = (
                load_subject_patch_data(key, period_start, period_end)
            )

            block_subject_patch_pref = block_subject_patch_pref.dropna(
                subset=["final_preference_by_time", "final_preference_by_wheel"]
            )

            if not patch_info.empty:
                patch_info.insert(0, "experiment_name", exp["name"])
                patch_info.insert(1, "period", period_name)

            if not block_subject_patch_data.empty:
                block_subject_patch_data.insert(1, "period", period_name)

                if period_name in ["presocial", "postsocial"]:
                    n_subjects = (
                        block_subject_patch_data.groupby("block_start")["subject_name"].nunique()
                    )
                    if (n_subjects != 1).any():
                        warnings.warn(
                            f"{period_name.capitalize()} data for {exp['name']} has multi-subject blocks."
                        )

            if not block_subject_patch_pref.empty:
                block_subject_patch_pref.insert(1, "period", period_name)

                for col in ["pellet_timestamps", "in_patch_rfid_timestamps", "in_patch_timestamps"]:
                    if col in block_subject_patch_data.columns:
                        block_subject_patch_data[col] = block_subject_patch_data[col].apply(ensure_ts_arr_datetime)

            patch_info_dict[exp["name"]][period_name] = patch_info
            subject_patch_data_dict[exp["name"]][period_name] = block_subject_patch_data
            subject_patch_pref_dict[exp["name"]][period_name] = block_subject_patch_pref

    return patch_info_dict, subject_patch_data_dict, subject_patch_pref_dict


def load_foraging_bouts(
    key: dict[str, str],
    period_start: str,
    period_end: str
) -> pd.DataFrame:

    blocks = (
        Block
        & key
        & f"block_start >= '{period_start}'"
        & f"block_end <= '{period_end}'"
    ).fetch("block_start")

    bouts = []

    for block_start in blocks:
        block_key = key | {"block_start": str(block_start)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            bouts.append(get_foraging_bouts(block_key, min_pellets=1))

    if bouts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return pd.concat(bouts, ignore_index=True)
    else:
        return pd.DataFrame(columns=["start", "end", "n_pellets", "cum_wheel_dist", "subject"])


def load_all_foraging_bouts(experiments: list[dict[str, str]]) -> dict[str, dict[str, pd.DataFrame]]:
    foraging_data_dict = {}

    for exp in experiments:
        key = {"experiment_name": exp["name"]}
        foraging_data_dict[exp["name"]] = {}

        periods = {}
        if "presocial_start" in exp and "presocial_end" in exp:
            periods["presocial"] = (exp["presocial_start"], exp["presocial_end"])
        if "social_start" in exp and "social_end" in exp:
            periods["social"] = (exp["social_start"], exp["social_end"])
        if "postsocial_start" in exp and "postsocial_end" in exp:
            periods["postsocial"] = (exp["postsocial_start"], exp["postsocial_end"])

        for period_name, (period_start, period_end) in tqdm(
            periods.items(),
            desc=f"Loading foraging bouts for {exp['name']}",
            leave=False
        ):
            period_start = datetime.strptime(period_start, "%Y-%m-%d %H:%M:%S")
            period_end = datetime.strptime(period_end, "%Y-%m-%d %H:%M:%S")

            foraging_df = load_foraging_bouts(key, period_start, period_end)

            if not foraging_df.empty:
                if 'experiment_name' not in foraging_df.columns:
                    foraging_df.insert(0, "experiment_name", exp["name"])
                foraging_df["period"] = period_name

            foraging_data_dict[exp["name"]][period_name] = foraging_df

    return foraging_data_dict


def load_all_position_data(
    experiments: List[Dict[str, str]],
    data_dir: Path = Path("/ceph/aeon/aeon/code/scratchpad/methods_paper_data"),
) -> Dict[str, Dict[str, pd.DataFrame]]:
    position_data_dict: Dict[str, Dict[str, pd.DataFrame]] = {}

    for exp in experiments:
        exp_name = exp["name"]
        periods = [
            p for p in ["presocial", "social", "postsocial"]
            if f"{p}_start" in exp and f"{p}_end" in exp
        ]

        position_data_dict[exp_name] = {}
        for period in tqdm(periods, desc=f"Loading position for {exp_name}", leave=False):
            filepath = data_dir / f"{exp_name}_{period}_position.parquet"
            if not filepath.exists():
                position_data_dict[exp_name][period] = pd.DataFrame()
                continue

            df = pd.read_parquet(filepath).dropna(axis=1, how="all")
            if df.empty:
                position_data_dict[exp_name][period] = pd.DataFrame()
                continue

            # tag experiment and period
            df["experiment_name"] = exp_name
            df["period"] = period

            # restrict to the period time window
            start = pd.to_datetime(exp[f"{period}_start"])
            end = pd.to_datetime(exp[f"{period}_end"])
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                df = df[(df["time"] >= start) & (df["time"] < end)]
            
            position_data_dict[exp_name][period] = df

    return position_data_dict

