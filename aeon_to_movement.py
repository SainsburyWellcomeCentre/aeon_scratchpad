import numpy as np
import pandas as pd

# import xarray as xr
# from movement.io.load_poses import from_numpy
from swc.aeon.io import api as io_api
from swc.aeon.io import reader as io_reader

root = "Z:/aeon/data/ingest"
pose_reader = io_reader.Pose("CameraTop_222*", root)
pose = io_api.load(
    root=root,
    reader=pose_reader,
    start=pd.Timestamp("2024-07-02 14:00"),
    end=pd.Timestamp("2024-07-02 14:10"),
    include_model=False,
)
pose = pose[pose.identity.isin(["BAA-1104569", "BAA-1104568"])]
pose = pose[~pose["part"].str.startswith("anchor_")]
pose.sort_index(inplace=True)

# Infer FPS
frames = pose.index.drop_duplicates()
time_diff = frames.to_series().diff().dt.total_seconds()
mean_time_diff = time_diff[1:].mean()
fps = float(np.ceil(1 / mean_time_diff)) if mean_time_diff > 0 else None

keypoints = pose["part"].unique()
individuals = pose["identity"].unique()
# Create position array
# Slow...
frame_idx = {v: i for i, v in enumerate(frames)}
kp_idx = {v: i for i, v in enumerate(keypoints)}
ind_idx = {v: i for i, v in enumerate(individuals)}
pos = np.full((len(frames), 2, len(keypoints), len(individuals)), np.nan, dtype=float)
for idx, row in pose.iterrows():
    f = frame_idx[idx]
    kp = kp_idx[row["part"]]
    ind = ind_idx[row["identity"]]
    pos[f, 0, kp, ind] = row["x"]
    pos[f, 1, kp, ind] = row["x"]

# Faster?
multi_idx = pd.MultiIndex.from_product(
    [frames, keypoints, individuals], names=["frame", "part", "identity"]
)
pose_reset = pose.reset_index().set_index(["time", "part", "identity"])
pose_full = pose_reset.reindex(multi_idx)
x_arr = pose_full["x"].values.reshape(len(frames), len(keypoints), len(individuals))
y_arr = pose_full["y"].values.reshape(len(frames), len(keypoints), len(individuals))
pose_arr = np.stack([x_arr, y_arr], axis=1).astype(float)
np.allclose(pose_arr, pos, equal_nan=True)

# from_numpy args
# position_array (np.ndarray) – Array of shape (n_frames, n_space, n_keypoints, n_individuals) containing the poses. It will be converted to a xarray.DataArray object named “position”.

# confidence_array (np.ndarray, optional) – Array of shape (n_frames, n_keypoints, n_individuals) containing the point-wise confidence scores. It will be converted to a xarray.DataArray object named “confidence”. If None (default), the scores will be set to an array of NaNs.

# individual_names (list of str, optional) – List of unique names for the individuals in the video. If None (default), the individuals will be named “individual_0”, “individual_1”, etc.

# keypoint_names (list of str, optional) – List of unique names for the keypoints in the skeleton. If None (default), the keypoints will be named “keypoint_0”, “keypoint_1”, etc.

# fps (float, optional) – Frames per second of the video. Defaults to None, in which case the time coordinates will be in frame numbers.

# source_software (str, optional) – Name of the pose estimation software from which the data originate. Defaults to None.
