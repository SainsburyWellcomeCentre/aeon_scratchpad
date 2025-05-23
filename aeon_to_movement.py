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
# Out-of-order timestamps so full chunk returned
pose.sort_index(inplace=True)
pose = pose.loc["2024-07-02 14:00:00":"2024-07-02 14:10:00"]
# Convert index to seconds
pose.index = (pose.index - pose.index[0]).total_seconds()
# Infer FPS
frames = pose.index.drop_duplicates().to_numpy()
mean_time_diff = np.diff(frames).mean()
fps = float(np.ceil(1 / mean_time_diff)) if mean_time_diff > 0 else None

keypoints = pose["part"].unique()
individuals = pose["identity"].unique()

# Create position array
multi_idx = pd.MultiIndex.from_product(
    [frames, keypoints, individuals], names=["frame", "part", "identity"]
)
pose_reset = pose.reset_index().set_index(["time", "part", "identity"])
pose_full = pose_reset.reindex(multi_idx)
x_arr = pose_full["x"].values.reshape(len(frames), len(keypoints), len(individuals))
y_arr = pose_full["y"].values.reshape(len(frames), len(keypoints), len(individuals))
pose_arr = np.stack([x_arr, y_arr], axis=1).astype(float)
# Create confidence array
conf_arr = pose_full["part_likelihood"].values.reshape(
    len(frames), len(keypoints), len(individuals)
)
