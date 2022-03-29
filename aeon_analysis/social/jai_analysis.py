from pathlib import Path
from importlib import reload

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datajoint as dj

from aeon.preprocess import api
from aeon_analysis.social import helpers

# <s Read metadata and filter sessions
exp_root = '/nfs/winstor/delab/data/arena0.1/socialexperiment0'
session_metadata = helpers.loadSessions(exp_root)
# NOTE: data of 187.5 thresh sessions from 2022 02 15-17 is MISSING!
start_ts = pd.Timestamp('2022-01-16')
end_ts = pd.Timestamp('2022-02-11')
i_good_sess = (np.logical_and(session_metadata['start'] > start_ts, session_metadata[
    'start'] < end_ts))
session_metadata = session_metadata[i_good_sess]
# manual_metadata = pd.read_csv(
#     '/nfs/winstor/delab/data/arena0.1/exp01_session_spreadsheet_cohorta_2022_01.tsv')
bad_sess = [pd.Timestamp('2022-01-18 12:32:00'), pd.Timestamp('2022-01-19 12:28:00'),
            pd.Timestamp('2022-01-19 14:57:00'), pd.Timestamp('2022-01-31 14:18:00')]
i_bad_sess = [np.argmin(np.abs(session_metadata['start'] - ts)) for ts in bad_sess]
session_metadata.drop(index=session_metadata.iloc[i_bad_sess].index, inplace=True)
# /s>

# <s Load session data (per patch: wheel distance travelled; pellet; patch)

# Final df placeholder
data = pd.DataFrame(
    columns=['SessionType', 'id', 'id2', 'thresh_change_ts', 'p1_thresh', 'p2_thresh',
             'p2-p1_thresh', 'p1_cum_wheel_dist', 'p2_cum_wheel_dist', 'p1_dist_pref',
             'p1_cum_wheel_dist_prethresh', 'p2_cum_wheel_dist_prethresh',
             'p1_cum_wheel_dist_postthresh', 'p2_cum_wheel_dist_postthresh',
             'p1_dist_pref_post_pre_ratio'], index=session_metadata.index)
for s in session_metadata.itertuples():
    # Load relevant data: per session per patch: patch, pellet, wheel distance
    p1_patch_data = api.patchdata(exp_root, 'Patch1', start=s.start, end=s.end)
    p2_patch_data = api.patchdata(exp_root, 'Patch2', start=s.start, end=s.end)
    p1_pellet_data = api.pelletdata(exp_root, 'Patch1', start=s.start, end=s.end)
    p2_pellet_data = api.patchdata(exp_root, 'Patch2', start=s.start, end=s.end)
    p1_wheel_data = api.encoderdata(exp_root, 'Patch1', start=s.start, end=s.end)
    p2_wheel_data = api.encoderdata(exp_root, 'Patch2', start=s.start, end=s.end)

    # Set up `data` df
    data.loc[s.Index, "SessionType"] = "social" if ";" in s.id else "individual"
    if ";" in s.id:
        data.loc[s.Index, "id"], data.loc[s.Index, "id2"] = s.id.split(";")
    else:
        data.loc[s.Index, "id"] = s.id
    both_patch_data = pd.concat([p1_patch_data, p2_patch_data])
    change_ts = (
        both_patch_data.iloc[np.where(np.diff(both_patch_data.threshold))[0][0]].name)
    data.loc[s.Index, "thresh_change_ts"] = change_ts
    data.loc[s.Index, "p1_thresh"] = np.max(p1_patch_data.threshold) / 100
    data.loc[s.Index, "p2_thresh"] = np.max(p2_patch_data.threshold) / 100
    data.loc[s.Index, "p2-p1_thresh"] = (
            data.loc[s.Index, "p2_thresh"] - data.loc[s.Index, "p1_thresh"])
    p1_dist = api.distancetravelled(p1_wheel_data.angle) / 100
    p2_dist = api.distancetravelled(p2_wheel_data.angle) / 100
    data.loc[s.Index, "p1_cum_wheel_dist"] = p1_dist[-1]
    data.loc[s.Index, "p2_cum_wheel_dist"] = p2_dist[-1]
    data.loc[s.Index, "p1_dist_pref"] = p1_dist[-1] / (p1_dist[-1] + p2_dist[-1])
    p1_dist_pre = p1_dist[p1_dist.index > change_ts][0]
    p2_dist_pre = p2_dist[p2_dist.index > change_ts][0]
    p1_dist_post = p1_dist[-1] - (p1_dist[p1_dist.index > change_ts][0])
    p2_dist_post = p2_dist[-1] - (p2_dist[p2_dist.index > change_ts][0])
    data.loc[s.Index, "p1_cum_wheel_dist_prethresh"] = p1_dist_pre
    data.loc[s.Index, "p2_cum_wheel_dist_prethresh"] = p2_dist_pre
    data.loc[s.Index, "p1_cum_wheel_dist_postthresh"] = p1_dist_post
    data.loc[s.Index, "p2_cum_wheel_dist_postthresh"] = p2_dist_post
    p1_pref_post = p1_dist_post / (p1_dist_post + p2_dist_post)
    p1_pref_pre = p1_dist_pre / (p1_dist_pre + p2_dist_pre)
    data.loc[s.Index, "p1_dist_pref_post_pre_ratio"] = p1_pref_post / p1_pref_pre
# /s>