"""
Works with exp01 data and codebase (666cff commit on aeon_mecha)
"""

# Imports
from pathlib import Path
from glob import glob
import random

import pandas as pd
import numpy as np
from numpy import logical_and as ele_AND
from numpy import logical_or as ele_OR
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import datajoint as dj
from dotmap import DotMap

from aeon.preprocess import api
from aeon_analysis.social import helpers

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 200

# Plots:
# 1) 2-d occupancy histograms
# 2) 1-d occupancy distributions
# 3) sns pairplot

# <s Get session metadata

exp_root = '/nfs/winstor/delab/data/arena0.1/socialexperiment0_raw'
session_metadata = helpers.loadSessions(exp_root)
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

# <s Individual session analysis: for individual sessions, get occupancy information
# (proportion of time in rich patch (rp), poor patch (pp), or neither (n))

# Patch locations, and threshold to be considered "in patch"
p2_x, p2_y = (370, 590)
p1_x, p1_y = (1070, 590)
pix_radius_thresh = 180

# Craft occupancy arrays for each individual
n_ind_ses = 7
n_occ_locs = 3  # (rp, pp, np)
occ_704 = np.zeros((n_ind_ses, n_occ_locs))
occ_704_pre = np.zeros((n_ind_ses, n_occ_locs))
occ_704_post = np.zeros((n_ind_ses, n_occ_locs))
ses_ct_704 = 0
occ_705 = np.zeros((n_ind_ses, n_occ_locs))
occ_705_pre = np.zeros((n_ind_ses, n_occ_locs))
occ_705_post = np.zeros((n_ind_ses, n_occ_locs))
ses_ct_705 = 0
occ_706 = np.zeros((n_ind_ses, n_occ_locs))
occ_706_pre = np.zeros((n_ind_ses, n_occ_locs))
occ_706_post = np.zeros((n_ind_ses, n_occ_locs))
ses_ct_706 = 0

# Individual session analysis
for s in session_metadata.itertuples():
    # Skip social sessions.
    if ";" in s.id:
        continue
    # Find rich patch (rp) and poor patch (pp).
    p1_patch_data = api.patchdata(exp_root, 'Patch1', start=s.start, end=s.end)
    p2_patch_data = api.patchdata(exp_root, 'Patch2', start=s.start, end=s.end)
    if p1_patch_data.threshold[-1] == 100:
        rp_x, rp_y = p1_x, p1_y
        pp_x, pp_y = p2_x, p2_y
    else:
        rp_x, rp_y = p2_x, p2_y
        pp_x, pp_y = p1_x, p1_y
    pos = api.positiondata(exp_root, start=s.start, end=s.end)
    pos = pos[pos.id == 0.0]  # look at id 0 only b/c we don't care for duplicate ts
    pos = pos[~np.isnan(pos.x)]  # very rarely nans occur in bonsai tracking: throwaway
    both_patch_data = pd.concat([p1_patch_data, p2_patch_data])
    change_ts = (
        both_patch_data.iloc[np.where(np.diff(both_patch_data.threshold))[0][0]].name)

    # *Note*: there should be a function with arguments (timestamp,
    # mouse position, arena roi position) that given a timestamp, calculates the
    # distance of any mouse to any specified location
    change_idx = (change_ts - s.start).seconds * 50
    dist2rp = np.sqrt( ((pos.x - rp_x) ** 2) + ((pos.y - rp_y) ** 2) )
    rp_p = len(np.where(dist2rp < pix_radius_thresh)[0]) / len(pos)
    rp_p_pre = (len(np.where(dist2rp[0:change_idx] < pix_radius_thresh)[0]) 
                / len(pos[0:change_idx]))
    rp_p_post = (len(np.where(dist2rp[change_idx:] < pix_radius_thresh)[0]) 
                 / len(pos[change_idx:]))
    dist2pp = np.sqrt(((pos.x - pp_x) ** 2) + ((pos.y - pp_y) ** 2))
    pp_p = len(np.where(dist2pp < pix_radius_thresh)[0]) / len(pos)
    pp_p_pre = (len(np.where(dist2pp[0:change_idx] < pix_radius_thresh)[0]) 
                / len(pos[0:change_idx]))
    pp_p_post = (len(np.where(dist2pp[change_idx:] < pix_radius_thresh)[0]) 
                 / len(pos[change_idx:]))
    if '704' in s.id:
        occ_704[ses_ct_704, :] = (rp_p, pp_p, (1 - (rp_p + pp_p)))
        occ_704_pre[ses_ct_704, :] = (rp_p_pre, pp_p_pre, (1 - (rp_p_pre + pp_p_pre)))
        occ_704_post[ses_ct_704, :] = (rp_p_post, pp_p_post, 
                                       (1 - (rp_p_post + pp_p_post)))
        ses_ct_704 += 1
    elif '705' in s.id:
        occ_705[ses_ct_705, :] = (rp_p, pp_p, (1 - (rp_p + pp_p)))
        occ_705_pre[ses_ct_705, :] = (rp_p_pre, pp_p_pre, (1 - (rp_p_pre + pp_p_pre)))
        occ_705_post[ses_ct_705, :] = (rp_p_post, pp_p_post, 
                                       (1 - (rp_p_post + pp_p_post)))
        ses_ct_705 += 1
    elif '706' in s.id:
        occ_706[ses_ct_706, :] = (rp_p, pp_p, (1 - (rp_p + pp_p)))
        occ_706_pre[ses_ct_706, :] = (rp_p_pre, pp_p_pre, (1 - (rp_p_pre + pp_p_pre)))
        occ_706_post[ses_ct_706, :] = (rp_p_post, pp_p_post, 
                                       (1 - (rp_p_post + pp_p_post)))
        ses_ct_706 += 1
# /s>

# <s Social session analysis:
# For each session, get occupancy information for each mouse individually,

p2_x, p2_y = (280, 350)
p1_x, p1_y = (970, 350)
pix_radius_thresh = 180

# pairings = DotMap({
#     '01-21': DotMap({'ind1': '705', 'ind2': '706'}),
#     '01-25': DotMap({'ind1': '704', 'ind2': '705'}),
#     '01-26': DotMap({'ind1': '704', 'ind2': '706'}),
#     '01-27': DotMap({'ind2': '705', 'ind1': '706'}),
#     '02-01': DotMap({'ind1': '704', 'ind2': '705'}),
#     '02-02': DotMap({'ind1': '704', 'ind2': '706'}),
#     '02-03': DotMap({'ind1': '705', 'ind2': '706'}),
#     '02-08': DotMap({'ind1': '704', 'ind2': '705'}),
#     '02-09': DotMap({'ind1': '704', 'ind2': '706'}),
#     '02-10': DotMap({'ind1': '705', 'ind2': '706'}),
# })
# [1, 2, 3] = rp, pp, np: ind1; [11, 12, 13] = rp, pp, np: ind2
# ['rp_rp' : 11, 'rp_pp' : 12, 'rp_np' : 13,
#  'pp_rp' : 22, 'pp_pp' : 24, 'pp_np' : 26,
#  'np_rp' : 33, 'np_pp' : 36, 'np_np' : 39]

# Good ma id-tracked sessions are: 1-21, 1-25, 2-03, 2-08, 2-09
h5s = glob("/nfs/nhome/live/jbhagat/dlc_playground/occupancy_video_analysis/*.h5")
h5s.sort()
h5s = [h5s[0], h5s[1], h5s[6], h5s[7], h5s[8]]
# Each figure is 3x3
# fig_704_705
# fig_704_706
# fig_705_706
# for each social session
# social_sess = [54, 55, 56, 58, 67, 68, 70, 76, 78, 80]
social_sess = [54, 55, 70, 76, 78]
# p_diffs = [14, 14, 14, 14, 6.5, 6.5, 6.5, 2.75, 2.75, 2.75]
p_diffs = [1400, 1400, 650, 275, 275]

# CHANGE
i = 4; h5 = h5s[i]
for i, h5 in enumerate(h5s):
    p_diff = p_diffs[i]
    s = session_metadata.loc[social_sess[i]]
    print(s.id)
    occ1 = occ_704
    occ1_pre = occ_704_pre
    occ1_post = occ_704_post
    occ2 = occ_706
    occ2_pre = occ_706_pre
    occ2_post = occ_706_post
    ids = '704-706_' + str(s.start)[0:10]
    # 704: autumn, 705: summer, 706: winter
    map1 = 'autumn'; map2 = 'winter'
    
    # Find rich patch (rp) and poor patch (pp).
    p1_patch_data = api.patchdata(exp_root, 'Patch1', start=s.start, end=s.end)
    p2_patch_data = api.patchdata(exp_root, 'Patch2', start=s.start, end=s.end)
    if p1_patch_data.threshold[-1] == 100:
        rp_x, rp_y = p1_x, p1_y
        pp_x, pp_y = p2_x, p2_y
    else:
        rp_x, rp_y = p2_x, p2_y
        pp_x, pp_y = p1_x, p1_y
    both_patch_data = pd.concat([p1_patch_data, p2_patch_data])
    change_ts = (
        both_patch_data.iloc[np.where(np.diff(both_patch_data.threshold))[0][0]].name)
    r = random.randint(0, 1000)
    df = pd.read_hdf(h5)
    # flatten multi-index df
    new_cols = np.zeros_like(df.columns.values)
    for i_col, col in enumerate(df.columns.values):
        new_cols[i_col] = '_'.join(df.columns.values[i_col][1:])
    df.columns = new_cols

    change_idx = (change_ts - pd.Timestamp(Path(h5).parts[-1].split('_')[
                                              1]).tz_localize(None)).seconds * 50
    # Get ind1 positions
    ind1_x = df.ind1_body_right_top_x
    ind1_x[np.isnan(ind1_x)] = df.ind1_body_right_bottom_x[np.isnan(ind1_x)]
    ind1_x[np.isnan(ind1_x)] = df.ind1_neck_base_x[np.isnan(ind1_x)]
    ind1_x[np.isnan(ind1_x)] = df.ind1_tail_base_x[np.isnan(ind1_x)]
    ind1_nans = np.where(np.isnan(ind1_x))[0]
    ind1_x2sample = ind1_x[~np.isnan(ind1_x)]
    random.seed(r)
    ind1_x[ind1_nans] = (
        random.choices(ind1_x2sample.values, k=len(np.where(np.isnan(ind1_x))[0])))
    ind1_x_pre = ind1_x[0:change_idx]
    ind1_x_post = ind1_x[change_idx:]
    
    ind1_y = df.ind1_body_right_top_y
    ind1_y[np.isnan(ind1_y)] = df.ind1_body_right_bottom_y[np.isnan(ind1_y)]
    ind1_y[np.isnan(ind1_y)] = df.ind1_neck_base_y[np.isnan(ind1_y)]
    ind1_y[np.isnan(ind1_y)] = df.ind1_tail_base_y[np.isnan(ind1_y)]
    ind1_y2sample = ind1_y[~np.isnan(ind1_y)]
    random.seed(r)
    ind1_y[np.isnan(ind1_y)] = (
        random.choices(ind1_y2sample.values, k=len(np.where(np.isnan(ind1_y))[0])))
    ind1_y_pre = ind1_y[0:change_idx]
    ind1_y_post = ind1_y[change_idx:]

    ind1_dist2rp = np.sqrt(((ind1_x - rp_x) ** 2) + ((ind1_y - rp_y) ** 2))
    ind1_rp_locs = np.where(ind1_dist2rp < pix_radius_thresh)[0]
    ind1_rp_locs_pre = ind1_rp_locs[0:change_idx]
    ind1_rp_locs_post = ind1_rp_locs[change_idx:]
    ind1_dist2pp = np.sqrt(((ind1_x - pp_x) ** 2) + ((ind1_y - pp_y) ** 2))
    ind1_pp_locs = np.where(ind1_dist2pp < pix_radius_thresh)[0]
    ind1_pp_locs_pre = ind1_pp_locs[0:change_idx]
    ind1_pp_locs_post = ind1_pp_locs[change_idx:]
    ind1_np_locs = np.setdiff1d(df.index.values,
                                np.concatenate((ind1_rp_locs, ind1_pp_locs)))
    ind1_np_locs_pre = ind1_np_locs[0:change_idx]
    ind1_np_locs_post = ind1_np_locs[change_idx:]

    # Get ind2 positions
    ind2_x = df.ind2_body_right_top_x
    ind2_x[np.isnan(ind2_x)] = df.ind2_body_right_bottom_x[np.isnan(ind2_x)]
    ind2_x[np.isnan(ind2_x)] = df.ind2_neck_base_x[np.isnan(ind2_x)]
    ind2_x[np.isnan(ind2_x)] = df.ind2_tail_base_x[np.isnan(ind2_x)]
    ind2_nans = np.where(np.isnan(ind2_x))[0]
    ind2_x2sample = ind2_x[~np.isnan(ind2_x)]
    random.seed(r)
    ind2_x[ind2_nans] = (
        random.choices(ind2_x2sample.values, k=len(np.where(np.isnan(ind2_x))[0])))
    ind2_x_pre = ind2_x[0:change_idx]
    ind2_x_post = ind2_x[change_idx:]

    ind2_y = df.ind2_body_right_top_y
    ind2_y[np.isnan(ind2_y)] = df.ind2_body_right_bottom_y[np.isnan(ind2_y)]
    ind2_y[np.isnan(ind2_y)] = df.ind2_neck_base_y[np.isnan(ind2_y)]
    ind2_y[np.isnan(ind2_y)] = df.ind2_tail_base_y[np.isnan(ind2_y)]
    ind2_y2sample = ind2_y[~np.isnan(ind2_y)]
    random.seed(r)
    ind2_y[np.isnan(ind2_y)] = (
        random.choices(ind2_y2sample.values, k=len(np.where(np.isnan(ind2_y))[0])))
    ind2_y_pre = ind2_y[0:change_idx]
    ind2_y_post = ind2_y[change_idx:]

    ind2_dist2rp = np.sqrt(((ind2_x - rp_x) ** 2) + ((ind2_y - rp_y) ** 2))
    ind2_rp_locs = np.where(ind2_dist2rp < pix_radius_thresh)[0]
    ind2_rp_locs_pre = ind2_rp_locs[0:change_idx]
    ind2_rp_locs_post = ind2_rp_locs[change_idx:]
    ind2_dist2pp = np.sqrt(((ind2_x - pp_x) ** 2) + ((ind2_y - pp_y) ** 2))
    ind2_pp_locs = np.where(ind2_dist2pp < pix_radius_thresh)[0]
    ind2_pp_locs_pre = ind2_pp_locs[0:change_idx]
    ind2_pp_locs_post = ind2_pp_locs[change_idx:]
    ind2_np_locs = np.setdiff1d(df.index.values,
                                np.concatenate((ind2_rp_locs, ind2_pp_locs)))
    ind2_np_locs_pre = ind2_np_locs[0:change_idx]
    ind2_np_locs_post = ind2_np_locs[change_idx:]

    # Create 2d occupancy histogram plots
    fig, ax = plt.subplots(nrows=1, ncols=2)
    # Change binsizes for these so that its proportional to x and y pixels
    sns.histplot(x=ind1_x_pre, y=ind1_y_pre, ax=ax[0], stat='percent', bins=30, 
                 cbar=True, cmap=map1)
    sns.histplot(x=ind2_x_pre, y=ind2_y_pre, ax=ax[1], stat='percent', bins=30, 
                 cbar=True, cmap=map2)
    fig.suptitle(ids + f"_pre rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} pel; "
                       f"thresh_diff: {p_diff}")
    fig.savefig(f"/nfs/winstor/delab/lab members/jai/social_analysis_figs/2d_occ_hists"
                f"/{ids}_pre.png")
    fig.show()
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.histplot(x=ind1_x_post, y=ind1_y_post, ax=ax[0], stat='percent', bins=30, 
                 cbar=True, cmap=map1)
    sns.histplot(x=ind2_x_post, y=ind2_y_post, ax=ax[1], stat='percent', bins=30, 
                 cbar=True, cmap=map2)
    fig.suptitle(ids + f"_post rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} pel; "
                       f"thresh_diff: {p_diff}")
    fig.savefig(f"/nfs/winstor/delab/lab members/jai/social_analysis_figs/2d_occ_hists"
                f"/{ids}_post.png")
    fig.show()

    # Get combined positions
    ind1_locs = np.zeros(len(df))
    ind1_locs[ind1_rp_locs] = 1   # rp
    ind1_locs[ind1_pp_locs] = 2   # pp
    ind1_locs[ind1_np_locs] = 3   # np
    ind2_locs = np.zeros(len(df))
    ind2_locs[ind2_rp_locs] = 11  # rp
    ind2_locs[ind2_pp_locs] = 12  # pp
    ind2_locs[ind2_np_locs] = 13  # np
    comb_locs = ind1_locs * ind2_locs
    rp_rp_p = len(comb_locs[comb_locs == 11]) / len(df)
    rp_pp_p = len(comb_locs[comb_locs == 12]) / len(df)
    rp_np_p = len(comb_locs[comb_locs == 13]) / len(df)
    pp_rp_p = len(comb_locs[comb_locs == 22]) / len(df)
    pp_pp_p = len(comb_locs[comb_locs == 24]) / len(df)
    pp_np_p = len(comb_locs[comb_locs == 26]) / len(df)
    np_rp_p = len(comb_locs[comb_locs == 33]) / len(df)
    np_pp_p = len(comb_locs[comb_locs == 36]) / len(df)
    np_np_p = len(comb_locs[comb_locs == 39]) / len(df)
    comb_pos = np.array(((rp_rp_p, rp_pp_p, rp_np_p), 
                        (pp_rp_p, pp_pp_p, pp_np_p),
                        (np_rp_p, np_pp_p, np_np_p)))
    
    comb_locs_pre = ind1_locs[0:change_idx] * ind2_locs[0:change_idx]
    rp_rp_p_pre = len(comb_locs_pre[comb_locs_pre == 11]) / len(df[0:change_idx])
    rp_pp_p_pre = len(comb_locs_pre[comb_locs_pre == 12]) / len(df[0:change_idx])
    rp_np_p_pre = len(comb_locs_pre[comb_locs_pre == 13]) / len(df[0:change_idx])
    pp_rp_p_pre = len(comb_locs_pre[comb_locs_pre == 22]) / len(df[0:change_idx])
    pp_pp_p_pre = len(comb_locs_pre[comb_locs_pre == 24]) / len(df[0:change_idx])
    pp_np_p_pre = len(comb_locs_pre[comb_locs_pre == 26]) / len(df[0:change_idx])
    np_rp_p_pre = len(comb_locs_pre[comb_locs_pre == 33]) / len(df[0:change_idx])
    np_pp_p_pre = len(comb_locs_pre[comb_locs_pre == 36]) / len(df[0:change_idx])
    np_np_p_pre = len(comb_locs_pre[comb_locs_pre == 39]) / len(df[0:change_idx])
    comb_pos_pre = np.array(((rp_rp_p_pre, rp_pp_p_pre, rp_np_p_pre),
                         (pp_rp_p_pre, pp_pp_p_pre, pp_np_p_pre),
                         (np_rp_p_pre, np_pp_p_pre, np_np_p_pre)))

    comb_locs_post = ind1_locs[change_idx:] * ind2_locs[change_idx:]
    rp_rp_p_post = len(comb_locs_post[comb_locs_post == 11]) / len(df[change_idx:])
    rp_pp_p_post = len(comb_locs_post[comb_locs_post == 12]) / len(df[change_idx:])
    rp_np_p_post = len(comb_locs_post[comb_locs_post == 13]) / len(df[change_idx:])
    pp_rp_p_post = len(comb_locs_post[comb_locs_post == 22]) / len(df[change_idx:])
    pp_pp_p_post = len(comb_locs_post[comb_locs_post == 24]) / len(df[change_idx:])
    pp_np_p_post = len(comb_locs_post[comb_locs_post == 26]) / len(df[change_idx:])
    np_rp_p_post = len(comb_locs_post[comb_locs_post == 33]) / len(df[change_idx:])
    np_pp_p_post = len(comb_locs_post[comb_locs_post == 36]) / len(df[change_idx:])
    np_np_p_post = len(comb_locs_post[comb_locs_post == 39]) / len(df[change_idx:])
    comb_pos_post = np.array(((rp_rp_p_post, rp_pp_p_post, rp_np_p_post),
                             (pp_rp_p_post, pp_pp_p_post, pp_np_p_post),
                             (np_rp_p_post, np_pp_p_post, np_np_p_post)))
    

    # Generate synthetic null distribution for each combo of occupancy
    rp_rp_null_dist_pre = np.outer(occ1_pre[:, 0], occ2_pre[:, 0])
    rp_rp_null_dist_pre = random.choices(rp_rp_null_dist_pre.flatten(), k=100)
    rp_pp_null_dist_pre = np.outer(occ1_pre[:, 0], occ2_pre[:, 1])
    rp_pp_null_dist_pre = random.choices(rp_pp_null_dist_pre.flatten(), k=100)
    rp_np_null_dist_pre = np.outer(occ1_pre[:, 0], occ2_pre[:, 2])
    rp_np_null_dist_pre = random.choices(rp_np_null_dist_pre.flatten(), k=100)
    pp_rp_null_dist_pre = np.outer(occ1_pre[:, 1], occ2_pre[:, 0])
    pp_rp_null_dist_pre = random.choices(pp_rp_null_dist_pre.flatten(), k=100)
    pp_pp_null_dist_pre = np.outer(occ1_pre[:, 1], occ2_pre[:, 1])
    pp_pp_null_dist_pre = random.choices(pp_pp_null_dist_pre.flatten(), k=100)
    pp_np_null_dist_pre = np.outer(occ1_pre[:, 1], occ2_pre[:, 2])
    pp_np_null_dist_pre = random.choices(pp_np_null_dist_pre.flatten(), k=100)
    np_rp_null_dist_pre = np.outer(occ1_pre[:, 2], occ2_pre[:, 0])
    np_rp_null_dist_pre = random.choices(np_rp_null_dist_pre.flatten(), k=100)
    np_pp_null_dist_pre = np.outer(occ1_pre[:, 2], occ2_pre[:, 1])
    np_pp_null_dist_pre = random.choices(np_pp_null_dist_pre.flatten(), k=100)
    np_np_null_dist_pre = np.outer(occ1_pre[:, 2], occ2_pre[:, 2])
    np_np_null_dist_pre = random.choices(np_np_null_dist_pre.flatten(), k=100)
    all_dists_pre = np.array((
        (rp_rp_null_dist_pre),
        (rp_pp_null_dist_pre),
        (rp_np_null_dist_pre),
        (pp_rp_null_dist_pre),
        (pp_pp_null_dist_pre),
        (pp_np_null_dist_pre),
        (np_rp_null_dist_pre),
        (np_pp_null_dist_pre),
        (np_np_null_dist_pre),
    ))
    all_dists_pre = all_dists_pre.transpose()

    rp_rp_null_dist_post = np.outer(occ1_post[:, 0], occ2_post[:, 0])
    rp_rp_null_dist_post = random.choices(rp_rp_null_dist_post.flatten(), k=100)
    rp_pp_null_dist_post = np.outer(occ1_post[:, 0], occ2_post[:, 1])
    rp_pp_null_dist_post = random.choices(rp_pp_null_dist_post.flatten(), k=100)
    rp_np_null_dist_post = np.outer(occ1_post[:, 0], occ2_post[:, 2])
    rp_np_null_dist_post = random.choices(rp_np_null_dist_post.flatten(), k=100)
    pp_rp_null_dist_post = np.outer(occ1_post[:, 1], occ2_post[:, 0])
    pp_rp_null_dist_post = random.choices(pp_rp_null_dist_post.flatten(), k=100)
    pp_pp_null_dist_post = np.outer(occ1_post[:, 1], occ2_post[:, 1])
    pp_pp_null_dist_post = random.choices(pp_pp_null_dist_post.flatten(), k=100)
    pp_np_null_dist_post = np.outer(occ1_post[:, 1], occ2_post[:, 2])
    pp_np_null_dist_post = random.choices(pp_np_null_dist_post.flatten(), k=100)
    np_rp_null_dist_post = np.outer(occ1_post[:, 2], occ2_post[:, 0])
    np_rp_null_dist_post = random.choices(np_rp_null_dist_post.flatten(), k=100)
    np_pp_null_dist_post = np.outer(occ1_post[:, 2], occ2_post[:, 1])
    np_pp_null_dist_post = random.choices(np_pp_null_dist_post.flatten(), k=100)
    np_np_null_dist_post = np.outer(occ1_post[:, 2], occ2_post[:, 2])
    np_np_null_dist_post = random.choices(np_np_null_dist_post.flatten(), k=100)
    all_dists_post = np.array((
        (rp_rp_null_dist_post),
        (rp_pp_null_dist_post),
        (rp_np_null_dist_post),
        (pp_rp_null_dist_post),
        (pp_pp_null_dist_post),
        (pp_np_null_dist_post),
        (np_rp_null_dist_post),
        (np_pp_null_dist_post),
        (np_np_null_dist_post),
    ))
    all_dists_post = all_dists_post.transpose()
    
    titles = ['rp-rp', 'rp-pp', 'rp-np', 'pp-rp', 'pp-pp', 'pp-np', 'np-rp', 'np-pp',
              'np-np']
    # 1-d dist plots
    fig, axs = plt.subplots(nrows=3, ncols=3)
    for occ_idx in range(9):
        ax = axs.flatten()[occ_idx]
        tru_val = comb_pos_pre.flatten()[occ_idx].round(4)
        sns.histplot(all_dists_pre[:, occ_idx], bins=25, kde=True, ax=ax, fill=True)
        vh = ax.axvline(x=tru_val, ymax=ax.get_ylim()[1], color='m')
        ax.set_title(titles[occ_idx])
        ax.set_ylabel('')
        # ax.legend([vh], ['syn_null_dist', f'actual={tru_val}'])
    axs[2, 0].set_xlabel('P[x]')
    axs[2, 0].set_ylabel('Count')
    fig.suptitle(ids + f"_pre rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} pel; "
                       f"thresh_diff: {p_diff}")
    fig.savefig(f"/nfs/winstor/delab/lab members/jai/social_analysis_figs/1d_occ_dists/"
                f"/{ids}_pre.png")
    fig.show()

    fig, axs = plt.subplots(nrows=3, ncols=3)
    for occ_idx in range(9):
        ax = axs.flatten()[occ_idx]
        tru_val = comb_pos_post.flatten()[occ_idx].round(4)
        sns.histplot(all_dists_post[:, occ_idx], bins=25, kde=True, ax=ax, fill=True)
        vh = ax.axvline(x=tru_val, ymax=ax.get_ylim()[1], color='m')
        ax.set_title(titles[occ_idx])
        ax.set_ylabel('')
        # ax.legend([vh], ['syn_null_dist', f'actual={tru_val}'])
    axs[2, 0].set_xlabel('P[x]')
    axs[2, 0].set_ylabel('Count')
    fig.suptitle(ids + f"_post rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} pel; "
                       f"thresh_diff: {p_diff}")
    fig.savefig(f"/nfs/winstor/delab/lab members/jai/social_analysis_figs/1d_occ_dists/"
                f"/{ids}_post.png")
    fig.show()

    # Pairplots
    all_dists_df_pre = pd.DataFrame(all_dists_pre)
    all_dists_df_pre.columns = titles
    emp_vs_syn = pd.Series()
    all_dists_df_pre['emp_vs_syn'] = 'syn'
    # add empirical data
    all_dists_df_pre = all_dists_df_pre.append(pd.Series(comb_pos_pre.flatten(),
                                                     index=titles),
                                       ignore_index=True)
    all_dists_df_pre.emp_vs_syn.iloc[-1] = 'emp'
    fig_pp = sns.pairplot(all_dists_df_pre, hue="emp_vs_syn")
    fig_pp.fig.suptitle(ids + f"_pre rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} "
                              f"pel; thresh_diff: {p_diff}")
    fig_pp.savefig(f"/nfs/winstor/delab/lab members/jai/social_analysis_figs/pairplots/"
                f"/{ids}_pre.png")
    fig_pp.fig.show()

    all_dists_df_post = pd.DataFrame(all_dists_post)
    all_dists_df_post.columns = titles
    emp_vs_syn = pd.Series()
    all_dists_df_post['emp_vs_syn'] = 'syn'
    # add empirical data
    all_dists_df_post = all_dists_df_post.append(pd.Series(comb_pos_post.flatten(),
                                                         index=titles),
                                               ignore_index=True)
    all_dists_df_post.emp_vs_syn.iloc[-1] = 'emp'
    fig_pp = sns.pairplot(all_dists_df_post, hue="emp_vs_syn")
    fig_pp.fig.suptitle(ids + f"_post rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} "
                              f"pel; thresh_diff: {p_diff}")
    fig_pp.savefig(f"/nfs/winstor/delab/lab members/jai/social_analysis_figs/pairplots/"
                   f"/{ids}_post.png")
    fig_pp.fig.show()


#     find animal ids corresponding to social sessions
#         for zipped list of occupancy hists with those animal ids
#             uniformly randomly sample each list
#             combine them to get 3x3 "null distribution"
#             randomly sample 9 values 100 times, normalize, plot on axes