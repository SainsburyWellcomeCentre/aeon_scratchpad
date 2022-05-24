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

# Plots:
# 1) 2-d occupancy histograms
# 2) 1-d occupancy distributions
# 3) sns pairplot

# <s Get session metadata

exp_root = '/nfs/winstor/delab/data/arena0.1/socialexperiment0'
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
p1_x, p1_y = (370, 590)
p2_x, p2_y = (1070, 590)
pix_radius_thresh = 180

# Craft occupancy arrays for each individual
n_ind_ses = 7
n_occ_locs = 3  # (rp, pp, n)
occ_704 = np.zeros((n_ind_ses, n_occ_locs))
ses_ct_704 = 0
occ_705 = np.zeros((n_ind_ses, n_occ_locs))
ses_ct_705 = 0
occ_706 = np.zeros((n_ind_ses, n_occ_locs))
ses_ct_706 = 0
for s in session_metadata.itertuples():
    # Skip social sessions.
    if ";" in s.id:
        continue
    # Find rich patch (rp) and poor patch (pp).
    p1_patch_data = api.patchdata(exp_root, 'Patch1', start=s.start, end=s.end)
    if p1_patch_data.threshold[-1] == 100:
        rp_x, rp_y = p1_x, p1_y
        pp_x, pp_y = p2_x, p2_y
    else:
        rp_x, rp_y = p2_x, p2_y
        pp_x, pp_y = p1_x, p1_y
    pos = api.positiondata(exp_root, start=s.start, end=s.end)
    pos = pos[pos.id == 0.0]
    pos = pos[~np.isnan(pos.x)]

    dist2rp = np.sqrt( ((pos.x - rp_x) ** 2) + ((pos.y - rp_y) ** 2) )
    rp_p = len(np.where(dist2rp < pix_radius_thresh)[0]) / len(pos)
    dist2pp = np.sqrt(((pos.x - pp_x) ** 2) + ((pos.y - pp_y) ** 2))
    pp_p = len(np.where(dist2pp < pix_radius_thresh)[0]) / len(pos)
    if '704' in s.id:
        occ_704[ses_ct_704, :] = (rp_p, pp_p, (1 - (rp_p + pp_p)))
        ses_ct_704 += 1
    elif '705' in s.id:
        occ_705[ses_ct_705, :] = (rp_p, pp_p, (1 - (rp_p + pp_p)))
        ses_ct_705 += 1
    elif '706' in s.id:
        occ_706[ses_ct_706, :] = (rp_p, pp_p, (1 - (rp_p + pp_p)))
        ses_ct_706 += 1
# /s>

# <s Social session analysis:
# For each session, get occupancy information for each mouse individually,

p1_x, p1_y = (280, 350)
p2_x, p2_y = (970, 350)
pix_radius_thresh = 180

pairings = DotMap({
    '01-21': DotMap({'ind1': '705', 'ind2': '706'}),
    '01-25': DotMap({'ind1': '704', 'ind2': '705'}),
    '01-26': DotMap({'ind1': '704', 'ind2': '706'}),
    '01-27': DotMap({'ind2': '705', 'ind1': '706'}),
    '02-01': DotMap({'ind1': '704', 'ind2': '705'}),
    '02-02': DotMap({'ind1': '704', 'ind2': '706'}),
    '02-03': DotMap({'ind1': '705', 'ind2': '706'}),
    '02-08': DotMap({'ind1': '704', 'ind2': '705'}),
    '02-09': DotMap({'ind1': '704', 'ind2': '706'}),
    '02-10': DotMap({'ind1': '705', 'ind2': '706'}),
})
# [1, 2, 3] = rp, pp, np: ind1; [11, 12, 13] = rp, pp, np: ind2
# ['rp_rp' : 11, 'rp_pp' : 12, 'rp_np' : 13,
#  'pp_rp' : 22, 'pp_pp' : 24, 'pp_np' : 26,
#  'np_rp' : 33, 'np_pp' : 36, 'np_np' : 39]
eval_key = DotMap({
    'pp_pp'
})



# Compute list of occupancy hist for each individual session for all animals
occ_705_706 = np.zeros((4, 3, 3))  # n_sessions, 704: (p1, p2, n), 705: (p1, p2, n)
ses_ct_705_706 = 0
occ_704_705 = np.zeros((3, 3, 3))  # n_sessions, 704: (p1, p2, n), 705: (p1, p2, n)
ses_ct_704_705 = 0
occ_704_706 = np.zeros((3, 3, 3))  # n_sessions, 704: (p1, p2, n), 705: (p1, p2, n)
ses_ct_704_706 = 0
h5s = glob("/nfs/nhome/live/jbhagat/dlc_playground/occupancy_video_analysis/*.h5")
h5s.sort()
# Each figure is 3x3
# fig_704_705
# fig_704_706
# fig_705_706
# for each social session
social_sess = [54, 55, 56, 58, 67, 68, 70, 76, 78, 80]
p_diffs = [14, 14, 14, 14, 6.5, 6.5, 6.5, 2.75, 2.75, 2.75]
for i, h5 in enumerate(h5s):
    s = session_metadata.loc[social_sess[i]]
    # Find rich patch (rp) and poor patch (pp).
    p1_patch_data = api.patchdata(exp_root, 'Patch1', start=s.start, end=s.end)
    if p1_patch_data.threshold[-1] == 100:
        rp_x, rp_y = p1_x, p1_y
        pp_x, pp_y = p2_x, p2_y
    else:
        rp_x, rp_y = p2_x, p2_y
        pp_x, pp_y = p1_x, p1_y
    r = random.randint(0, 1000)
    df = pd.read_hdf(h5)
    # flatten multi-index df
    new_cols = np.zeros_like(df.columns.values)
    for i, col in enumerate(df.columns.values):
        new_cols[i] = '_'.join(df.columns.values[i][1:])
    df.columns = new_cols

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
    
    ind1_y = df.ind1_body_right_top_y
    ind1_y[np.isnan(ind1_y)] = df.ind1_body_right_bottom_y[np.isnan(ind1_y)]
    ind1_y[np.isnan(ind1_y)] = df.ind1_neck_base_y[np.isnan(ind1_y)]
    ind1_y[np.isnan(ind1_y)] = df.ind1_tail_base_y[np.isnan(ind1_y)]
    ind1_y2sample = ind1_y[~np.isnan(ind1_y)]
    random.seed(r)
    ind1_y[np.isnan(ind1_y)] = (
        random.choices(ind1_y2sample.values, k=len(np.where(np.isnan(ind1_y))[0])))

    ind1_dist2rp = np.sqrt(((ind1_x - rp_x) ** 2) + ((ind1_y - rp_y) ** 2))
    ind1_rp_locs = np.where(ind1_dist2rp < pix_radius_thresh)[0]
    ind1_dist2pp = np.sqrt(((ind1_x - pp_x) ** 2) + ((ind1_y - pp_y) ** 2))
    ind1_pp_locs = np.where(ind1_dist2pp < pix_radius_thresh)[0]
    ind1_np_locs = np.setdiff1d(df.index.values,
                                np.concatenate((ind1_rp_locs, ind1_pp_locs)))

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

    ind2_y = df.ind2_body_right_top_y
    ind2_y[np.isnan(ind2_y)] = df.ind2_body_right_bottom_y[np.isnan(ind2_y)]
    ind2_y[np.isnan(ind2_y)] = df.ind2_neck_base_y[np.isnan(ind2_y)]
    ind2_y[np.isnan(ind2_y)] = df.ind2_tail_base_y[np.isnan(ind2_y)]
    ind2_y2sample = ind2_y[~np.isnan(ind2_y)]
    random.seed(r)
    ind2_y[np.isnan(ind2_y)] = (
        random.choices(ind2_y2sample.values, k=len(np.where(np.isnan(ind2_y))[0])))

    ind2_dist2rp = np.sqrt(((ind2_x - rp_x) ** 2) + ((ind2_y - rp_y) ** 2))
    ind2_rp_locs = np.where(ind2_dist2rp < pix_radius_thresh)[0]
    ind2_dist2pp = np.sqrt(((ind2_x - pp_x) ** 2) + ((ind2_y - pp_y) ** 2))
    ind2_pp_locs = np.where(ind2_dist2pp < pix_radius_thresh)[0]
    ind2_np_locs = np.setdiff1d(df.index.values,
                                np.concatenate((ind2_rp_locs, ind2_pp_locs)))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.histplot(x=ind1_x, y=ind1_y, ax=ax[0], stat='percent', bins=30, cbar=True,
                 cmap='autumn')
    sns.histplot(x=ind2_x, y=ind2_y, ax=ax[1], stat='percent', bins=30, cbar=True,
                 cmap='summer')
    fig.suptitle(ids + f" rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} pel; "
                       f"thresh_diff: {p_diff}")
    fig.savefig(f"/nfs/nhome/live/jbhagat/dlc_playground/de_lab_meeting"
                f"/paired_occ_histmaps"
                f"/{ids}.png")

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

    # Generate synthetic null distribution for each combo of occupancy
    rp_rp_null_dist = np.outer(occ1[:, 0], occ2[:, 0])
    rp_rp_null_dist = random.choices(rp_rp_null_dist.flatten(), k=100)
    rp_pp_null_dist = np.outer(occ1[:, 0], occ2[:, 1])
    rp_pp_null_dist = random.choices(rp_pp_null_dist.flatten(), k=100)
    rp_np_null_dist = np.outer(occ1[:, 0], occ2[:, 2])
    rp_np_null_dist = random.choices(rp_np_null_dist.flatten(), k=100)
    pp_rp_null_dist = np.outer(occ1[:, 1], occ2[:, 0])
    pp_rp_null_dist = random.choices(pp_rp_null_dist.flatten(), k=100)
    pp_pp_null_dist = np.outer(occ1[:, 1], occ2[:, 1])
    pp_pp_null_dist = random.choices(pp_pp_null_dist.flatten(), k=100)
    pp_np_null_dist = np.outer(occ1[:, 1], occ2[:, 2])
    pp_np_null_dist = random.choices(pp_np_null_dist.flatten(), k=100)
    np_rp_null_dist = np.outer(occ1[:, 2], occ2[:, 0])
    np_rp_null_dist = random.choices(np_rp_null_dist.flatten(), k=100)
    np_pp_null_dist = np.outer(occ1[:, 2], occ2[:, 1])
    np_pp_null_dist = random.choices(np_pp_null_dist.flatten(), k=100)
    np_np_null_dist = np.outer(occ1[:, 2], occ2[:, 2])
    np_np_null_dist = random.choices(np_np_null_dist.flatten(), k=100)
    all_dists = np.array((
        (rp_rp_null_dist),
        (rp_pp_null_dist),
        (rp_np_null_dist),
        (pp_rp_null_dist),
        (pp_pp_null_dist),
        (pp_np_null_dist),
        (np_rp_null_dist),
        (np_pp_null_dist),
        (np_np_null_dist),
    ))
    all_dists = all_dists.transpose()
    titles = ['rp-rp', 'rp-pp', 'rp-np', 'pp-rp', 'pp-pp', 'pp-np', 'np-rp', 'np-pp',
              'np-np']
    # 1-d dist plots
    fig, axs = plt.subplots(nrows=3, ncols=3)
    for occ_idx in range(9):
        ax = axs.flatten()[occ_idx]
        tru_val = comb_pos.flatten()[occ_idx].round(4)
        sns.histplot(all_dists[:, occ_idx], bins=25, kde=True, ax=ax, fill=True)
        vh = ax.axvline(x=tru_val, ymax=ax.get_ylim()[1], color='m')
        ax.set_title(titles[occ_idx])
        ax.set_ylabel('')
        # ax.legend([vh], ['syn_null_dist', f'actual={tru_val}'])
    axs[2, 0].set_xlabel('P[x]')
    axs[2, 0].set_ylabel('Count')
    fig.suptitle(ids + f" rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} pel; "
                       f"thresh_diff: {p_diff}")
    fig.savefig(f"/nfs/nhome/live/jbhagat/dlc_playground/de_lab_meeting/1d_occ_dists"
                f"/{ids}.png")
    fig.show()
    all_dists_df = pd.DataFrame(all_dists)
    all_dists_df.columns = titles
    emp_vs_syn = pd.Series()
    all_dists_df['emp_vs_syn'] = 'syn'
    # add empirical data
    all_dists_df = all_dists_df.append(pd.Series(comb_pos.flatten(), index=titles),
                                       ignore_index=True)
    all_dists_df.emp_vs_syn.iloc[-1] = 'emp'
    fig_pp = sns.pairplot(all_dists_df, hue="emp_vs_syn")
    fig_pp.fig.suptitle(ids + f" rp={rp_id}: {rp_pel} pel; pp={pp_id}: {pp_pel} pel; "
                              f"thresh_diff: {p_diff}")
    fig_pp.savefig(f"/nfs/nhome/live/jbhagat/dlc_playground/de_lab_meeting"
                   f"/sns_pairplots/{ids}.png")
    

    len(np.where(np.logical_and(np.logical_and(np.isnan(df.ind1_body_right_bottom_x),
                                               np.isnan(df.ind1_body_right_top_x)),
                                np.logical_and(np.isnan(df.ind1_neck_base_x),
                                               np.isnan(df.ind1_tail_base_x))))[0])
#     find animal ids corresponding to social sessions
#         for zipped list of occupancy hists with those animal ids
#             uniformly randomly sample each list
#             combine them to get 3x3 "null distribution"
#             randomly sample 9 values 100 times, normalize, plot on axes