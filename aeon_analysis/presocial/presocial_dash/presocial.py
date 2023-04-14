"""Creates a Dash dashboard for the presocial sessions"""

from pathlib import Path
from itertools import product

import dash
import dash_daq as daq
import ipdb
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.development.base_component import ComponentRegistry
from dotmap import DotMap

from aeon_analysis.presocial.presocial_dash import helpers


# Load presocial session data
df = pd.read_pickle(
    Path(
        "/nfs/nhome/live/jbhagat/ProjectAeon/aeon_analysis/aeon_analysis/presocial/data"
        "/presocial_data.pkl"
    )
)

# Set some constants relating to the initialized colors/plotting
bg_col = "#050505"
txt_col = "#f2f2f2"
plt_bg_col = "#0d0d0d"
tab_bg_col = "#003399"
tab_txt_col = "#f2f2f2"
table_max_height = "400px"
table_min_width = "1200px"
mrkr_sz = 14
color_dict = {
    "BAA-1103045": "rgb(31, 119, 180)",
    "BAA-1103047": "rgb(214, 39, 40)",
    "BAA-1103048": "rgb(44, 160, 44)",
    "BAA-1103049": "rgb(148, 103, 189)",
    "BAA-1103050": "rgb(255, 127, 14)",
}
# Set all relevant app.layout children names (for future color theme updates)
fig_names = [
    "weight_enter_session",
    "weight_diff_session",
    "weight_enter_subject",
    "weight_diff_subject",
    "duration_session",
    "post_thresh_dur_session",
    "pre_sampling_both_p_dur_session",
    "duration_subject",
    "post_thresh_dur_subject",
    "pre_sampling_both_p_dur_subject",
    "hard_patch_session",
    "hard_patch_subject",
    "wheel_session_abs",
    "wheel_session_norm",
    "wheel_subject_abs",
    "wheel_subject_norm",
    "pellet_session_abs",
    "pellet_session_norm",
    "pellet_subject_abs",
    "pellet_subject_norm",
    "prob_pels_session",
    "prob_pels_subject",
    "patch_pref_epoch_session",
    "cum_patch_pref_epoch_session",
]
tab_names = []

# Create the plots for the 6 divisions that will be visualized in the dashboard:

# 1. Data Table
pd.options.mode.chained_assignment = None
df_pretty = df.applymap(helpers.prettify_ele)  # prettify for table viz
data_table = dash_table.DataTable(
    id="data_table",
    data=df_pretty.to_dict("records"),
    columns=[{"name": i, "id": i} for i in df.columns],
    style_table={
        "overflowX": "auto",
        "overflowY": "auto",
        "maxHeight": table_max_height,
        "minWidth": table_min_width,
    },
    fixed_columns={"headers": True, "data": 2},
    fixed_rows={"headers": True},
    style_header={"fontWeight": "bold", "backgroundColor": plt_bg_col},
    style_cell={
        "backgroundColor": plt_bg_col,
        "color": txt_col,
        "textAlign": "left",
        "whiteSpace": "normal",
        "height": "auto",
        "minWidth": 190,
    },
)
pd.options.mode.chained_assignment = "warn"


# 2. Weight Viz
df = df.sort_values(by=["id", "enter"])
weight_diff = df["weight_exit"] - df["weight_enter"]
weight_enter_session = px.line(
    df,
    x="enter",
    y="weight_enter",
    color="id",
    markers=True,
    title="Weight at Entry by Session",
)
weight_enter_session.update_yaxes(title="weight (g)")
weight_enter_session.update_traces(marker={"size": mrkr_sz})

weight_diff_session = px.line(
    df,
    x="enter",
    y=weight_diff,
    color="id",
    markers=True,
    title="Weight Diff by Session",
)
weight_diff_session.update_yaxes(title="weight (g)")
weight_diff_session.update_traces(marker={"size": mrkr_sz})

weight_enter_subject = px.histogram(
    df,
    x="weight_enter",
    color="id",
    # histnorm="probability density",
    title="Weight Enter by Subject",
)
weight_enter_subject.update_xaxes(title="weight (g)")

weight_diff_subject = px.histogram(
    df,
    x=weight_diff,
    color="id",
    # histnorm="probability density",
    title="Weight Diff by Subject",
)
weight_diff_subject.update_xaxes(title="weight (g)")

# 3. Time Viz
duration_session = px.bar(
    df,
    x="enter",
    y=pd.to_timedelta(df["duration"]).dt.total_seconds() / 60,
    color="id",
    title="Session Duration by Session",
)
# duration_session.update_traces(marker={"size": mrkr_sz})
duration_session.update_yaxes(title="duration (mins)")

post_thresh_dur_session = px.bar(
    df,
    x="enter",
    y=pd.to_timedelta(df["post_thresh_dur"]).dt.total_seconds() / 60,
    color="id",
    title="Post-Threshold Duration by Session",
)
# post_thresh_dur_session.update_traces(marker={"size": mrkr_sz})
post_thresh_dur_session.update_yaxes(title="duration (mins)")

pre_sampling_both_p_dur_session = px.bar(
    df,
    x="enter",
    y=pd.to_timedelta(df["pre_sampling_both_p_dur"]).dt.total_seconds() / 60,
    color="id",
    title="Pre-Sampling Both Patches Duration by Session",
)
# pre_sampling_both_p_dur_session.update_traces(marker={"size": mrkr_sz})
pre_sampling_both_p_dur_session.update_yaxes(title="duration (mins)")

duration_subject = px.histogram(
    df,
    x=pd.to_timedelta(df["duration"]).dt.total_seconds() / 60,
    color="id",
    # histnorm="density",
    title="Session Duration by Subject",
)
duration_subject.update_xaxes(title="duration (mins)")

post_thresh_dur_subject = px.histogram(
    df,
    x=pd.to_timedelta(df["post_thresh_dur"]).dt.total_seconds() / 60,
    color="id",
    # histnorm="density",
    title="Post-Threshold Duration by Subject",
)
post_thresh_dur_subject.update_xaxes(title="duration (mins)")
pre_sampling_both_p_dur_subject = px.histogram(
    df,
    x=pd.to_timedelta(df["pre_sampling_both_p_dur"]).dt.total_seconds() / 60,
    color="id",
    # histnorm="density",
    title="Pre-Sampling Both Patches Duration by Subject",
)
pre_sampling_both_p_dur_subject.update_xaxes(title="duration (mins)")

# 4. Hard Patch Info
hard_patch_ct_subj = df.groupby(["id", "hard_patch"]).size().reset_index(name="count")
hard_patch_ct_patch = df.groupby("hard_patch").size().reset_index(name="count")
hard_patch_session = px.bar(
    df,
    x="enter",
    y="hard_patch",
    color="id",
    title="Hard Patch by Session",
)
hard_patch_subject = px.histogram(
    df,
    x="hard_patch",
    color="id",
    # histnorm="density",
    title="Hard Patch by Subject",
)

# 5. Wheel Viz
patch_pref_epoch_session = go.Figure()
sesh_subj_counter = DotMap(
    {
        "BAA-1103045": 0,
        "BAA-1103047": 0,
        "BAA-1103048": 0,
        "BAA-1103049": 0,
        "BAA-1103050": 0,
    }
)
for i in df.index:
    uid = df["id"][i]
    y = df["easy_pref_epoch"][i]
    sesh_subj_counter[uid] += 1
    patch_pref_epoch_session.add_trace(
        go.Scatter(
            y=y,
            name=f"{uid}: {sesh_subj_counter[uid]}",
            mode="lines+markers",
            marker={"size": mrkr_sz},
            line=dict(color=color_dict[uid]),
        )
    )
    zidx = int(df["epoch_thresh_change_idx"][i])
    patch_pref_epoch_session.add_trace(
        go.Scatter(
            x=np.array((zidx, zidx + 1)),
            y=y[zidx : zidx + 2],
            mode="lines+markers",
            marker={"size": mrkr_sz},
            line=dict(color="black"),
            name=f"{uid}: {sesh_subj_counter[uid]}: thresh change",
        )
    )
patch_pref_epoch_session.update_layout(
    title="Patch Preference by Distance Quantile within Session",
    xaxis_title="Distance Quantile",
    yaxis_title="Easy Patch Preference",
    legend_title="Session",
)

cum_patch_pref_epoch_session = go.Figure()
sesh_subj_counter = DotMap(
    {
        "BAA-1103045": 0,
        "BAA-1103047": 0,
        "BAA-1103048": 0,
        "BAA-1103049": 0,
        "BAA-1103050": 0,
    }
)
for i in df.index:
    uid = df["id"][i]
    y = df["easy_pref_epoch_cum"][i]
    sesh_subj_counter[uid] += 1
    cum_patch_pref_epoch_session.add_trace(
        go.Scatter(
            y=y,
            name=f"{uid}: {sesh_subj_counter[uid]}",
            mode="lines+markers",
            marker={"size": mrkr_sz},
            line=dict(color=color_dict[uid]),
        )
    )
    zidx = int(df["epoch_thresh_change_idx"][i])
    cum_patch_pref_epoch_session.add_trace(
        go.Scatter(
            x=np.array((zidx, zidx + 1)),
            y=y[zidx : zidx + 2],
            mode="lines+markers",
            marker={"size": mrkr_sz},
            line=dict(color="black"),
            name=f"{uid}: {sesh_subj_counter[uid]}: thresh change",
        )
    )
cum_patch_pref_epoch_session.update_layout(
    title="Cumulative Patch Preference by Distance Quantile within Session",
    xaxis_title="Distance Quantile",
    yaxis_title="Easy Patch Preference",
    legend_title="Session",
)

wheel_session_abs = go.Figure()
df["tot_hard_wheel"] = df["pre_hard_wheel_dist"] + df["post_hard_wheel_dist"]
df["tot_easy_wheel"] = df["pre_easy_wheel_dist"] + df["pre_hard_wheel_dist"]
df["tot_pre_wheel"] = df["pre_hard_wheel_dist"] + df["pre_easy_wheel_dist"]
df["tot_post_wheel"] = df["post_hard_wheel_dist"] + df["post_easy_wheel_dist"]
df["tot_wheel"] = (
    df["tot_hard_wheel"]
    + df["tot_easy_wheel"]
    + df["tot_pre_wheel"]
    + df["tot_post_wheel"]
)
uids = np.sort(df["id"].unique())
markers = [
    "circle",
    "square",
    "diamond",
    "cross",
    "star",
    "triangle-up",
    "triangle-left",
    "triangle-right",
    "triangle-down",
]
cols = [
    "pre_easy_wheel_dist",
    "pre_hard_wheel_dist",
    "post_easy_wheel_dist",
    "post_hard_wheel_dist",
    "tot_hard_wheel",
    "tot_easy_wheel",
    "tot_pre_wheel",
    "tot_post_wheel",
    "tot_wheel",
]
for idx, col in enumerate(cols):
    for uid in uids:
        wheel_session_abs.add_trace(
            go.Scatter(
                x=df[df["id"] == uid]["enter"],
                y=df[df["id"] == uid][col],
                mode="lines+markers",
                marker={"symbol": markers[idx], "size": mrkr_sz},
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                line=dict(color=color_dict[uid]),
            )
        )
wheel_session_abs.update_layout(
    title="Wheel Distance Spun (Absolute) by Session",
    xaxis_title="Enter",
    yaxis_title="Distance Spun (cm)",
    legend_title="Wheel Distances",
)

wheel_session_norm = go.Figure()
df["pre_pref"] = df["tot_pre_wheel"] / (df["tot_pre_wheel"] + df["tot_post_wheel"])
df["post_pref"] = 1 - df["pre_pref"]
df["easy_pref"] = df["tot_easy_wheel"] / (df["tot_easy_wheel"] + df["tot_hard_wheel"])
df["hard_pref"] = 1 - df["easy_pref"]
cols = [
    "pre_easy_pref",
    "pre_hard_pref",
    "post_easy_pref",
    "post_hard_pref",
    "pre_pref",
    "post_pref",
    "hard_pref",
    "easy_pref",
]
for idx, col in enumerate(cols):
    for uid in uids:
        wheel_session_norm.add_trace(
            go.Scatter(
                x=df[df["id"] == uid]["enter"],
                y=df[df["id"] == uid][col],
                mode="lines+markers",
                marker={"symbol": markers[idx], "size": mrkr_sz},
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                line=dict(color=color_dict[uid]),
            )
        )
wheel_session_norm.update_layout(
    title="Wheel Distance Spun (Normalized) by Session",
    xaxis_title="Enter",
    yaxis_title="Distance Spun (a.u. 0-1)",
    legend_title="Wheel Distances",
)

wheel_subject_abs = go.Figure()
cols = [
    "pre_easy_wheel_dist",
    "pre_hard_wheel_dist",
    "post_easy_wheel_dist",
    "post_hard_wheel_dist",
    "tot_hard_wheel",
    "tot_easy_wheel",
    "tot_pre_wheel",
    "tot_post_wheel",
    "tot_wheel",
]
x_positions = []
for j, uid in enumerate(uids):
    for i, col in enumerate(cols):
        xpos = i + j * (len(cols) + 0.5)
        x_positions.append(xpos)
        y = df[df["id"] == uid][col].tolist()
        wheel_subject_abs.add_trace(
            go.Box(
                y=y,
                x=[xpos] * len(y),
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                boxpoints="all",
                pointpos=-1.5,
                jitter=0.1,
                line=dict(color=color_dict[uid]),
            )
        )
xlabels = [" ".join(combo) for combo in list(product(uids, cols))]
wheel_subject_abs.update_layout(
    title="Wheel Distance Spun (Absolute) by Subject",
    xaxis=dict(
        tickmode="array",
        tickvals=x_positions,
        ticktext=xlabels,
    ),
    yaxis_title="Distance Spun (cm)",
    legend_title="Wheel Distances",
)

wheel_subject_norm = go.Figure()
cols = [
    "pre_easy_pref",
    "pre_hard_pref",
    "post_easy_pref",
    "post_hard_pref",
    "pre_pref",
    "post_pref",
    "hard_pref",
    "easy_pref",
]
x_positions = []
for j, uid in enumerate(uids):
    for i, col in enumerate(cols):
        xpos = i + j * (len(cols) + 0.5)
        x_positions.append(xpos)
        y = df[df["id"] == uid][col].tolist()
        wheel_subject_norm.add_trace(
            go.Box(
                y=y,
                x=[xpos] * len(y),
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                boxpoints="all",
                pointpos=-1.5,
                jitter=0.1,
                line=dict(color=color_dict[uid]),
            )
        )
wheel_subject_norm.update_layout(
    title="Wheel Distance Spun (Normalized) by Subject",
    xaxis=dict(
        tickmode="array",
        tickvals=x_positions,
        ticktext=xlabels,
    ),
    yaxis_title="Distance Spun (a.u. 0-1)",
    legend_title="Wheel Distances",
)

# 6. Pellet Viz
pellet_session_abs = go.Figure()
df["tot_hard_n_pel"] = df["pre_hard_n_pel"] + df["post_hard_n_pel"]
df["tot_easy_n_pel"] = df["pre_easy_n_pel"] + df["pre_hard_n_pel"]
df["tot_pre_n_pel"] = df["pre_hard_n_pel"] + df["pre_easy_n_pel"]
df["tot_post_n_pel"] = df["post_hard_n_pel"] + df["post_easy_n_pel"]
df["tot_n_pel"] = df["tot_hard_n_pel"] + df["tot_easy_n_pel"]
cols = [
    "pre_easy_n_pel",
    "pre_hard_n_pel",
    "post_easy_n_pel",
    "post_hard_n_pel",
    "tot_hard_n_pel",
    "tot_easy_n_pel",
    "tot_pre_n_pel",
    "tot_post_n_pel",
    "tot_n_pel",
]
for idx, col in enumerate(cols):
    for uid in uids:
        pellet_session_abs.add_trace(
            go.Scatter(
                x=df[df["id"] == uid]["enter"],
                y=df[df["id"] == uid][col],
                mode="lines+markers",
                marker={"symbol": markers[idx], "size": mrkr_sz},
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                line=dict(color=color_dict[uid]),
            )
        )
pellet_session_abs.update_layout(
    title="Pellets by Session",
    xaxis_title="Enter",
    yaxis_title="Count",
    legend_title="Divisions",
)

pellet_session_norm = go.Figure()
df["pre_easy_pel_pref"] = df["pre_easy_n_pel"] / (
    df["pre_easy_n_pel"] + df["pre_hard_n_pel"]
)
df["pre_hard_pel_pref"] = 1 - df["pre_easy_pel_pref"]
df["post_easy_pel_pref"] = df["post_easy_n_pel"] / (
    df["post_easy_n_pel"] + df["post_hard_n_pel"]
)
df["post_hard_pel_pref"] = 1 - df["post_easy_pel_pref"]
df["pre_pel_pref"] = df["tot_pre_n_pel"] / (df["tot_pre_n_pel"] + df["tot_post_n_pel"])
df["post_pel_pref"] = 1 - df["pre_pel_pref"]
df["easy_pel_pref"] = df["tot_easy_n_pel"] / (
    df["tot_easy_n_pel"] + df["tot_hard_n_pel"]
)
df["hard_pel_pref"] = 1 - df["easy_pel_pref"]
cols = [
    "pre_easy_pel_pref",
    "pre_hard_pel_pref",
    "post_easy_pel_pref",
    "post_hard_pel_pref",
    "pre_pel_pref",
    "post_pel_pref",
    "hard_pel_pref",
    "easy_pel_pref",
]
for idx, col in enumerate(cols):
    for uid in uids:
        pellet_session_norm.add_trace(
            go.Scatter(
                x=df[df["id"] == uid]["enter"],
                y=df[df["id"] == uid][col],
                mode="lines+markers",
                marker={"symbol": markers[idx], "size": mrkr_sz},
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                line=dict(color=color_dict[uid]),
            )
        )
pellet_session_norm.update_layout(
    title="Pellets (Normalized) by Session",
    xaxis_title="Enter",
    yaxis_title="Pellets (a.u. 0-1)",
    legend_title="Divisions",
)

pellet_subject_abs = go.Figure()
cols = [
    "pre_easy_n_pel",
    "pre_hard_n_pel",
    "post_easy_n_pel",
    "post_hard_n_pel",
    "tot_hard_n_pel",
    "tot_easy_n_pel",
    "tot_pre_n_pel",
    "tot_post_n_pel",
    "tot_n_pel",
]
x_positions = []
for j, uid in enumerate(uids):
    for i, col in enumerate(cols):
        xpos = i + j * (len(cols) + 0.5)
        x_positions.append(xpos)
        y = df[df["id"] == uid][col].tolist()
        pellet_subject_abs.add_trace(
            go.Box(
                y=y,
                x=[xpos] * len(y),
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                boxpoints="all",
                pointpos=-1.5,
                jitter=0.1,
                line=dict(color=color_dict[uid]),
            )
        )
pellet_subject_abs.update_layout(
    title="Pellets by Subject",
    xaxis=dict(
        tickmode="array",
        tickvals=x_positions,
        ticktext=xlabels,
    ),
    yaxis_title="Pellets",
    legend_title="Divisions",
)

pellet_subject_norm = go.Figure()
cols = [
    "pre_easy_pel_pref",
    "pre_hard_pel_pref",
    "post_easy_pel_pref",
    "post_hard_pel_pref",
    "pre_pel_pref",
    "post_pel_pref",
    "hard_pel_pref",
    "easy_pel_pref",
]
x_positions = []
for j, uid in enumerate(uids):
    for i, col in enumerate(cols):
        xpos = i + j * (len(cols) + 0.5)
        x_positions.append(xpos)
        y = df[df["id"] == uid][col].tolist()
        pellet_subject_norm.add_trace(
            go.Box(
                y=y,
                x=[xpos] * len(y),
                name=f"{col}",
                legendgroup=col,
                showlegend=(uid == uids[0]),
                boxpoints="all",
                pointpos=-1.5,
                jitter=0.1,
                line=dict(color=color_dict[uid]),
            )
        )
pellet_subject_norm.update_layout(
    title="Pellets by Subject",
    xaxis=dict(
        tickmode="array",
        tickvals=x_positions,
        ticktext=xlabels,
    ),
    yaxis_title="Pellets (a.u. 0-1)",
    legend_title="Divisions",
)

# Probabilistic Pellets (Prob Pels Viz by Session and Prob Pels Viz by Subject)
prob_pels_session = go.Figure()
cols = ["post_easy_pel_thresh", "post_hard_pel_thresh"]
col_idxs = ["post_easy_pel_thresh_idx", "post_hard_pel_thresh_idx"]
for uid in uids:
    for idx, (idx_col, col) in enumerate(zip(col_idxs, cols)):
        if not len(df[df["id"] == uid][col]):  # skip empty
            continue
        if len(df[df["id"] == uid][col]) > 1:
            y = np.concatenate(df[df["id"] == uid][col].tolist())
            x = np.concatenate(df[df["id"] == uid][idx_col].tolist())
        else:
            y = df[df["id"] == uid][col].values[0]
            x = df[df["id"] == uid][idx_col].values[0]
        prob_pels_session.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                marker={"symbol": markers[idx], "size": mrkr_sz},
                name=f"{uid}: {col}",
                # legendgroup=col,
                # showlegend=(uid == uids[0]),
                line=dict(color=color_dict[uid]),
            )
        )
# for uid in uids:
#     # dummy trace for `uid` in legend
#     prob_pels_session.add_trace(
#         go.Scatter(
#             x=[None],
#             y=[None],
#             mode='none',
#             showlegend=True,
#             name=uid,
#             legendgroup=uid,
#             line=dict(color=color_dict[uid])
#         )
#     )
prob_pels_session.update_layout(
    title="Threshold Values During Probabilistic Period",
    xaxis_title="Datetime",
    yaxis_title="Distance (cm)",
    legend_title="Divisions",
)

prob_pels_subject = go.Figure()
cols = ["post_easy_pel_thresh", "post_hard_pel_thresh"]
col_idxs = ["post_easy_pel_thresh_idx", "post_hard_pel_thresh_idx"]
x_positions = []
for j, uid in enumerate(uids):
    for i, (col_idx, col) in enumerate(zip(col_idxs, cols)):
        if not len(df[df["id"] == uid][col]):  # skip empty
            continue
        if len(df[df["id"] == uid][col]) > 1:
            y = np.concatenate(df[df["id"] == uid][col].tolist())
        else:
            y = df[df["id"] == uid][col].values[0]
        xpos = i + j * (len(cols) + 0.5)
        x_positions.append(xpos)
        prob_pels_subject.add_trace(
            go.Box(
                y=y,
                x=[xpos] * len(y),
                name=f"{uid} {col}",
                boxpoints="all",
                pointpos=-1.5,
                jitter=0.1,
                line=dict(color=color_dict[uid]),
            )
        )
xlabels = [" ".join(combo) for combo in list(product(uids, cols))]
prob_pels_subject.update_layout(
    title="Threshold Values Distributions During Probabilistic Period",
    xaxis=dict(
        tickmode="array",
        tickvals=x_positions,
        ticktext=xlabels,
    ),
    yaxis_title="Distance (cm)",
    legend_title="Divisions",
)

# Update default fig colors
for fig_name in fig_names:
    eval(fig_name).update_layout(
        paper_bgcolor=bg_col,
        plot_bgcolor=plt_bg_col,
        font={"color": txt_col},
    )

# Create dashboard layout
app = Dash(__name__)
app.layout = html.Div(
    id="app",
    style={"backgroundColor": bg_col, "color": txt_col},
    children=[
        html.H1("Presocial Data Dashboard"),
        # Commenting out the Color Picker for now
        # html.Div(
        #     id="col_picker_div",
        #     children=[
        #         html.Div(
        #             id="bg_col_picker_div",
        #             children=[
        #                 daq.ColorPicker(
        #                     id="dash_bg_col_picker",
        #                     label=("Dashboard Background Color"),
        #                     value={"hex": bg_col},
        #                 ),
        #             ],
        #             style={"display": "inline-block", "margin-right": "20px"},
        #         ),
        #         html.Div(
        #             id="txt_col_picker_div",
        #             children=[
        #                 daq.ColorPicker(
        #                     id="dash_txt_col_picker",
        #                     label=("Dashboard Text Color"),
        #                     value={"hex": txt_col},
        #                 ),
        #             ],
        #             style={"display": "inline-block", "margin-right": "20px"},
        #         ),
        #         html.Div(
        #             id="plt_bg_picker_div",
        #             children=[
        #                 daq.ColorPicker(
        #                     id="plt_bg_col_picker",
        #                     label=("Plot Background Color"),
        #                     value={"hex": plt_bg_col},
        #                 ),
        #             ],
        #             style={"display": "inline-block", "margin-right": "20px"},
        #         ),
        #         html.Div(
        #             id="col_btn_div",
        #             children=[
        #                 html.Button("Update Colors", id="col_button"),
        #             ],
        #             style={"display": "flex", "justify-content": "left"},
        #         ),
        #     ],
        # ),
        html.Div(id="data_table_div", children=[html.H2("Data Table"), data_table]),
        html.Div(
            id="weight_viz_div",
            children=[
                html.H2("Weight Viz"),
                dcc.Tabs(
                    id="weight_viz_tabs",
                    children=[
                        dcc.Tab(
                            id="weight_viz_session_tab",
                            label="Weight Viz by Session",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=weight_enter_session,
                                    id="weight_enter_session",
                                ),
                                dcc.Graph(
                                    figure=weight_diff_session, id="weight_diff_session"
                                ),
                            ],
                        ),
                        dcc.Tab(
                            id="weight_viz_subject_tab",
                            label="Weight Viz by Subject",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=weight_enter_subject,
                                    id="weight_enter_subject",
                                ),
                                dcc.Graph(
                                    figure=weight_diff_subject, id="weight_diff_subject"
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="time_viz_div",
            children=[
                html.H2("Time Viz"),
                dcc.Tabs(
                    id="time_viz_tabs",
                    children=[
                        dcc.Tab(
                            id="time_viz_session_tab",
                            label="Time Viz by Session",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=duration_session, id="duration_session"
                                ),
                                dcc.Graph(
                                    figure=post_thresh_dur_session,
                                    id="post_thresh_dur_session",
                                ),
                                dcc.Graph(
                                    figure=pre_sampling_both_p_dur_session,
                                    id="pre_sampling_both_p_dur_session",
                                ),
                            ],
                        ),
                        dcc.Tab(
                            id="time_viz_subject_tab",
                            label="Time Viz by Subject",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=duration_subject, id="duration_subject"
                                ),
                                dcc.Graph(
                                    figure=post_thresh_dur_subject,
                                    id="post_thresh_dur_subject",
                                ),
                                dcc.Graph(
                                    figure=pre_sampling_both_p_dur_subject,
                                    id="pre_sampling_both_p_dur_subject",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="hard_patch_info_div",
            children=[
                html.H2("Hard Patch Info"),
                dcc.Tabs(
                    id="hard_patch_info_tabs",
                    children=[
                        dcc.Tab(
                            id="hard_patch_session_tab",
                            label="Hard Patch by Session",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=hard_patch_session, id="hard_patch_session"
                                ),
                            ],
                        ),
                        dcc.Tab(
                            id="hard_patch_subject_tab",
                            label="Hard Patch by Subject",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=hard_patch_subject, id="hard_patch_subject"
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="wheel_viz_div",
            children=[
                html.H2("Wheel Viz"),
                dcc.Tabs(
                    id="patch_pref_over_time_tabs",
                    children=[
                        dcc.Tab(
                            id="pref_over_time_tab",
                            label="Patch Preference Over Time within a Session",
                            style={
                                "backgroundColor": bg_col,
                                "color": txt_col,
                            },
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=patch_pref_epoch_session,
                                    id="patch_pref_epoch_session",
                                ),
                            ],
                        ),
                        dcc.Tab(
                            id="pref_over_time_cum_tab",
                            label="Cumulative Patch Preference Over Time within a "
                            "Session",
                            style={
                                "backgroundColor": bg_col,
                                "color": txt_col,
                            },
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Graph(
                                    figure=cum_patch_pref_epoch_session,
                                    id="cum_patch_pref_epoch_session",
                                ),
                            ],
                        ),
                    ],
                ),
                dcc.Tabs(
                    id="wheel_viz_tabs",
                    children=[
                        dcc.Tab(
                            id="wheel_viz_session_tab",
                            label="Wheel Viz by Session",
                            style={
                                "backgroundColor": bg_col,
                                "color": txt_col,
                            },
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Tabs(
                                    id="wheel_viz_session_tabs",
                                    children=[
                                        dcc.Tab(
                                            id="wheel_viz_session_abs_tab",
                                            label="Absolute",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(
                                                    figure=wheel_session_abs,
                                                    id="wheel_session_abs",
                                                ),
                                            ],
                                        ),
                                        dcc.Tab(
                                            id="wheel_viz_session_norm_tab",
                                            label="Normalized",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(
                                                    figure=wheel_session_norm,
                                                    id="wheel_session_norm",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Tab(
                            id="wheel_viz_subject_tab",
                            label="Wheel Viz by Subject",
                            style={
                                "backgroundColor": bg_col,
                                "color": txt_col,
                            },
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Tabs(
                                    id="wheel_viz_subject_tabs",
                                    children=[
                                        dcc.Tab(
                                            id="wheel_viz_subject_abs_tab",
                                            label="Absolute",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(
                                                    figure=wheel_subject_abs,
                                                    id="wheel_subject_abs",
                                                ),
                                            ],
                                        ),
                                        dcc.Tab(
                                            id="wheel_viz_subject_norm_tab",
                                            label="Normalized",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(
                                                    figure=wheel_subject_norm,
                                                    id="wheel_subject_norm",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="pellet_viz_div",
            children=[
                html.H2("Pellet Viz"),
                dcc.Tabs(
                    id="pellet_viz_tabs",
                    children=[
                        dcc.Tab(
                            id="pellet_viz_session_tab",
                            label="Pellet Viz by Session",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Tabs(
                                    id="pellet_viz_session_tabs",
                                    children=[
                                        dcc.Tab(
                                            id="pellet_session_abs_tab",
                                            label="Absolute",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(figure=pellet_session_abs)
                                            ],
                                        ),
                                        dcc.Tab(
                                            id="pellet_session_norm_tab",
                                            label="Normalized",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(figure=pellet_session_norm)
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                        dcc.Tab(
                            id="pellet_viz_subject_tab",
                            label="Pellet Viz by Subject",
                            style={"backgroundColor": bg_col, "color": txt_col},
                            selected_style={
                                "backgroundColor": tab_bg_col,
                                "color": tab_txt_col,
                            },
                            children=[
                                dcc.Tabs(
                                    id="pellet_viz_subject_tabs",
                                    children=[
                                        dcc.Tab(
                                            id="pellet_subject_abs_tab",
                                            label="Absolute",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(figure=pellet_subject_abs)
                                            ],
                                        ),
                                        dcc.Tab(
                                            id="pellet_subject_norm_tab",
                                            label="Normalized",
                                            style={
                                                "backgroundColor": bg_col,
                                                "color": txt_col,
                                            },
                                            selected_style={
                                                "backgroundColor": tab_bg_col,
                                                "color": tab_txt_col,
                                            },
                                            children=[
                                                dcc.Graph(figure=pellet_subject_norm)
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            id="prob_pellet_viz_div",
            children=[
                html.H2("Probabilistic Pellet Viz"),
                dcc.Tab(
                    id="prob_pel_tab",
                    label="Probabilistic Pellets",
                    style={"backgroundColor": bg_col, "color": txt_col},
                    selected_style={
                        "backgroundColor": tab_bg_col,
                        "color": tab_txt_col,
                    },
                    children=[
                        dcc.Tabs(
                            id="prob_pel_tabs",
                            children=[
                                dcc.Tab(
                                    id="prob_pels_session_tab",
                                    label="Probabilistic Pellets by Session",
                                    style={"backgroundColor": bg_col, "color": txt_col},
                                    selected_style={
                                        "backgroundColor": tab_bg_col,
                                        "color": tab_txt_col,
                                    },
                                    children=[
                                        dcc.Graph(
                                            figure=prob_pels_session,
                                            id="prob_pels_session",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    id="prob_pels_subject_tab",
                                    label="Probabilistic Pellets by Subject",
                                    style={"backgroundColor": bg_col, "color": txt_col},
                                    selected_style={
                                        "backgroundColor": tab_bg_col,
                                        "color": tab_txt_col,
                                    },
                                    children=[dcc.Graph(figure=prob_pels_subject)],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# @app.callback(
#     Output("prob_pels_session", "figure"),
#     Input("prob_pels_session", "relayoutData"),
#     State("prob_pels_session", "figure"),
# )
# def update_trace_visibility(relayout_data, figure):
#     if relayout_data is None:
#         raise dash.exceptions.PreventUpdate
#     if 'legendgroup' not in str(relayout_data):
#         raise dash.exceptions.PreventUpdate
#     pdb.set_trace()
#     clicked_legend = list(relayout_data.keys())[0]
#     clicked_status = relayout_data[clicked_legend]
#     clicked_uid = clicked_legend.split('.')[0]
#     new_traces = []
#     for trace in figure['data']:
#         if trace['legendgroup'].startswith(clicked_uid):
#             trace['visible'] = clicked_status
#         new_traces.append(trace)
#     figure['data'] = new_traces
#     return figure

# Commenting out potential color picker callbacks for now
# Option 1:
# Callback for updating colors
# @app.callback(
#     [
#         Output("dash_bg_col_picker", "style"),
#         Output("dash_txt_col_picker", "style"),
#         Output("plt_bg_col_picker", "style"),
#     ],
#     Input("col_button", "n_clicks"),
#     [
#         State("dash_bg_col_picker", "value"),
#         State("dash_txt_col_picker", "value"),
#         State("plt_bg_col_picker", "value"),
#     ],
# )
# def update_colors(n_clicks, bg_color, text_color, plot_bg_color):
#     # Update the styles of the layout and the plots
#     # based on the selected colors
#     return [
#         {"backgroundColor": bg_color["hex"]},
#         {"color": text_color["hex"]},
#         {"backgroundColor": plot_bg_color["hex"]},
#     ]
#
#
# Option 2:
# def find_graphs(layout):
#     graphs = []
#     if isinstance(layout, list):
#         for element in layout:
#             graphs.extend(find_graphs(element))
#     elif isinstance(layout, dict):
#         for key, value in layout.items():
#             graphs.extend(find_graphs(value))
#     elif hasattr(layout, "children"):
#         graphs.extend(find_graphs(layout.children))
#     elif isinstance(layout, dcc.Graph):
#         graphs.append(layout)
#     return graphs
#
#
# @app.callback(
#     # Output("app", "style"),
#     Output("dummy_output", "style"),
#     Input("col_button", "n_clicks"),
#     State("dash_bg_col_picker", "value"),
#     State("dash_txt_col_picker", "value"),
#     State("plt_bg_col_picker", "value"),
#     State("fig_ids_storage", "children"),
# )
# def update_colors(n_clicks, bg_col, txt_col, plt_bg_col, fig_ids):
#     if n_clicks:
#         import ipdb
#         ipdb.set_trace()
#         global app
#         graphs = find_graphs(app.layout)
#         for graph in graphs:
#             graph.figure["layout"]["paper_bgcolor"] = bg_col
#             graph.figure["layout"]["font"] = {"color": txt_col}
#             graph.figure["layout"]['plot_bgcolor'] = plt_bg_col
#         html.Script(src="assets/reload_page.js")


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="7777", debug=True, dev_tools_hot_reload=True)
