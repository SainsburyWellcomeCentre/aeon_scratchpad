""" Plots some stats on foraging bouts.
Things to plot per mouse:
    - experiment duration
    - total number of bouts and bouts per day
    - pellets per bout distribution
    - inter-bout-interval distribution
    - time of bout distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import aeon.io.api as api
from aeon.schema.dataset import exp02, exp01
from aeon.analysis.utils import visits, distancetravelled

BOUT_DUR = pd.Timedelta("2 min")
root = '/ceph/aeon/aeon/data/raw/AEON3/presocial0.1'
events = api.load(root, exp02.ExperimentalMetadata.SubjectState)
sessions = events[events.id.str.startswith('BAA')]
if len(sessions) % 2 != 0:
    sessions = sessions.drop(sessions.index[-1])
sessions = visits(sessions)
sessions = sessions[sessions.duration > pd.Timedelta("1 day")]
data = pd.DataFrame(columns=(
    "id", "enter", "exit", "duration", "n_bouts", "pel_per_bout", "bout_start", "ibi",
))

for i, s in enumerate(sessions.itertuples()):
    id, enter, exit, duration = s.id, s.enter, s.exit, s.duration
    pellets1 = api.load(root, exp02.Patch1.DeliverPellet, start=enter, end=exit)
    pellets2 = api.load(root, exp02.Patch2.DeliverPellet, start=enter, end=exit)
    pstate1 = api.load(root, exp02.Patch1.DepletionState, start=enter, end=exit)
    pstate2 = api.load(root, exp02.Patch2.DepletionState, start=enter, end=exit)
    thresholds = pd.concat((pstate1.threshold, pstate2.threshold))
    if np.any(thresholds > 400):
        print(f"max patch thresh: {np.max(thresholds)}. skipping {s}")
        continue
    pellets = pd.concat((pellets1, pellets2))
    pellets.sort_index(inplace=True)
    t_diff = pellets.index.to_series().diff()
    bout_indxs = np.where(t_diff > BOUT_DUR)[0]
    n_bouts = len(bout_indxs)
    bout_start = t_diff[bout_indxs].index
    ibi = pd.to_timedelta(bout_start.to_series().diff().values[1:])
    pel_per_bout = np.insert(np.diff(bout_indxs), 0, bout_indxs[0])
    for col in data.columns:
        data.loc[i, col] = eval(col)
data.set_index(np.arange(len(data)), inplace=True)

# Plot experiment duration in hours
sns.set_theme(style="whitegrid")

v_fn = np.vectorize(lambda x: x.total_seconds() / 3600)
hours = v_fn(data.duration.values)
ax_exp_dur = sns.barplot(x=data.id, y=hours)
ax_exp_dur.set_title("Hours in Arena")
ax_exp_dur.set_ylabel("hours")
ax_exp_dur.set_xlabel("mouse")
ax_exp_dur.set_xticklabels(ax_exp_dur.get_xticklabels(), rotation=-15)

# Plot total number of bouts and bouts per day
bouts_per_day = data.n_bouts / hours * 24
bouts_df = pd.concat((data.n_bouts, bouts_per_day))
hue = (6 * ["n_bouts"]) + (6 * ["bouts_per_day"])
ax_bouts = sns.barplot(x=bouts_df.index, y=bouts_df.values, hue=hue)
ax_bouts.set_title("Total and per-day Number of Feeding Bouts")
ax_bouts.set_ylabel("bouts")
ax_bouts.set_xticklabels(data.id, rotation=-15)
ax_bouts.set_yticks(np.arange(start=0, stop=181, step=10))

# Plot pellets per bout distribution
ax_pel_per_bout = sns.violinplot(data=data.pel_per_bout, scale="count")
pel_df = data.pel_per_bout.explode()
sns.stripplot(
    x=pel_df.index[:-1], y=pel_df.values[pel_df.values < 50], jitter=True, zorder=1, 
    palette=["0.6"] * len(data), ax=ax_pel_per_bout
)
ax_pel_per_bout.set_title("Pellets per Feeding Bout")
ax_pel_per_bout.set_ylabel("pellets")
ax_pel_per_bout.set_xticklabels(data.id, rotation=-15)
ax_pel_per_bout.set_yticks(np.arange(start=0, stop=51, step=10))

# Plot inter-bout-interval distribution
ax_ibi = sns.violinplot(data=(data.ibi / 1e9 / 60), scale="count")
ibi_df = data.ibi.explode()
sns.stripplot(
    x=ibi_df.index, y=(ibi_df.values.astype(int) / 1e9 / 60), jitter=True, zorder=1, 
    palette=["0.6"] * len(data), ax=ax_ibi
)
ax_ibi.set_title("Feeding Inter-bout Intervals")
ax_ibi.set_ylabel("Time (minutes)")
ax_ibi.set_xticklabels(data.id, rotation=-15)

# Plot time of bout distribution
v_fn = lambda x: x.hour + (x.minute / 60)
bout_starts = [v_fn(x) for x in data.bout_start.values]
ax_bout_starts = sns.violinplot(data=bout_starts, scale="count")
bout_starts_s = pd.Series(bout_starts).explode()
sns.stripplot(
    x=bout_starts_s.index, y=bout_starts_s.values, jitter=True, zorder=1, 
    palette=["0.6"] * len(data), ax=ax_bout_starts
)
ax_bout_starts.set_title("Start Times of Feeding Bouts")
ax_bout_starts.set_ylabel("hour of the day")
ax_bout_starts.set_xticklabels(data.id, rotation=-15)
ax_bout_starts.set_ylim([0, 24])
