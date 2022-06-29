"""
Contains general functions for multianimal analysis and plotting.
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


# General process for writing functions:
# D: describe (can put this in the function docstring)
# S: signaturize (define inputs and outputs and create docstring)
# E: exemplify (can include examples in the docstrings OR create separate examples python file)
# C: code
# T: test (test code with examples in python console, and then use some automated testing framework e.g. pytest)


def good_ses_metadata(exp_root, start_ts, end_ts, bad_sessions=None):
    """
    Get good session metadata.

    Args:
        exp_root (str): The path to the root of the experiment data.
        start_ts (Timestamp): The initial timestamp to get data from.
        end_ts (Timestamp): The final timestamp to get data from.
        bad_sessions (array-like of Timestamps; optional): Bad sessions to exclude.

    Returns:
        session_metadata (DataFrame): A dataframe containing session metadata for user-defined good sessions.

    Examples:
        # Get good sessions from January to February 2022
        ```
        exp_root = '/nfs/winstor/delab/data/arena0.1/socialexperiment0_raw'
        session_metadata = helpers.loadSessions(exp_root)
        start_ts = pd.Timestamp('2022-01-16')
        end_ts = pd.Timestamp('2022-02-11')
        bad_ses = [pd.Timestamp('2022-01-18 12:32:00'), pd.Timestamp('2022-01-19 12:28:00'),
            pd.Timestamp('2022-01-19 14:57:00'), pd.Timestamp('2022-01-31 14:18:00')]
        ses_metadata = good_ses_metadata(exp_root, start_ts, end_ts, bad_ses)
        ```
    """
    # Get all session metadata
    session_metadata = helpers.loadSessions(exp_root)
    # Restrict all session metadata to sessions within start and end ts
    good_ses_mask = (np.logical_and(session_metadata['start'] > start_ts, session_metadata[
        'start'] < end_ts))
    session_metadata = session_metadata[good_ses_mask]
    # Exclude specified bad sessions
    if bad_sessions is not None:
        i_bad_ses = [np.argmin(np.abs(session_metadata['start'] - ts)) for ts in bad_sessions]
        session_metadata.drop(index=session_metadata.iloc[i_bad_ses].index, inplace=True)
    return session_metadata