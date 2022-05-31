"""
Generates plots for social behavior analysis
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


# Plots side-by-side 2-d occupancy histogram heatmap
def gen_2d_occ_hist(df, bins='auto', cmap=None):
    """
    Args:
        df (DataFrame): columns are names of subjects, each containing a 2d (x,y) array
        bins (int OR vector OR str): bins passed to `np.histogram_bin_edges()`
        cmap (list of str): matplotlib colormap to use
    Returns:
        fig (Figure): handle to pyplot Figure.
    """

    n_subjects = df.shape[-1]
    fig, axs = plt.subplots(nrows=1, ncols=n_subjects)
    for i in n_subjects:
        x, y = df.iloc[:, i]
        try:
            sns.histplot(x=ind1_x, y=ind1_y, ax=axs[i], stat='percent', bins=bins, 
                         cbar=True, cmap=cmap[i])
        except:
            sns.histplot(x=ind1_x, y=ind1_y, ax=axs[i], stat='percent', bins=bins,
                         cbar=True)
    return fig
        
            

