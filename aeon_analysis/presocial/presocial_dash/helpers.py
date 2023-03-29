import numpy as np
import pandas as pd


def prettify_ele(ele):
    """Prettifies elements in a DataFrame - to be called with `applymap()`"""
    if isinstance(ele, (list, np.ndarray)):
        return f"Array (Length: {len(ele)})"
    if isinstance(ele, (pd.Timestamp, pd.Timedelta)):
        return str(ele)
    return ele
