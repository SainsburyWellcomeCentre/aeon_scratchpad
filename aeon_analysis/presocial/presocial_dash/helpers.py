import numpy as np
import pandas as pd


def prettify_ele(ele):
    """Prettifies elements in a DataFrame - to be called with `applymap()`"""
    # import ipdb; ipdb.set_trace()
    if isinstance(ele, (list, np.ndarray)):
        try:
            return f"Array (Length: {len(ele)})"
        # catch TypeError unsized object
        except TypeError:
            return "Array (Length: 1)"
    if isinstance(ele, (pd.Timestamp, pd.Timedelta)):
        return str(ele)
    return ele
