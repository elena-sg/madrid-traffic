import numpy as np
import pandas as pd


def rows_no_change(col):
    """
    Get number of rows with no change
    :param col: Pandas Series
    :return: Pandas Series with
    """
    return col.groupby(
        ((col != col.shift())).cumsum()
    ).transform('size')


def summary_change(df):
    dfs = pd.concat([
        df[c].value_counts(normalize=True).sort_index().cumsum()
        for c in df.columns if c.startswith('nochange-')
    ], axis=1)
    return dfs


def make_stable_values_null(col, nrows=4):
    no_change = rows_no_change(col)
    return np.where(no_change > nrows, np.nan, col)
