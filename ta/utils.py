# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd


def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)]  # big number
    df = df[df != 0.0]
    return df.dropna()


def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()


def get_min_max(x1, x2, f='min'):
    if np.isnan(x1) or np.isnan(x2):
        return np.nan
    if f == 'max':
        return max(x1, x2)
    elif f == 'min':
        return min(x1, x2)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')
