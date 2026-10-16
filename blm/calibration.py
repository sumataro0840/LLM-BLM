"""
How far to trust the views, measured from past forecasts and outcomes only.

The calibration slope b is the regression slope of the realized returns on the forecasts after the
average of each decision date is taken out of both (only differences between stocks are scored).
b is the share of a forecast that, on average, showed up in the outcome; it is used as the adoption
rate of the views (Omega proportional to Sigma: posterior = prior + b (views - prior)).
"""
import numpy as np


def calibration_slope(forecasts, outcomes, dates):
    """Pooled slope over all past (stock, date) pairs, and the standard error of the monthly slopes.

    forecasts, outcomes, dates: aligned 1-d arrays. Returns (slope, standard error, number of dates).
    """
    forecasts, outcomes, dates = np.asarray(forecasts, float), np.asarray(outcomes, float), np.asarray(dates)
    keep = np.isfinite(forecasts) & np.isfinite(outcomes)
    forecasts, outcomes, dates = forecasts[keep], outcomes[keep], dates[keep]
    unique = np.unique(dates)
    if len(unique) == 0:
        return 0.0, np.inf, 0
    f_dev, r_dev, monthly = [], [], []
    for d in unique:
        f = forecasts[dates == d] - forecasts[dates == d].mean()
        r = outcomes[dates == d] - outcomes[dates == d].mean()
        f_dev.append(f)
        r_dev.append(r)
        if (f ** 2).sum() > 0:
            monthly.append((f * r).sum() / (f ** 2).sum())
    f_all, r_all = np.concatenate(f_dev), np.concatenate(r_dev)
    slope = float((f_all * r_all).sum() / (f_all ** 2).sum()) if (f_all ** 2).sum() > 0 else 0.0
    se = float(np.std(monthly, ddof=1) / np.sqrt(len(monthly))) if len(monthly) > 1 else np.inf
    return slope, se, len(unique)


def adoption_rate(slope, se=None, n_dates=None, min_dates=3, cap=0.99):
    """Adoption rate used for a decision date: the slope clipped to [0, cap], 0 before min_dates outcomes exist.

    With se given, the slope is first shrunk toward 0 by its estimation error: b max(0, 1 - se^2 / b^2)
    (a slope measured with a t-value of 1 or less is not used at all).
    """
    if n_dates is not None and n_dates < min_dates:
        return 0.0
    if slope <= 0:
        return 0.0
    if se is not None:
        slope = slope * max(0.0, 1 - se ** 2 / slope ** 2)
    return float(min(slope, cap))
