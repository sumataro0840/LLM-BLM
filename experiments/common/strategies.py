"""
Black-Litterman strategies of the experiments H03 / H04: a view form (q, P), an Omega and an optimizer.

Every strategy sees, on a decision date, only that date's inputs and the outcomes of earlier decision
dates (the calibration slope b_t). The coefficients are those of the design (資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.pdf).
"""
import numpy as np
import pandas as pd
from scipy import stats

from blm.black_litterman import posterior_returns_general, posterior_returns_proportional
from blm.calibration import adoption_rate, calibration_slope
from blm.equilibrium import implied_returns
from blm.optimizer import optimize, optimize_with_tracking_error

TAU, DELTA = 0.025, 2.5
TE_TARGET = 0.03                      # ex-ante tracking error per year
TRADING_DAYS = 252
MIN_OUTCOME_DAYS = 15                 # an outcome month needs 15 daily returns or more (as in the reports)


def past_slope(panel, date):
    """Calibration slope of the LLM forecasts from decision dates before `date` (their outcomes are known on it)."""
    past = panel[(panel.date < date) & (panel.n_next >= MIN_OUTCOME_DAYS)]
    return calibration_slope(past.Q.values, past.r_next.values, past.date.values)


def blom_scores(values):
    """Normal scores of the ranks (Blom), ties averaged."""
    r = stats.rankdata(values, method='average')
    return stats.norm.ppf((r - 0.375) / (len(r) + 0.25))


def make_views(x, form, forecasts=None):
    """(P, q) of a view form. forecasts: the predicted average daily returns (default: the LLM run 1)."""
    Q = (x['view_mean'] if forecasts is None else forecasts).reindex(x['universe'])
    rf = x['risk_free_daily']
    n = len(Q)
    wm = x['market_weights'].reindex(Q.index).values
    if form == 'absolute':
        return pd.DataFrame(np.eye(n), columns=Q.index), pd.Series((Q - rf).values)
    if form == 'rank':
        z = blom_scores(Q.values)
        q = Q.mean() + Q.std() * z - rf
        return pd.DataFrame(np.eye(n), columns=Q.index), pd.Series(q)
    if form == 'relative':
        # each view: stock i minus the cap-weighted average of all stocks
        pick = np.eye(n) - np.outer(np.ones(n), wm)
        return pd.DataFrame(pick, columns=Q.index), pd.Series(pick @ Q.values)
    raise ValueError(form)


def posterior(x, pick, q, omega_form, b, answer_variance=None):
    S = x['covariance']
    Pi = implied_returns(S, x['market_weights'], DELTA)
    if omega_form == 'proportional':
        return Pi, posterior_returns_proportional(Pi, S, pick, q, b)
    P = pick.values
    PSP = P @ S.values @ P.T
    if omega_form == 'answer':
        # the answers' variance of each stock, carried to the views like a diagonal covariance would be
        omega = np.diag(np.diag(P @ np.diag(answer_variance.values) @ P.T))
    elif omega_form == 'he_litterman':
        omega = TAU * np.diag(np.diag(PSP))
    elif omega_form == 'calibrated_diagonal':
        if b <= 0:
            return Pi, Pi.copy()
        omega = TAU * np.diag(np.diag(PSP)) * (1 / b - 1)
    else:
        raise ValueError(omega_form)
    return Pi, posterior_returns_general(Pi, S, pick, q, omega, TAU)


def adoption_figures(x, Pi, mu, forecasts=None):
    """Median share of the gap to the forecasts that the posterior moved, in absolute and relative terms."""
    Q = (x['view_mean'] if forecasts is None else forecasts).reindex(Pi.index) - x['risk_free_daily']
    wm = x['market_weights'].reindex(Pi.index)
    gap, move = Q - Pi, mu - Pi
    rel_gap = gap - float(wm @ gap)
    rel_move = move - float(wm @ move)
    absolute = (move / gap).where(gap.abs() > 1e-5).median()
    relative = (rel_move / rel_gap).where(rel_gap.abs() > 1e-5).median()
    return float(absolute), float(relative)


def bl_strategy(panel, form='relative', omega='proportional', shrink=False, te=TE_TARGET, permute_seed=None,
                forecasts=None):
    """A strategy function for experiments.common.backtest.run.

    permute_seed: the forecasts are shuffled across the stocks of each date (an uninformative control with
    the same distribution); the calibration slope is still the one of the real forecasts.
    forecasts: {date: Series} to use instead of the LLM run 1 (same calibration rules, measured on them:
    the panel given must then hold these forecasts in its column Q).
    """
    def strategy(d, x, history):
        if forecasts is None:
            slope, se, n_dates = past_slope(panel, d)
        else:
            past = panel[(panel.date < d) & (panel.n_next >= MIN_OUTCOME_DAYS)]
            slope, se, n_dates = calibration_slope(past.Q.values, past.r_next.values, past.date.values)
        b = adoption_rate(slope, se if shrink else None, n_dates)
        f = None if forecasts is None else forecasts[d]
        if permute_seed is not None:
            base = (x['view_mean'] if f is None else f).reindex(x['universe'])
            rng = np.random.default_rng([permute_seed, int(d.replace('-', ''))])
            f = pd.Series(rng.permutation(base.values), index=base.index)
        pick, q = make_views(x, form, f)
        Pi, mu = posterior(x, pick, q, omega, b, x['view_variance'])
        if te is None:
            w, status = optimize(mu, x['covariance'], DELTA), 'optimal'
        else:
            w, status = optimize_with_tracking_error(mu, x['covariance'], DELTA, x['market_weights'],
                                                     te / np.sqrt(TRADING_DAYS))
        a_abs, a_rel = adoption_figures(x, Pi, mu, f)
        active = (w - x['market_weights']).values
        ex_ante_te = float(np.sqrt(active @ x['covariance'].values @ active) * np.sqrt(TRADING_DAYS))
        wm = x['market_weights']
        q_level = float(wm @ ((x['view_mean'] if f is None else f).reindex(wm.index)))
        return w, {'b': b, 'slope_raw': slope, 'slope_se': se, 'adoption_abs': a_abs, 'adoption_rel': a_rel,
                   'status': status, 'ex_ante_te': ex_ante_te, 'forecast_level': q_level}
    return strategy
