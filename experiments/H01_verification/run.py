"""
H01: reproduce the current calculation and check its consistency (no API calls).

    python experiments/H01_verification/run.py

Writes results/summary.json (every figure) and results/monthly.csv.
The checks and pass conditions are those of 資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.pdf, chapter H01.
"""
import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from blm.black_litterman import (posterior_returns, posterior_returns_general, posterior_returns_inverse_form,
                                 posterior_returns_original)
from blm.covariance import sample_covariance
from blm.equilibrium import capm_returns_original, implied_returns
from blm.optimizer import optimize, optimize_original
from experiments.common import data as D
from experiments.common.metrics import t_summary
from marketdb.db.index import load_members
from marketdb.db.market import last_trading_days

OUT = Path(__file__).resolve().parent / 'results'
TAU, DELTA, RF_ORIGINAL, RETURN_WEIGHT = 0.025, 2.5, 0.02, 0.1   # the values in .env and the reports


def effective_n(w):
    return float(1 / (w ** 2).sum())


def overlap(a, b):
    idx = a.index.union(b.index)
    return float(np.minimum(a.reindex(idx).fillna(0), b.reindex(idx).fillna(0)).sum())


def slsqp_original(mu, sigma):
    """optimize_original, keeping scipy's result (success flag, iterations, message)."""
    n = len(mu)
    f = lambda x: x @ sigma @ x - RETURN_WEIGHT * (x @ mu)
    res = minimize(f, np.ones(n) / n,
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, {'type': 'ineq', 'fun': lambda x: x}),
                   bounds=tuple((0, 1) for _ in range(n)))
    return res, f


def exact_original(mu, sigma):
    """The same objective solved to optimality (quadratic program)."""
    w = cp.Variable(len(mu))
    prob = cp.Problem(cp.Minimize(1e4 * (cp.quad_form(w, cp.psd_wrap(sigma)) - RETURN_WEIGHT * mu @ w)),
                      [cp.sum(w) == 1, w >= 0, w <= 1])
    prob.solve(solver=cp.CLARABEL)
    return np.clip(w.value, 0, None), prob.status


def main():
    OUT.mkdir(exist_ok=True)
    conn = D.connect()
    all_data = D.load_all(conn)
    dates = list(all_data)
    stored = pd.read_sql("SELECT r.strategy, w.as_of_date, w.symbol, w.weight FROM portfolio_weights w "
                         "JOIN portfolio_runs r USING (portfolio_id) WHERE r.portfolio_id IN (4, 5)", conn)
    summary, monthly = {}, []
    t0 = time.time()

    for d in dates:
        x = all_data[d]
        S, wm = x['covariance'], x['market_weights']
        Pi = implied_returns(S, wm, DELTA)
        q = x['view_mean'] - x['risk_free_daily']
        var = x['view_variance']
        row = {'date': d, 'n': len(wm)}

        # (2) no views -> market weights
        w0 = optimize(Pi, S, DELTA)
        row['no_view_max_diff'] = float((w0 - wm).abs().max())

        # (1) the two forms of the posterior on the real data (P = I)
        identity = pd.DataFrame(np.eye(len(q)), columns=q.index)
        mu_update = posterior_returns(Pi, S, q, var, TAU)
        mu_inverse = posterior_returns_inverse_form(Pi, S, identity, pd.Series(q.values), np.diag(var.values), TAU)
        row['forms_rel_diff'] = float((mu_update - mu_inverse).abs().max() / mu_update.abs().max())
        # (4) conditioning of the system that is solved
        m = TAU * S.values + np.diag(var.values)
        row['cond_tauSigma_plus_Omega'] = float(np.linalg.cond(m))
        sol = np.linalg.solve(m, (q - Pi).values)
        row['solve_residual'] = float(np.abs(m @ sol - (q - Pi).values).max() / np.abs((q - Pi).values).max())
        # (5) adoption of the fixed calculation
        gap = q - Pi
        row['adoption_fixed'] = float(((mu_update - Pi) / gap).where(gap.abs() > 1e-5).median())
        row['view_share_of_tauSigma'] = float((var / (TAU * np.diag(S.values))).median())

        # (6) the optimizer of the fixed calculation, and a second solver on the same problem
        w_fixed = optimize(mu_update, S, DELTA)
        row['fixed_effective_n'] = effective_n(w_fixed)
        obj = lambda w: float(mu_update.values @ w - DELTA / 2 * w @ S.values @ w)
        wv = cp.Variable(len(q))
        p2 = cp.Problem(cp.Maximize(1e4 * (mu_update.values @ wv - DELTA / 2 * cp.quad_form(wv, cp.psd_wrap(S.values)))),
                        [cp.sum(wv) == 1, wv >= 0])
        p2.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, max_iter=200000)
        w_osqp = np.clip(wv.value, 0, None) / np.clip(wv.value, 0, None).sum()
        row['fixed_objective_gap_vs_osqp'] = obj(w_fixed.values) - obj(w_osqp)   # >= 0 means Clarabel is at least as good
        st = stored[(stored.strategy == 'black_litterman') & (stored.as_of_date == d)].set_index('symbol').weight
        row['fixed_vs_db_max_diff'] = float((w_fixed - st.reindex(w_fixed.index).fillna(0)).abs().max()) if len(st) else np.nan

        # the original calculation
        month = x['month_returns']
        S0 = sample_covariance(month)
        prior0 = capm_returns_original(month, wm, RF_ORIGINAL)
        mu0 = posterior_returns_original(prior0, S0, x['view_mean'], var, TAU)
        gap0 = x['view_mean'] - prior0
        row['adoption_original'] = float(((mu0 - prior0) / gap0).where(gap0.abs() > 1e-5).median())
        row['original_prior_mean'] = float(prior0.mean())
        row['original_cov_rank'] = int(np.linalg.matrix_rank(S0.values))
        res, f = slsqp_original(mu0.values, S0.values)
        w_slsqp = pd.Series(np.clip(res.x, 0, None), index=mu0.index)
        w_slsqp /= w_slsqp.sum()
        w_exact, status = exact_original(mu0.values, S0.values)
        f0 = f(np.ones(len(mu0)) / len(mu0))
        row.update({
            'slsqp_success': bool(res.success), 'slsqp_iterations': int(res.nit), 'slsqp_message': str(res.message),
            'slsqp_progress': float((f0 - f(res.x)) / (f0 - f(w_exact))),
            'original_effective_n': effective_n(w_slsqp), 'exact_effective_n': effective_n(pd.Series(w_exact)),
            'slsqp_vs_exact_overlap': overlap(w_slsqp, pd.Series(w_exact, index=mu0.index)), 'exact_status': status,
        })
        st0 = stored[(stored.strategy == 'black_litterman_original') & (stored.as_of_date == d)].set_index('symbol').weight
        row['original_vs_db_max_diff'] = float((w_slsqp - st0.reindex(w_slsqp.index).fillna(0)).abs().max()) if len(st0) else np.nan
        # also the existing function (it must give the same weights as the copy above)
        row['optimize_original_same'] = float((optimize_original(mu0, S0, RETURN_WEIGHT) - w_slsqp).abs().max())

        # (7) look-ahead audit of the inputs
        days = last_trading_days(conn, d, D.COV_WINDOW_DAYS + 1)
        row['window_last_day_is_date'] = days[-1] == d
        row['month_returns_last_day'] = str(month.index.max().date())
        row['outcome_first_day'] = str(x['next_daily'].index.min().date())
        row['universe_in_members'] = set(x['universe']) <= set(load_members(conn, d))
        monthly.append(row)
        print(f"{d}  fixed adoption {row['adoption_fixed']:.3f}  original {row['adoption_original']:.4f}  "
              f"slsqp {row['slsqp_progress']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    mon = pd.DataFrame(monthly).set_index('date')
    mon.to_csv(OUT / 'monthly.csv')

    # forecast panel: reproduce the figures of the reports
    panel = D.forecast_panel(all_data)
    p15 = panel[panel.n_next >= 15]
    per = []
    for d, g in p15.groupby('date'):
        per.append({'date': d, 'ic': g[['Q', 'r_next']].corr('spearman').iloc[0, 1],
                    'corr_input': g[['Q', 'r_in']].corr().iloc[0, 1],
                    'ratio': float(np.median(g.variance / (g.Q - g.r_next) ** 2)),
                    'ic_monthly_return': panel[panel.date == d][['Q', 'R_next']].corr('spearman').iloc[0, 1]})
    per = pd.DataFrame(per).set_index('date')
    dq = p15.Q - p15.groupby('date').Q.transform('mean')
    dr = p15.r_next - p15.groupby('date').r_next.transform('mean')

    # Omega = population variance of the 30 answers (direct from llm_views)
    raw = pd.read_sql("SELECT as_of_date, symbol, expected_return FROM llm_views WHERE run_id = 1", conn)
    direct = raw.groupby(['as_of_date', 'symbol']).expected_return.agg(lambda v: np.var(v.dropna()))
    view = pd.read_sql("SELECT as_of_date, symbol, variance FROM llm_view_stats WHERE run_id = 1", conn).set_index(['as_of_date', 'symbol']).variance
    # stock_metrics never uses a quarter filed after the decision date
    ages = conn.execute("SELECT MIN(value) FROM stock_metrics WHERE metric IN ('filing_age_days', 'fundamentals_age_days')").fetchone()[0]

    summary = {
        'months': len(mon),
        'universe_mean': float(mon.n.mean()),
        'check_forms_max_rel_diff': float(mon.forms_rel_diff.max()),
        'check_no_view_max_diff': float(mon.no_view_max_diff.max()),
        'check_cond_max': float(mon.cond_tauSigma_plus_Omega.max()),
        'check_solve_residual_max': float(mon.solve_residual.max()),
        'check_fixed_objective_gap_vs_osqp_min': float(mon.fixed_objective_gap_vs_osqp.min()),
        'check_fixed_vs_db_max_diff': float(mon.fixed_vs_db_max_diff.max()),
        'check_original_vs_db_max_diff': float(mon.original_vs_db_max_diff.max()),
        'check_optimize_original_same': float(mon.optimize_original_same.max()),
        'check_omega_is_population_variance_max_diff': float((direct - view.reindex(direct.index)).abs().max()),
        'check_min_filing_age_days': ages,
        'check_window_ends_on_date': bool(mon.window_last_day_is_date.all()),
        'check_month_returns_not_after_date': bool((pd.to_datetime(mon.month_returns_last_day) <= pd.to_datetime(mon.index)).all()),
        'check_outcomes_after_date': bool((pd.to_datetime(mon.outcome_first_day) > pd.to_datetime(mon.index)).all()),
        'check_universe_in_members': bool(mon.universe_in_members.all()),
        'adoption_original_median': float(mon.adoption_original.median()),
        'adoption_original_min': float(mon.adoption_original.min()),
        'adoption_fixed_median': float(mon.adoption_fixed.median()),
        'original_prior_mean_daily': [float(mon.original_prior_mean.min()), float(mon.original_prior_mean.max())],
        'original_cov_rank': [int(mon.original_cov_rank.min()), int(mon.original_cov_rank.max())],
        'slsqp_success_all': bool(mon.slsqp_success.all()),
        'slsqp_iterations': [int(mon.slsqp_iterations.min()), int(mon.slsqp_iterations.max())],
        'slsqp_progress_mean': float(mon.slsqp_progress.mean()),
        'slsqp_progress_min': float(mon.slsqp_progress.min()),
        'original_effective_n': float(mon.original_effective_n.mean()),
        'exact_effective_n': float(mon.exact_effective_n.mean()),
        'slsqp_vs_exact_overlap': float(mon.slsqp_vs_exact_overlap.mean()),
        'fixed_effective_n': float(mon.fixed_effective_n.mean()),
        'corr_forecast_input_mean': float(per.corr_input.mean()),
        'ic_mean_daily_outcome': t_summary(per.ic),
        'ic_monthly_return_outcome': t_summary(per.ic_monthly_return),
        'slope_pooled': float((dq * dr).sum() / (dq ** 2).sum()),
        'variance_over_squared_error_median': float(np.median(p15.variance / (p15.Q - p15.r_next) ** 2)),
        'share_positive_forecasts': float((panel.Q > 0).mean()),
    }
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
