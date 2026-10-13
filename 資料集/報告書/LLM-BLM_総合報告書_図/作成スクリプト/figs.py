"""Figures of the report (vector PDF). Palette: validated reference palette, slots in fixed order."""
import sys, json, sqlite3, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
B = '/private/tmp/claude-501/-Users-kazuki-develop---/9d2180f2-ca7f-42f9-a43d-a8ea8ccec7c4/scratchpad/report/build/'
F = '/Users/kazuki/develop/工房/LLM-BLM_総合報告書_図/'
E = '/Users/kazuki/develop/工房/LLM-BLM_実験/'
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
plt.rcParams.update({'font.family': 'Arial Unicode MS', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984',
                     'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b', 'axes.grid': True, 'grid.color': '#e6e5e1',
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'axes.titlesize': 10, 'axes.titleweight': 'normal'})
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # fixed slot order
INK2 = '#52514e'; NEUTRAL = '#b9b8b2'
def save(fig, name): fig.savefig(F + name + '.pdf'); plt.close(fig); print(name)
def barlabels(ax, bars, fmt='{:+.2f}', dy=0.012):
    for b in bars:
        h = b.get_height(); ax.text(b.get_x() + b.get_width() / 2, h + (dy if h >= 0 else -dy), fmt.format(h), ha='center', va='bottom' if h >= 0 else 'top', fontsize=7.5, color='#0b0b0b')

# 1 index check
d = pd.read_csv(B + 'index_check.csv', index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.plot(d.index, d.gspc, color=C[1], lw=2, label='実際の S&P500 指数（^GSPC）'); ax.plot(d.index, d.db, color=C[0], lw=1.2, ls=(0, (4, 2)), label='DB から作った時価総額加重の指数')
ax.set_ylabel('2023-03-01 を 1 とした値'); ax.legend(loc='upper left'); save(fig, 'index_check')

# 2 answers histogram
a = np.load(B + 'answers_sample.npy') * 100
fig, ax = plt.subplots(figsize=(6.3, 2.6))
ax.hist(np.clip(a, -1.5, 2.0), bins=np.arange(-1.5, 2.01, 0.05), color=C[0], edgecolor='white', linewidth=0.3)
ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel('LLM の回答（翌月の日次平均リターン、%）'); ax.set_ylabel('回答の数（6万個を抽出）'); save(fig, 'answers_hist')

# 3 view vs input scatter + monthly correlation
s = pd.read_csv(B + 'scatter_2025-06.csv', index_col=0).dropna(); m = pd.read_csv(B + 'views_monthly.csv')
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axs[0]; ax.scatter(s.input_mean * 100, s['view'] * 100, s=7, color=C[0], alpha=0.55, linewidths=0)
lim = [-1.2, 1.6]; ax.plot(lim, lim, color=NEUTRAL, lw=0.8, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('渡した月の日次平均リターン（%）'); ax.set_ylabel('LLM の予想（%）'); ax.set_title(f'判断日 2025-06-30（相関 {s.input_mean.corr(s["view"]):.2f}）')
ax = axs[1]; ax.bar(range(len(m)), m.corr_input, color=C[0], width=0.72); ax.set_ylim(0, 1); ax.set_xticks(range(0, len(m), 6)); ax.set_xticklabels([x[:7] for x in m.as_of_date[::6]])
ax.axhline(m.corr_input.mean(), color=INK2, lw=0.8, ls='--'); ax.text(len(m) - 0.5, m.corr_input.mean() + 0.02, f'平均 {m.corr_input.mean():.2f}', ha='right', fontsize=8)
ax.set_title('予想と入力の相関（判断日ごと）'); ax.grid(axis='x', visible=False); save(fig, 'view_vs_input')

# 4 rank IC per month
fig, ax = plt.subplots(figsize=(6.3, 2.4))
ax.bar(range(len(m)), m.rank_ic, color=[C[0] if x >= 0 else C[1] for x in m.rank_ic], width=0.72); ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(0, len(m), 3)); ax.set_xticklabels([x[:7] for x in m.as_of_date[::3]], rotation=0, fontsize=7.5); ax.set_ylabel('順位相関'); ax.grid(axis='x', visible=False)
ax.set_title(f'予想と翌月の実際のリターンの順位相関（平均 {m.rank_ic.mean():+.3f}、プラスは {int((m.rank_ic > 0).sum())}/30 か月）'); save(fig, 'rank_ic')

# 5 cumulative returns (original calculation)
r = pd.read_csv(B + 'monthly_returns.csv', index_col=0)
lab = {'market_cap': '時価総額加重', 'equal_weight': '等金額', 'mean_variance_original': '平均分散（元の計算）', 'black_litterman_original': 'BL（元の計算）', 'mean_variance': '平均分散（直した計算）', 'black_litterman': 'BL（直した計算）'}
def cum(ax, cols, colors, log=False, direct=True):
    x = pd.to_datetime([r.index[0]] + list(r.index[1:]) + ['2026-09-30'])
    for c, col in zip(cols, colors):
        w = np.r_[1, (1 + r[c]).cumprod()]; ax.plot(x, w, color=col, lw=2, label=f'{lab[c]}（最終 {w[-1]:.2f}）')
        if direct: ax.text(x[-1] + pd.Timedelta(days=8), w[-1], f'{lab[c]}  {w[-1]:.2f}', va='center', fontsize=7.8)
    if log: ax.set_yscale('log')
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=250 if direct else 20))
    tk = pd.date_range('2024-04-01', '2026-10-01', freq='6MS'); ax.set_xticks(tk); ax.set_xticklabels([t.strftime('%Y-%m') for t in tk])
    if not direct: ax.legend(fontsize=8, loc='upper left')
    ax.set_ylabel('資産（開始時 = 1）' + ('、対数目盛' if log else ''))
fig, ax = plt.subplots(figsize=(6.5, 3.0)); cum(ax, ['black_litterman_original', 'mean_variance_original', 'market_cap', 'equal_weight'], [C[0], C[1], C[2], C[3]]); save(fig, 'cumret_original')
fig, ax = plt.subplots(figsize=(6.5, 3.0)); cum(ax, ['mean_variance', 'black_litterman', 'market_cap', 'equal_weight'], [C[1], C[0], C[2], C[3]], log=True, direct=False); save(fig, 'cumret_fixed')

# 6 effective holdings
e = pd.read_csv(B + 'effective_holdings.csv', index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(6.5, 2.9))
for c, col in zip(['equal_weight', 'market_cap', 'black_litterman_original', 'black_litterman'], [C[3], C[2], C[0], C[1]]):
    ax.plot(e.index, e[c], color=col, lw=2); ax.text(e.index[-1] + pd.Timedelta(days=8), e[c].iloc[-1], lab[c], va='center', fontsize=7.8)
ax.set_yscale('log'); ax.set_xlim(e.index[0], e.index[-1] + pd.Timedelta(days=230)); ax.set_ylabel('実質的な銘柄数（対数目盛）'); tk = pd.date_range('2024-04-01', '2026-10-01', freq='6MS'); ax.set_xticks(tk); ax.set_xticklabels([t.strftime('%Y-%m') for t in tk]); save(fig, 'effective_holdings')

# 7 filing age
ages = np.load(B + 'ages.npy')
fig, ax = plt.subplots(figsize=(6.3, 2.5)); ax.hist(np.clip(ages, 0, 240), bins=np.arange(0, 245, 5), color=C[0], edgecolor='white', linewidth=0.3)
ax.axvline(150, color=C[1], lw=1.4); ax.text(153, ax.get_ylim()[1] * 0.85, '150日：これを超えたら\nファンダの指標を作らない', fontsize=8, color='#0b0b0b')
ax.set_xlabel('最新の決算の期末から判断日までの日数（240日以上は右端にまとめた）'); ax.set_ylabel('銘柄 × 月'); save(fig, 'filing_age')

# 8 scorecard check
sc = json.load(open(E + 'scorecards.json')); conn = sqlite3.connect("file:/Users/kazuki/develop/工房/LLM-BLM/data/blm.sqlite?mode=ro", uri=True)
sm = pd.read_csv(B + 'sector_metrics.csv'); pts = {k: ([], []) for k in ['3m', '6m', '12m', 'vol']}
for x in sc:
    g = sm[sm.as_of_date == x['as_of']]; z = g.pivot(index='gics_sector', columns='metric', values='z_score'); val = g.pivot(index='gics_sector', columns='metric', values='value')
    mom = pd.DataFrame(x['tables'][4], index=['3m', '6m', '12m', 't']).T; vol = pd.DataFrame(x['tables'][5], index=['rv', 'iv', 'p', 'c']).T
    for k in ['3m', '6m', '12m']: pts[k][0].extend(mom[k]); pts[k][1].extend(z[f'momentum_{k}'][mom.index])
    pts['vol'][0].extend(vol['rv']); pts['vol'][1].extend(val['volatility_30d'][vol.index] * 100)
fig, axs = plt.subplots(1, 4, figsize=(6.8, 2.2))
for ax, k, t in zip(axs, ['3m', '6m', '12m', 'vol'], ['モメンタム 3か月', 'モメンタム 6か月', 'モメンタム 12か月', '実現ボラティリティ（%）']):
    x, y = np.array(pts[k][0]), np.array(pts[k][1]); ax.scatter(x, y, s=8, color=C[0], alpha=0.7, linewidths=0); lo, hi = min(x.min(), y.min()), max(x.max(), y.max()); ax.plot([lo, hi], [lo, hi], color=NEUTRAL, lw=0.8, ls='--')
    ax.set_title(f'{t}\n相関 {np.corrcoef(x, y)[0, 1]:.2f}', fontsize=8.5); ax.tick_params(labelsize=7)
axs[0].set_ylabel('自作の値'); fig.supxlabel('スコアカードの値（7か月 × 11セクター）', fontsize=8.5, color=INK2); fig.tight_layout(); save(fig, 'scorecard')

# 9 sector momentum heatmap (diverging, neutral midpoint)
from matplotlib.colors import LinearSegmentedColormap
hm = sm[(sm.metric == 'momentum_12m') & (sm.as_of_date >= '2024-03-01')].pivot(index='gics_sector', columns='as_of_date', values='z_score')
cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])
fig, ax = plt.subplots(figsize=(6.8, 2.9)); im = ax.imshow(hm.values, cmap=cmap, vmin=-2.5, vmax=2.5, aspect='auto')
ax.set_yticks(range(len(hm.index))); ax.set_yticklabels(hm.index, fontsize=7.5); ax.set_xticks(range(0, hm.shape[1], 3)); ax.set_xticklabels([c[:7] for c in hm.columns[::3]], fontsize=7); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label('z スコア', fontsize=8); cb.outline.set_visible(False); save(fig, 'sector_momentum')

# 10 experiment D
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={'width_ratios': [1.5, 1]})
labs = ['1か月\nリターン', '12か月\nモメンタム', 'PBR', 'PER', '純利益\n前年比', '純利\n益率', 'ROE']
D = {'REIT': [0.52, 0.43, 0.10, 0.15, 0.25, -0.04, -0.05], '資本財': [0.55, 0.42, 0.15, 0.18, 0.20, -0.03, 0.07]}
x = np.arange(len(labs)); ax = axs[0]
for i, (k, v) in enumerate(D.items()): ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=C[i], label=k)
ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=7.2); ax.set_ylim(-0.15, 0.7); ax.legend(fontsize=8); ax.set_title('予想と各指標の順位相関'); ax.grid(axis='x', visible=False)
ax = axs[1]; g = ['REIT', '銀行など', '資本財']; eff = [0.073, 0.061, 0.071]; noi = [0.045, 0.066, 0.057]; x = np.arange(3)
b1 = ax.bar(x - 0.19, eff, 0.36, color=C[0], label='指標を渡したときの変化'); b2 = ax.bar(x + 0.19, noi, 0.36, color=NEUTRAL, label='同じ質問を2回したときのぶれ')
barlabels(ax, list(b1) + list(b2), '{:.3f}', 0.002); ax.set_xticks(x); ax.set_xticklabels(g); ax.set_ylim(0, 0.105); ax.set_ylabel('予想の差（日次、%）'); ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.02)); ax.grid(axis='x', visible=False)
fig.tight_layout(); save(fig, 'exp_d')

# 11 experiment E
conds = ['指定なし', '50：50', '20：80', '0：100']; fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={'width_ratios': [1.6, 1]})
Ed = {'直近1か月リターン': [0.58, 0.64, 0.68, 0.24], '12か月モメンタム': [0.36, 0.39, 0.34, 0.23], '売上の前年比': [0.21, 0.13, 0.23, 0.42], '純利益の前年比': [0.15, 0.14, 0.26, 0.51]}
x = np.arange(4); ax = axs[0]
for i, (k, v) in enumerate(Ed.items()): ax.bar(x + (i - 1.5) * 0.2, v, 0.19, color=C[i], label=k)
ax.set_xticks(x); ax.set_xticklabels(conds); ax.set_ylim(0, 0.85); ax.legend(fontsize=7, ncol=2, loc='upper center'); ax.set_title('予想と各指標の順位相関'); ax.set_xlabel('値動き：ファンダの指定'); ax.grid(axis='x', visible=False)
ax = axs[1]; b = ax.bar(x, [2, 40, 37, 16], 0.6, color=C[0]); barlabels(ax, b, '{:.0f}%', 0.6); ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=7.5); ax.set_ylim(0, 48); ax.set_title('日次 1% を超える回答の割合'); ax.grid(axis='x', visible=False)
fig.tight_layout(); save(fig, 'exp_e')

# 12 experiment F
conds = ['1ターン\n数字', '3ターン\n数字', '1ターン\n5段階', '3ターン\n5段階']; fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6))
for ax, t, v1, v2 in [(axs[0], '資本財（238）', [0.58, 0.80, 0.77, 0.65], [0.36, 0.49, 0.46, 0.69]), (axs[1], 'REIT（81）', [0.53, 0.66, 0.67, 0.38], [0.37, 0.42, 0.28, 0.69])]:
    ax.bar(x - 0.2, v1, 0.38, color=C[0], label='直近1か月リターン'); ax.bar(x + 0.2, v2, 0.38, color=C[1], label='12か月モメンタム'); ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=7); ax.set_ylim(0, 1); ax.set_title(t); ax.grid(axis='x', visible=False)
axs[0].legend(fontsize=7, loc='upper left'); axs[0].set_ylabel('予想との順位相関')
ax = axs[2]; b = ax.bar(x, [0.46, 0.84, 0.83, 0.85], 0.6, color=C[2]); barlabels(ax, b, '{:.2f}', 0.01); ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=7); ax.set_ylim(0, 1); ax.set_title('入力の順位で説明できる割合'); ax.grid(axis='x', visible=False)
fig.tight_layout(); save(fig, 'exp_f')

# 13 experiment G
g = pd.read_pickle(E + 'premium_scored.pkl'); mk = g[(g.kind == 'factor') & (g.name == 'market')].sort_values('date')
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6))
ax = axs[0]; ax.scatter(mk.sp_1m * 100, mk.score, s=16, color=C[0], linewidths=0); ax.set_xlabel('S&P500 の直近1か月のリターン（%）'); ax.set_ylabel('市場リスクの点数（1〜5）'); ax.set_title(f'点数と入力（順位相関 {mk.score.rank().corr(mk.sp_1m.rank()):+.2f}）', fontsize=8.5)
ax = axs[1]; ax.scatter(mk.score, mk.realized * 100, s=16, color=C[1], linewidths=0); ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel('市場リスクの点数（1〜5）'); ax.set_ylabel('翌月の実際の超過リターン（%）'); ax.set_title(f'点数と実際（順位相関 {mk.score.rank().corr(mk.realized.rank()):+.2f}）', fontsize=8.5)
ax = axs[2]; P = np.array([json.loads(x)['probs'] for x in g.answer]); P = P / P.sum(1, keepdims=True); b = ax.bar(range(1, 6), P.mean(0) * 100, 0.6, color=C[2]); barlabels(ax, b, '{:.0f}%', 0.6); ax.set_ylim(0, 58); ax.set_xlabel('回答の数字'); ax.set_title('回答の確率の配分（540問の平均）', fontsize=8.5); ax.grid(axis='x', visible=False)
fig.tight_layout(); save(fig, 'exp_g')
f = g[g.kind == 'factor']; piv = f.pivot(index='date', columns='name', values='score'); piv.index = pd.to_datetime(piv.index)
fig, ax = plt.subplots(figsize=(6.5, 2.7))
for c, col, lb in zip(['market', 'value', 'momentum', 'size'], C, ['市場', '割安さ', '勢い', '規模']):
    ax.plot(piv.index, piv[c], color=col, lw=1.8, marker='o', ms=3, label=lb)
ax.axhline(3, color=INK2, lw=0.8, ls='--'); ax.set_ylim(1.5, 5.2); ax.legend(ncol=4, fontsize=8, loc='lower center'); ax.set_ylabel('点数（3 = 同じくらい）'); save(fig, 'exp_g_series')
s = g[g.kind == 'sector']; ic = s.groupby('date').apply(lambda x: x.score.rank().corr(x.realized.rank())); ms = s.groupby('name').score.mean().sort_values()
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.8), gridspec_kw={'width_ratios': [1, 1.2]})
ax = axs[0]; ax.barh(range(len(ms)), ms.values, color=C[0], height=0.65); ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms.index, fontsize=7.3); ax.axvline(3, color=INK2, lw=0.8, ls='--'); ax.set_xlim(1, 4); ax.set_title('セクターの点数（30か月の平均）', fontsize=8.5); ax.grid(axis='y', visible=False)
ax = axs[1]; ax.bar(range(len(ic)), ic.values, color=[C[0] if v >= 0 else C[1] for v in ic.values], width=0.72); ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(range(0, len(ic), 6)); ax.set_xticklabels([d[:7] for d in ic.index[::6]], fontsize=7.5)
ax.set_title(f'点数と翌月の成績の順位相関（平均 {ic.mean():+.2f}）', fontsize=8.5); ax.grid(axis='x', visible=False); fig.tight_layout(); save(fig, 'exp_g_sector')

# 14 report extraction: stance and S&P targets
rp = pd.read_csv(B + 'reports.csv'); fig, axs = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.3]})
x = pd.to_datetime(rp.report_month + '-01'); ax = axs[0]
ax.step(x, rp.us_equity_stance, where='mid', color=C[0], lw=2, label='米国株'); ax.step(x, rp.us_economy_stance, where='mid', color=C[1], lw=1.6, label='米国経済'); ax.set_yticks([-2, -1, 0, 1, 2]); ax.set_ylim(-2.3, 2.3); ax.legend(ncol=2, fontsize=8, loc='lower right'); ax.set_ylabel('見通し（−2 弱気 〜 +2 強気）')
ax = axs[1]; ax.plot(x, rp.sp500_target_this_year, color=C[0], lw=1.8, marker='o', ms=3, label='年末の予想値（その年）'); ax.plot(x, rp.sp500_target_next_year, color=C[1], lw=1.8, marker='o', ms=3, label='年末の予想値（翌年）')
ax.set_ylim(3800, 10500); bad = rp[rp.sp500_target_this_year > 20000]
for _, q in bad.iterrows(): ax.annotate(f"{q.report_month} は {q.sp500_target_this_year:,.0f} と回答\n（別の指数の行を読んだ誤り）", (pd.Timestamp(q.report_month + '-01'), 10500), (pd.Timestamp(q.report_month + '-01') + pd.Timedelta(days=60), 9300), fontsize=7.5, arrowprops={'arrowstyle': '->', 'color': '#52514e', 'lw': 0.8})
ax.set_ylabel('S&P500（LLM が読み取った値）'); ax.legend(fontsize=8, loc='lower right'); fig.tight_layout(); save(fig, 'reports')
