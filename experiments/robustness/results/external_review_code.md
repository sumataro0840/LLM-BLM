# 共通部品のコードへのレビュー：gpt-5.3-codex の回答（2026-10-16）

以下、**実害が出るバグのみ**を重要度順で挙げます（最大8件以内）。

---

1. **将来データ漏洩の入口がある**
- **場所**: `experiments/common/backtest.py` / `run`
- **問題**: `strategy(d, x, rows)` に渡す `x` が `next_return`, `next_daily`, `next_risk_free` などの**将来アウトカムを含む**。  
  現在の同梱戦略は使っていなくても、戦略側が参照できる時点でバックテスト基盤としてリーク。
- **修正**: 戦略へ渡すのは入力のみ（`next_`系を除外）にする。  
  例: `x_in = {k:v for k,v in x.items() if not k.startswith("next_")}` を渡す。  
  可能なら `data` 構造を `{inputs, outcomes}` に明示分離。

---

2. **lagged実行の損益計上が壊れている（1営業日分が欠落）**
- **場所**: `experiments/common/backtest.py` / `run`（`lagged=True`）
- **問題**: `next_return_lagged` だけで `gross` を計算するため、**決定日クローズ→翌営業日エントリーまでの1日リターン**がどの資産にも計上されない（暗黙に現金化）。  
  また売買回転も決定日クローズ時ウェイト基準で、実際の約定時点ウェイト（翌日）と不整合。
- **修正**:
  1) `data._build` で `ret_to_entry = next_price/start_price - 1` を保存  
  2) `lagged=True` 時は `before` をこの1日でドリフトさせてから turnover 計算  
  3) 同1日分のPnL（旧保有 or 明示的cash）を `gross` に加える

---

3. **欠損価格を0%扱いしており、成績が上振れしうる**
- **場所**:  
  - `experiments/common/data.py` / `_build`（`last_price = prices.ffill().iloc[-1]`）  
  - `experiments/common/backtest.py` / `run`（`returns.fillna(0.0)`）
- **問題**: 期末価格欠損やlaggedのエントリー価格欠損が**0%リターン化**される。  
  データ欠損・上場廃止・取引停止が実質無害化され、バイアス。
- **修正**: NaNを0で埋めない。  
  ルールを明示して実装（例:  
  - エントリー不可銘柄は当月ウェイト0にして再正規化、  
  - 期中欠損は最終実価格で強制清算し以後cash運用、  
  - それも無理なら当該銘柄をその期の投資対象から除外）。

---

4. **外部forecast使用時のキャリブレーションが誤っている**
- **場所**: `experiments/common/strategies.py` / `bl_strategy`
- **問題**: `forecasts` 引数を渡しても、`slope`計算は `panel.Q`（既定LLM）を使っている。  
  ドキュメント「measured on them」と不一致で、採用率 `b` が別モデル由来になる。
- **修正**: `forecasts` 指定時は `panel` の `Q` を date+symbol で置換してから `past_slope` を計算。  
  もしくは `forecasts` 用の panel を事前生成して必ずそれを渡す。

---

5. **日付順が保証されず、将来情報がhistoryに混入しうる**
- **場所**: `experiments/common/backtest.py` / `run`
- **問題**: `for d, x in all_data.items():` で順序を前提。  
  `load_all()`由来なら通常問題化しにくいが、外部から順不同dictを渡すと**futureがhistoryに入る**。
- **修正**: `for d in sorted(all_data): x = all_data[d]` とし、時系列順を強制。

---

6. **キャッシュキー不足で別設定データが混線する**
- **場所**: `experiments/common/data.py` / `load_date`
- **問題**: キャッシュファイル名が `as_of_date.pkl` のみ。  
  `LLM_RUN_ID`, `COV_WINDOW_DAYS`, `MIN_HISTORY_RATIO` 等を変えても同じファイルを再利用し、実験が汚染される。
- **修正**: ファイル名 or メタデータ検証に設定値を含める。  
  例: `f"{as_of_date}_run{LLM_RUN_ID}_cov{COV_WINDOW_DAYS}_min{MIN_HISTORY_RATIO}_v{CACHE_VERSION}.pkl"`。

---

7. **BL後方平均計算の数値不安定**
- **場所**: `blm/black_litterman.py` / `posterior_returns_general`（同様に `posterior_returns`）
- **問題**: `np.linalg.solve(P τΣ P' + Ω, ...)` を直で解いており、行列が悪条件/準特異だと不安定・失敗。  
  特に relative view + 小さいΩで起こりやすい。
- **修正**:  
  - 行列対称化 `A = 0.5*(A + A.T)`  
  - 小さなリッジ `A += 1e-10*I`  
  - 失敗時 `pinv` フォールバック（または `cho_factor/cho_solve`）

---

必要なら、上の1〜3を先に直すための最小パッチ案（差分形式）を出します。
