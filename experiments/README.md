# experiments/

仮説 H01〜H07 の実験です。仮説と判定の条件は `資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.pdf`、結果は `資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験結果レポート.pdf` にあります。

## 構成

```
experiments/
├── common/            実験で共通に使う部品
│   ├── data.py        判断日ごとの入力と翌月の結果（data/cache/dates/ にキャッシュ。DB は読み取り専用で開く）
│   ├── backtest.py    月次のバックテスト（値動きによる比率の変化、売買の量、片道 0.1% のコスト）
│   ├── strategies.py  BL の戦略（ビューの形、Ω の決め方、較正、TE の上限、置換の対照）
│   ├── metrics.py     IC、Fama-MacBeth、成績の指標、Holm の補正
│   └── llm.py         OpenAI の呼び出しと記録（プロンプト、回答、トークン数、費用を JSONL に残す）
├── H01_verification/       計算の再現と整合性
├── H02_statistical_baseline/
├── H03_view_shape/
├── H04_omega_calibration/
├── H05_text_vs_numbers/
├── H06_earnings_text/      fetch.py → score.py → evaluate.py
├── H07_feedback_llm/
└── robustness/            結果を別のモデルに読ませた後の追加の確認（探索的）とレビューの記録
```

各フォルダの `results/` に、条件と数値（`summary.json` など）と LLM の全回答（`*.jsonl`）を置きます。

## 再現の手順（リポジトリの直下で）

```bash
python -m unittest discover tests                       # 式・最適化・較正・バックテストのテスト
python experiments/H01_verification/run.py
python experiments/H02_statistical_baseline/run.py
python experiments/H03_view_shape/run.py
python experiments/H04_omega_calibration/run.py          # 置換の対照 200 回（12 プロセスで並列）
python experiments/H05_text_vs_numbers/run.py ask        # API を使う
python experiments/H05_text_vs_numbers/run.py evaluate
SEC_USER_AGENT="<名前> <連絡先メール>" python experiments/H06_earnings_text/fetch.py   # SEC EDGAR（毎秒 8 回以下）
python experiments/H06_earnings_text/score.py            # API を使う
python experiments/H06_earnings_text/evaluate.py
python experiments/H07_feedback_llm/run.py ask           # API を使う
python experiments/H07_feedback_llm/run.py evaluate
python experiments/robustness/run.py nw            # nw | temperature | feedback | placebo
python 資料集/02_実験/実験1_仮説の検証/LLM-BLM_実験結果レポート_図/作成スクリプト/figs.py
```

- DB は `.env` の `DB_PATH`（`data/blm_2015.sqlite`）を読み取り専用で開きます。実験は DB に書き込みません。
- API キーは `.env` の `OPENAI_API_KEY` を使います。予想に使うモデルは gpt-4o-mini だけです（知識の期限が評価期間より前のため）。
- LLM の回答は乱数を含むので、同じコマンドを実行しても同じ回答にはなりません。再現性は「保存した回答（`*.jsonl`）から同じ数値を計算し直せること」とします。`ask` は回答済みのリクエストを飛ばすので、途中で止まっても続きから再開できます。
- 決算発表文の本文（`data/cache/edgar/`）は Git に入れていません。`fetch.py` で取り直せます。ただし、H06 の読み取りの記録（`H06_earnings_text/results/scores.jsonl`、91MB）には、LLM に送ったプロンプトとして本文の先頭 15,000 文字が入っていて、Git にコミット済みです（`experiment/H06-earnings-text` 以降）。
