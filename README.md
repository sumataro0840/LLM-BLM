# LLM-BLM

論文 [Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model](https://arxiv.org/abs/2504.14345) の再現用リポジトリです。
元の実装（[youngandbin/LLM-BLM](https://github.com/youngandbin/LLM-BLM)）をもとに、OpenAI API だけを使う形に作り直しています。

今あるのは、**S&P500 のデータを DB に入れる部分**と、**LLM に予想（ビュー）を聞く部分**です。
それに加えて、**予想から比率を計算する部分**（Black-Litterman と最適化）があります。

## 構成

```
.
├── run_llm.py      LLM に各銘柄の予想リターンを聞く
├── run_blm.py      予想から比率を計算する
├── marketdb/       データの取得・加工・DB
│   ├── config.py
│   ├── sources/    取得（Wikipedia、yfinance、SEC EDGAR）
│   ├── transform/  加工（在籍期間、株式分割の調整）
│   ├── db/         テーブル定義と読み書き
│   ├── build_db.py             株価・銘柄・発行済株式数・セクターETF を DB に入れる
│   │                           （株数は yfinance。2015年秋より前は EDGAR の財務諸表の株数で埋める）
│   ├── build_fundamentals.py   財務諸表を DB に入れる
│   ├── build_sector_metrics.py セクターごとの指標（モメンタムなど）を計算する
│   └── build_stock_metrics.py  銘柄ごとの指標（値動き、ファンダ）を計算する
├── llm/            LLM に渡す情報、プロンプト、API 呼び出し
├── blm/            共分散、均衡リターン、Black-Litterman、最適化
├── run_reports.py  月次レポートを LLM に読ませ、見通しの数値と要約を取り出す
├── reports/        レポートの読み込み、取り出す項目の一覧、プロンプト、API 呼び出し
└── data/           SQLite の DB ファイル（Git には入れない）
    └── reports/    月次レポートの原本（macro/ と sector/。Git には入れない）
```

`llm` と `blm` は `marketdb` を使いますが、互いには使いません（DB を通してつながります）。

## 準備

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # OPENAI_API_KEY を入れる
```

設定はすべて `.env` に書きます。

`run_llm.py` は、`.env` の `LLM_BACKEND=local` で、API の代わりにこの PC の GPU でモデル（`LOCAL_MODEL`）を動かせます。
そのときは追加のライブラリを入れます（torch は CUDA 版）。月次レポート（`run_reports.py`）は常に OpenAI を使います。

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-local.txt
```

## 使い方

```bash
python marketdb/build_db.py             # DB を作る（取得済みのテーブルはスキップ。財務諸表もここで取得する）
python marketdb/build_fundamentals.py   # 財務諸表を追加する（銘柄を足したときなど）
python marketdb/build_sector_metrics.py # セクターごとの指標を計算する
python marketdb/build_stock_metrics.py  # 銘柄ごとの指標を計算する
python run_llm.py                       # LLM に聞く（回答は DB に保存）
python run_blm.py                       # 比率を計算する（結果は DB に保存）
python run_reports.py                   # 月次レポートを読ませる（結果は DB に保存）
```
