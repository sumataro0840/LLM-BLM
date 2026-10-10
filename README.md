# LLM-BLM

論文 [Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model](https://arxiv.org/abs/2504.14345) の再現用リポジトリです。
元の実装（[youngandbin/LLM-BLM](https://github.com/youngandbin/LLM-BLM)）をもとに、OpenAI API だけを使う形に作り直しています。

今あるのは、**S&P500 のデータを DB に入れる部分**と、**LLM に予想（ビュー）を聞く部分**です。
Black-Litterman の計算と評価は、まだ入っていません。

## 構成

```
.
├── run.py          LLM に各銘柄の予想リターンを聞く
├── marketdb/       データの取得・加工・DB
│   ├── config.py
│   ├── sources/    取得（Wikipedia、yfinance、SEC EDGAR）
│   ├── transform/  加工（在籍期間、株式分割の調整）
│   ├── db/         テーブル定義と読み書き
│   ├── build_db.py             株価・銘柄・発行済株式数を DB に入れる
│   └── build_fundamentals.py   財務諸表を DB に入れる
├── llm/            LLM に渡す情報、プロンプト、API 呼び出し、出力
└── data/           SQLite の DB ファイル（Git には入れない）
```

`llm` は `marketdb` を使いますが、`marketdb` は `llm` を使いません。

## 準備

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # OPENAI_API_KEY を入れる
```

設定はすべて `.env` に書きます。

## 使い方

```bash
python marketdb/build_db.py             # DB を作る（取得済みのテーブルはスキップ）
python marketdb/build_fundamentals.py   # 財務諸表を追加する
python run.py                           # LLM に聞く（OUT_DIR に CSV と JSON を出力）
```
