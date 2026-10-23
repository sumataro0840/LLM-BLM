"""Summarize 照合/照合の結果.txt (output of LLM/refcheck.py) into 章/A2_照合の集計.tex.

Run in this folder after refcheck:  python 作成スクリプト/照合の集計.py
Manual notes for entries that did not match cleanly are kept in NOTES below (checked by hand).
"""
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
KEY = re.compile(r"\[([^\]]+)\]")
SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\^{}"}

# key -> what was checked by hand when the automatic match was weak or wrong
NOTES = {
    "hlz2016": "題名が短く、別の論文（Lewellen）に一致した。RFS 29(1) の論文として実在を確かめた",
}


def tex(s):
    return "".join(SPECIAL.get(c, c) for c in s)


def main():
    lines = (HERE / "照合" / "照合の結果.txt").read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        m = re.match(r"^(ok|\?\?|NG|src|--|題名\?|著者\?) (.*)$", line)
        if not m:
            continue
        mark, body = m.groups()
        detail = lines[i + 1].strip() if i + 1 < len(lines) and lines[i + 1].startswith("     ") else ""
        key = KEY.search(body) or KEY.search(detail)
        rows.append((mark, key.group(1) if key else "?", body, detail))
    count = Counter(r[0] for r in rows)
    out = [f"照合した {len(rows)} 本のうち、題名がほぼ一致した（ok）ものが {count['ok']} 本、"
           f"一致が弱い（??）ものが {count['??']} 本、見つからない（NG）ものが {count['NG']} 本、"
           f"照合できなかった（--）ものが {count['--']} 本だった。"]
    weak = [r for r in rows if r[0] != "ok" or r[1] in NOTES]
    if weak:
        out += ["", "ok でなかったもの、ok でも自分で確かめ直したものを次に示す。", "",
                r"{\footnotesize", r"\begin{longtable}{@{}cP{0.17\linewidth}P{0.39\linewidth}P{0.33\linewidth}@{}}",
                r"\toprule 結果 & キー & 照合の候補 & 自分で確かめた結果 \\ \midrule \endhead \bottomrule \endfoot"]
        for mark, key, body, detail in weak:
            cand = detail.lstrip("→ ").strip() or body
            out.append(rf"{tex(mark)} & \code{{{tex(key)}}} & {tex(cand[:160])} & {tex(NOTES.get(key, '（要確認）'))} \\")
        out += [r"\end{longtable}", "}"]
    (HERE / "章" / "A2_照合の集計.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(count, "weak:", [(r[0], r[1]) for r in weak])


if __name__ == "__main__":
    main()
