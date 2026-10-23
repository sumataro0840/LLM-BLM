"""Build 章/文献.tex (thebibliography) from 照合/文献の一覧.md, keeping only the keys cited in 章/*.tex.

Run in this folder:  python 作成スクリプト/文献.py
Each line of the list is  - [key] Authors (Year). "Title". Venue.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LINE = re.compile(r'^- \[(?P<key>[^\]]+)\] (?P<authors>.+?) \((?P<year>\d{4})\)\. "(?P<title>.+)"\. (?P<venue>.+?)\.?$')
SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}


def tex(s):
    return "".join(SPECIAL.get(c, c) for c in s)


def main():
    entries = {}
    for line in (HERE / "照合" / "文献の一覧.md").read_text(encoding="utf-8").splitlines():
        m = LINE.match(line.strip())
        if m:
            entries[m["key"]] = m.groupdict()
    cited = set()
    for path in sorted((HERE / "章").glob("*.tex")):
        for group in re.findall(r"\\cite\{([^}]+)\}", path.read_text(encoding="utf-8")):
            cited.update(k.strip() for k in group.split(","))
    missing = sorted(cited - entries.keys())
    unused = sorted(entries.keys() - cited)
    if missing:
        raise SystemExit(f"cited but not in the list: {missing}")
    keys = sorted(cited, key=lambda k: (entries[k]["authors"].lower(), entries[k]["year"]))
    out = ["% 作成スクリプト/文献.py が 照合/文献の一覧.md から作る。手で直さない。",
           r"\begin{thebibliography}{999}", r"\small", r"\setlength{\itemsep}{0.15em}"]
    for k in keys:
        e = entries[k]
        out.append(rf"\bibitem{{{k}}} {tex(e['authors'])} ({e['year']}). ``{tex(e['title'])}.'' \textit{{{tex(e['venue'])}}}.")
    out.append(r"\end{thebibliography}")
    (HERE / "章" / "文献.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(keys)} cited entries written; {len(unused)} listed but not cited: {', '.join(unused)}")


if __name__ == "__main__":
    main()
