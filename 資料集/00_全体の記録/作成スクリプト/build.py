"""
Builds LLM-BLM_全体の記録.tex: the new chapters (章/*.tex, written for this book) and the chapters taken from every
document of 資料集 (総合報告書 1・2, 実験 1〜11 and the supplementary experiments, the two 解説, the plans and surveys,
the design notes), in one systematic order.

    python 資料集/00_全体の記録/作成スクリプト/build.py
    cd 資料集/00_全体の記録 && tectonic -X compile LLM-BLM_全体の記録.tex

What the import does to each chapter taken from another document:
  - labels and references get the document's prefix (no clash between documents); a reference to a chapter that
    was not taken is replaced by a short note;
  - figures and the generated tables are copied to 取り込み/ and referred to by that path;
  - \\part, the title page and the table of contents of the source are dropped; \\chapter* becomes a numbered chapter;
  - a note of the source, and for experiments a box of inputs and outputs (io.py), is put under the heading.
The sources are not changed.
"""
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOOK = HERE.parent
SRC = BOOK.parent                      # 資料集
REPO = SRC.parent
sys.path.insert(0, str(HERE))
import md2tex                          # noqa: E402
from io_boxes import IO                # noqa: E402
from order import ORDER                # noqa: E402

IMP = BOOK / '取り込み'

DOCS = {
    'r1': dict(path='01_総合報告書/総合報告1/LLM-BLM_総合報告書.tex', figs=['LLM-BLM_総合報告書_図'],
               src='総合報告書 1（2026-10-13）'),
    'r2': dict(path='01_総合報告書/総合報告2/LLM-BLM_総合報告書2.tex', figs=['LLM-BLM_総合報告書2_図'],
               src='総合報告書 2（2026-10-18）'),
    'hyp': dict(path='02_実験/実験1_仮説の検証/LLM-BLM_実験仮説.tex', figs=[], src='実験仮説（実験 1 の設計書、2026-10-15）'),
    'hres': dict(path='02_実験/実験1_仮説の検証/LLM-BLM_実験結果レポート.tex', figs=['LLM-BLM_実験結果レポート_図'],
                 src='実験結果レポート（実験 1、2026-10-17）'),
    'e2': dict(path='02_実験/実験2_決算発表文/LLM-BLM_実験2.tex', figs=['図'], src='実験 2 のレポート（2026-10-18）'),
    'e3': dict(path='02_実験/実験3_アナリスト予想/LLM-BLM_実験3.tex', figs=['図'], src='実験 3 のレポート（2026-10-18）'),
    'e4': dict(path='02_実験/実験4_FinBERT追加学習/LLM-BLM_実験4.tex', figs=['図'], src='実験 4 のレポート（2026-10-18）'),
    'e5': dict(path='02_実験/実験5_ChronoGPTの埋め込み/LLM-BLM_実験5.tex', figs=['図'],
               src='実験 5・10 のレポート（2026-10-19）'),
    'e78': dict(path='02_実験/実験7・8_月次と日次のBLM/LLM-BLM_実験7・8.tex', figs=['図'],
                src='実験 7・8・9 のレポート（2026-10-19）'),
    'e11': dict(path='02_実験/実験11_読ませ方とまとめ方の修正/LLM-BLM_実験11.tex', figs=['図'],
                src='実験 11 のレポート（2026-10-21）'),
    'xte': dict(path='02_実験/補足実験_共分散とTE/LLM-BLM_補足実験_共分散とTE.tex', figs=['図'],
                src='補足実験 X02〜X04 のレポート（2026-10-18）'),
    'x05': dict(path='02_実験/補足実験_αの大きさ/LLM-BLM_補足実験_αの大きさ.tex', figs=['図'],
                src='補足実験 X05 のレポート（2026-10-19）'),
    'sum': dict(path='03_方針と文献調査/まとめ/LLM-BLM_文献調査と設計方針.tex', figs=[], article=True,
                src='文献調査と設計方針のまとめ（2026-10-17 朝）'),
    'blm': dict(path='05_解説/BLM解説/BLM解説.tex', figs=['BLM解説_生成'], tabs='BLM解説_生成',
                src='解説「平均分散から Black-Litterman へ」（2026-10-14）'),
    'tips': dict(path='05_解説/BLMビューとリスクのコツ/BLMビューとリスクのコツ.tex', figs=['生成'], tabs='生成',
                 chapters_dir='章', src='解説「BLM のビューとリスクのコツ」（2026-10-14）'),
}
REF_CMDS = 'label|ref|eqref|pageref|figref|tabref|chapref|secref|appref|autoref|nameref'


# ---------- reading the sources ----------

def braced(s, start):
    """(content, end index) of the {...} group that starts at s[start] == '{'."""
    depth = 0
    for k in range(start, len(s)):
        if s[k] == '{':
            depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0:
                return s[start + 1:k], k + 1
    raise ValueError('unbalanced braces')


def body_of(doc):
    d = DOCS[doc]
    text = (SRC / d['path']).read_text(encoding='utf-8')
    if d.get('chapters_dir'):
        files = re.findall(r'\\input\{' + d['chapters_dir'] + r'/([^}]+)\}', text.split('\\begin{document}')[1])
        folder = (SRC / d['path']).parent / d['chapters_dir']
        return '\n'.join((folder / (f + '.tex')).read_text(encoding='utf-8') for f in files)
    body = text.split('\\begin{document}', 1)[1].split('\\end{document}')[0]
    return body


def chapters(doc):
    """[(title, text)] of the chapters of a document (text starts with the \\chapter line)."""
    body = body_of(doc)
    if DOCS[doc].get('article'):
        start = body.index('\\section')
        return [('__all__', body[start:])]
    idx = [m.start() for m in re.finditer(r'^\\chapter\*?\{', body, re.M)]
    out = []
    for a, b in zip(idx, idx[1:] + [len(body)]):
        block = body[a:b]
        title, _ = braced(block, block.index('{'))
        out.append((title, block))
    return out


_cache = {}


def chapter(doc, title):
    if doc not in _cache:
        _cache[doc] = chapters(doc)
    hits = [t for t in _cache[doc] if t[0] == title or t[0].startswith(title)]
    if not hits:
        raise SystemExit(f'{doc}: no chapter starting with {title!r}; have {[t[0][:30] for t in _cache[doc]]}')
    return hits[0]


# ---------- transforming a chapter ----------

def find_fig(doc, name):
    d = DOCS[doc]
    base = (SRC / d['path']).parent
    for folder in d['figs'] + ['']:
        for ext in ['', '.pdf', '.png', '.jpg']:
            p = base / folder / (name + ext)
            if p.is_file():
                return p
    raise SystemExit(f'{doc}: figure {name} not found')


def copy_fig(doc, path):
    target = IMP / '図' / doc / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
        shutil.copy2(path, target)
    return target.relative_to(BOOK).as_posix()


def tikz_of(doc):
    """The \\tikzset{...} of the source's preamble (the styles box, arr, ... differ between documents)."""
    pre = (SRC / DOCS[doc]['path']).read_text(encoding='utf-8').split('\\begin{document}')[0]
    k = pre.find('\\tikzset{')
    if k < 0:
        return ''
    inner, _ = braced(pre, k + len('\\tikzset'))
    return '\\tikzset{' + inner + '}\n'


def transform(doc, text, title=None, inject=''):
    # drop things that belong to the source as a whole
    text = re.sub(r'^\\part\*?\{.*\}.*$', '', text, flags=re.M)
    text = re.sub(r'^\\appendix\s*$', '', text, flags=re.M)
    text = re.sub(r'^.*\\(tableofcontents|maketitle|listoffigures|listoftables)\b.*$', '', text, flags=re.M)
    text = re.sub(r'^\\setcounter\{tocdepth\}\{\d\}\s*$', '', text, flags=re.M)
    text = re.sub(r'^\s*\{\\small\\tableofcontents\}\s*$', '', text, flags=re.M)
    # labels and references
    text = re.sub(r'\\(' + REF_CMDS + r')\{([^}]*)\}', lambda m: '\\' + m.group(1) + '{' + doc + ':' + m.group(2) + '}', text)
    text = re.sub(r'(\\begin\{lstlisting\}\[[^\]]*?label=)\{?([^,\]}]+)\}?',
                  lambda m: m.group(1) + '{' + doc + ':' + m.group(2) + '}', text)
    # figures and generated tables
    text = re.sub(r'\\includegraphics(\[[^\]]*\])?\{([^}]+)\}',
                  lambda m: '\\includegraphics' + (m.group(1) or '') + '{' + copy_fig(doc, find_fig(doc, m.group(2))) + '}', text)
    text = text.replace('\\tabin{', '\\tabinD{' + doc + '}{')
    # heading
    m = re.search(r'\\chapter(\*?)\{', text)
    if m:
        old, end = braced(text, m.end() - 1)
        new = title or old
        rest = text[end:]
        if m.group(1):
            rest = re.sub(r'^\s*\\addcontentsline\{toc\}\{chapter\}\{[^\n]*\}\s*\n', '\n', rest, count=1)
        lab = re.match(r'\s*(\\label\{[^}]*\})', rest)
        label = lab.group(1) if lab else ''
        if lab:
            rest = rest[lab.end():]
        text = text[:m.start()] + '\\chapter{' + new + '}' + label + '\n' + inject + rest
    elif inject:
        text = inject + text
    # the source's tikz styles for this chapter, and the book's styles again after it
    return tikz_of(doc) + text + '\n\\booktikz\n'


def srcnote(doc, extra=''):
    return '\\srcnote{出典：' + DOCS[doc]['src'] + (('。' + extra) if extra else '') + '}\n'


# ---------- markdown ----------

def md_chapter(path, title, level=0, note=None, io=''):
    """A markdown file as a chapter (level 0) or a section (level 1). Its own '# title' is replaced by `title`."""
    p = REPO / path
    text = p.read_text(encoding='utf-8').replace('​', '')      # zero-width spaces of a PDF conversion
    for a, b in {'►': '▶', '✗': '×', 'ᵀ': '^T'}.items():   # glyphs no font of the book has
        text = text.replace(a, b)
    images = IMP / '図' / 'md' / re.sub(r'[\\/:]', '_', path.rsplit('.', 1)[0])

    def copy_image(img):
        target = images / img.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, target)
        return target.relative_to(BOOK).as_posix()
    first = re.search(r'^(#{1,6})\s', text, re.M)
    has_title = bool(first) and first.group(1) == '#'
    n_h1 = len(re.findall(r'^#\s', text, re.M))
    top = level if n_h1 <= 1 else level + 1      # '##' becomes the level under the chapter or section
    body = md2tex.convert(text, top_level=top, image_dir=p.parent, copy_image=copy_image,
                          skip_first_heading=has_title and n_h1 <= 1)
    head = ('\\chapter{' if level == 0 else '\\section{') + title + '}\n'
    note = note or ('出典：' + path)
    return head + '\\srcnote{' + md2tex.esc(note) + '}\n' + io + body


# ---------- chapters made of several pieces ----------

def demote(text):
    """Every heading one level down (a chapter of a source becomes a section of a chapter of the book)."""
    for a, b in [('subsubsection', 'paragraph'), ('subsection', 'subsubsection'), ('section', 'subsection'),
                 ('chapter', 'section')]:
        text = re.sub(r'\\' + a + r'(\*?)\{', lambda m: '\\' + b + m.group(1) + '{', text)
    return text


def piece(item, level):
    """The text of one ORDER item or one piece of a 'chap' item, as a chapter (level 0) or a section (level 1)."""
    kind = item[0]
    opts = item[3] if len(item) > 3 else {}
    if kind == 'imp':
        doc, title = item[1], item[2]
        if title == '__all__':
            text = '\\chapter{' + opts['title'] + '}\n' + chapters(doc)[0][1]
        else:
            _, text = chapter(doc, title)
        inject = srcnote(doc, opts.get('note', '')) + IO.get(opts.get('io', ''), '')
        text = transform(doc, text, title=opts.get('title'), inject=inject)
        return demote(text) if level else text
    if kind == 'md':
        return md_chapter(item[1], item[2], level=level, note=opts.get('note'), io=IO.get(opts.get('io', ''), ''))
    if kind == 'tex':
        return item[1] + '\n'
    raise SystemExit(f'unknown piece {item}')


# ---------- references ----------

def fix_references(body):
    """In the imported files: a reference to a label that was not taken becomes a short note, and a reference to a
    chapter, a section or an appendix is written as what the label is in this book (a chapter of a source may have
    become a section, a numbered chapter an appendix). Returns the labels not taken."""
    files, in_app = [], False
    for line in body.splitlines():
        if line.startswith('\\appendix'):
            in_app = True
        m = re.match(r'\\input\{(.+)\}', line)
        if m:
            files.append((BOOK / (m.group(1) + '.tex'), in_app))
    texts = {f: f.read_text(encoding='utf-8') for f, _ in files}
    labels, kind_of = set(), {}
    for f, app in files:
        t = texts[f]
        labels |= set(re.findall(r'\\label\{([^}]*)\}', t)) | set(re.findall(r'label=\{?([^,\]}]+)', t))
        for m in re.finditer(r'\\(chapter|section)\*?\{', t):
            _, end = braced(t, m.end() - 1)
            lab = re.match(r'\s*\\label\{([^}]*)\}', t[end:])
            if lab:
                kind_of[lab.group(1)] = ('appref' if app else 'chapref') if m.group(1) == 'chapter' else 'secref'
    missing = set()

    def as_kind(m):
        lab = m.group('lab') or m.group('lab2') or m.group('lab3')
        if lab not in labels:
            missing.add(lab)
            return '（元の資料の別の箇所）'
        if lab in kind_of:
            return '\\' + kind_of[lab] + '{' + lab + '}'
        return m.group(0)

    def fix(m):
        cmd, lab = m.group(1), m.group(2)
        if cmd == 'label':
            return m.group(0)
        if lab not in labels:
            missing.add(lab)
            return '（元の資料の別の箇所）'
        if cmd in ('chapref', 'appref', 'secref') and lab in kind_of:
            return '\\' + kind_of[lab] + '{' + lab + '}'
        return m.group(0)
    for f, _ in files:
        if IMP not in f.parents:
            continue
        t = texts[f]
        # chapters and appendices referred to by hand ("第~\ref{x}~章", "付録~\ref{x}", "\ref{x}~節")
        t2 = re.sub(r'第\s*~?\s*\\ref\{(?P<lab>[^}]*)\}\s*~?\s*章|付録\s*~?\s*\\ref\{(?P<lab2>[^}]*)\}'
                    r'|\\ref\{(?P<lab3>[^}]*)\}\s*~?\s*節',
                    as_kind, t)
        t2 = re.sub(r'\\(' + REF_CMDS + r')\{([^}]*)\}', fix, t2)
        if t2 != t:
            f.write_text(t2, encoding='utf-8')
    return missing


# ---------- assembling ----------

def build():
    if IMP.exists():
        shutil.rmtree(IMP)
    (IMP / '本文').mkdir(parents=True)
    for doc, d in DOCS.items():
        if d.get('tabs'):
            src = (SRC / d['path']).parent / d['tabs']
            dst = IMP / '表' / doc
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob('*.tex'):
                shutil.copy2(f, dst / f.name)
    parts, n = [], 0
    for item in ORDER:
        kind = item[0]
        if kind == 'part':
            parts.append('\\part{' + item[1] + '}\n')
        elif kind == 'appendix':
            # more than 26 appendices: A1, A2, ... instead of A..Z
            parts.append('\\appendix\n\\renewcommand{\\thechapter}{A\\arabic{chapter}}\n')
        elif kind == 'new':
            parts.append('\\input{章/' + item[1] + '}\n')
        else:
            if kind == 'chap':
                # ('chap', title, [pieces], {label, intro, io}): one chapter of the book, each piece a section
                _, title, pieces = item[:3]
                opts = item[3] if len(item) > 3 else {}
                text = '\\chapter{' + title + '}' + ('\\label{' + opts['label'] + '}' if opts.get('label') else '') + '\n'
                text += opts.get('intro', '') + '\n' + IO.get(opts.get('io', ''), '')
                text += '\n'.join(piece(p, 1) for p in pieces)
                tag = 'chap'
            elif kind in ('imp', 'impsec', 'md'):
                text = piece(('imp' if kind == 'impsec' else kind,) + tuple(item[1:]), 1 if kind == 'impsec' else 0)
                tag = item[1] if kind != 'md' else 'md'
            else:
                raise SystemExit(f'unknown item {item}')
            n += 1
            name = f'{n:03d}_{tag}.tex'
            (IMP / '本文' / name).write_text(text, encoding='utf-8')
            parts.append('\\input{取り込み/本文/' + name[:-4] + '}\n')
    body = ''.join(parts)
    missing = fix_references(body)
    preamble = (HERE / 'preamble.tex').read_text(encoding='utf-8')
    master = preamble.replace('%%BODY%%', body)
    (BOOK / 'LLM-BLM_全体の記録.tex').write_text(master, encoding='utf-8')
    print(f'{n} imported chapters, {len(missing)} references to chapters not taken')


if __name__ == '__main__':
    build()
