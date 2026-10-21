"""
Markdown of this project (headings, paragraphs, nested lists, pipe tables, code blocks, images, **bold**, `code`,
[links](url)) to LaTeX for LLM-BLM_全体の記録. No math is used in these files, so every special character is escaped.

    convert(text, top_level=1, image_dir=Path, copy_image=callable) -> str
top_level: the LaTeX level of '#' (0 = chapter, 1 = section, 2 = subsection).
"""
import re
from pathlib import Path

LEVELS = ['chapter', 'section', 'subsection', 'subsubsection', 'paragraph', 'paragraph']
SPECIAL = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '#': r'\#', '$': r'\$', '%': r'\%', '&': r'\&',
           '_': r'\_', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
SUP = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789−')


def esc(s):
    return ''.join(SPECIAL.get(c, c) for c in s)


def esc_code(s):
    # \code is \texttt: escape, and allow line breaks after / _ . -
    out = esc(s)
    for a in ['/', r'\_', '.', '-']:
        out = out.replace(a, a + r'\allowbreak{}')
    return out


def url_tex(u):
    """A url in typewriter type that may break after any character (long urls of the surveys overflow otherwise)."""
    return r'\texttt{' + ''.join(esc(c) + r'\allowbreak{}' for c in u) + '}'


URL = r'https?://[^\s<>（）()\[\]、。」]+'


def inline(s, long=40):
    """Inline markdown to LaTeX. ASCII runs of `long` characters or more (url pieces, arXiv numbers in a narrow
    table column) are set so that they may break anywhere."""
    parts = re.split(r'(`[^`]+`)', s)
    out = []
    for p in parts:
        if p.startswith('`') and p.endswith('`') and len(p) > 1:
            out.append(r'\code{' + esc_code(p[1:-1]) + '}')
            continue
        # links: keep the text and show the url after it; bare urls as they are. Both may break anywhere.
        urls = []

        def keep(u):
            urls.append(u)
            return '\x00U' + str(len(urls) - 1) + '\x00'
        p = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', lambda m: m.group(1) + ('（' + keep(m.group(2)) + '）'
                                                                       if m.group(2).startswith('http') else ''), p)
        p = re.sub(URL, lambda m: keep(m.group(0)), p)

        def breakable(m):
            t = m.group(0)
            a, b = len(t) - len(t.lstrip('*')), len(t) - len(t.rstrip('*'))     # keep **bold** marks outside
            core = t[a:len(t) - b]
            if len(core) < long or '*' in core or core.isalpha():     # plain words are left to hyphenation
                return t
            return t[:a] + keep(core) + t[len(t) - b:]
        p = re.sub(r'(?<![\x00\w])[!-~]{%d,}' % long, breakable, p)
        p = esc(p)
        p = re.sub('\x00U(\\d+)\x00', lambda m: url_tex(urls[int(m.group(1))]), p)
        # superscript digits (10⁻⁵): the fonts lack them
        p = re.sub('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+', lambda m: r'\textsuperscript{' + m.group(0).translate(SUP) + '}', p)
        p = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', p)
        p = re.sub(r'(?<![\\\w])\*(?!\s)(.+?)(?<!\s)\*', r'\\textit{\1}', p)
        p = p.replace('~~', '')
        out.append(p)
    return ''.join(out)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    header, body = cells[0], [r for r in cells[2:]]
    n = len(header)
    lengths = [max([len(header[i])] + [len(r[i]) if i < len(r) else 0 for r in body]) for i in range(n)]
    total = sum(max(4, x) for x in lengths)
    widths = [max(0.13 if n <= 5 else 0.09, min(0.6, max(4, x) / total)) for x in lengths]
    scale = 0.98 / sum(widths)
    # the column gaps (2 \tabcolsep each) are taken from each column, so the table is 0.98 of the line;
    # \hspace{0pt} lets TeX hyphenate the first word of a cell
    spec = ''.join(r'>{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr %.3f\linewidth-2\tabcolsep\relax}'
                   % (w * scale) for w in widths)
    lines = [r'{\small', r'\begin{longtable}{' + spec + '}', r'\toprule',
             ' & '.join(r'\textbf{' + inline(h) + '}' for h in header) + r' \\', r'\midrule', r'\endhead']
    for r in body:
        r = (r + [''] * n)[:n]
        lines.append(' & '.join(inline(c, long=14) for c in r) + r' \\')
    lines += [r'\bottomrule', r'\end{longtable}', '}']
    return '\n'.join(lines)


def convert(text, top_level=1, image_dir=None, copy_image=None, skip_first_heading=False):
    lines = text.splitlines()
    out, i = [], 0
    stack = []                     # open list environments: (indent, kind)
    first_heading_done = False

    def close_lists(to_indent=-1):
        while stack and stack[-1][0] > to_indent:
            out.append(r'\end{' + stack.pop()[1] + '}')

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('```'):
            close_lists()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith('```'):
                code.append(lines[j])
                j += 1
            out.append(r'\begin{lstlisting}[basicstyle=\ttfamily\scriptsize]')
            out += code
            out.append(r'\end{lstlisting}')
            i = j + 1
            continue
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            close_lists()
            if skip_first_heading and not first_heading_done:
                first_heading_done = True
                i += 1
                continue
            first_heading_done = True
            level = min(len(LEVELS) - 1, top_level + len(m.group(1)) - 1)
            star = '*' if level >= 3 else ''
            out.append('\\' + LEVELS[level] + star + '{' + inline(m.group(2).strip()) + '}')
            i += 1
            continue
        if line.strip().startswith('|') and i + 1 < len(lines) and re.match(r'^\s*\|?\s*:?-{2,}', lines[i + 1]):
            close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i])
                i += 1
            out.append(table(rows))
            continue
        m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', line.strip())
        if m:
            close_lists()
            if copy_image and image_dir is not None:
                name = copy_image(image_dir / m.group(2))
                out += [r'\begin{figure}[H]', r'\centering', r'\includegraphics[width=0.85\linewidth]{' + name + '}',
                        r'\caption{' + inline(m.group(1)) + '}', r'\end{figure}']
            i += 1
            continue
        m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', line)
        if m:
            indent = len(m.group(1).replace('\t', '    '))
            kind = 'enumerate' if m.group(2)[0].isdigit() else 'itemize'
            close_lists(indent)
            if not stack or stack[-1][0] < indent:
                out.append(r'\begin{' + kind + '}')
                stack.append((indent, kind))
            item = m.group(3)
            # continuation lines (indented, not a new item)
            j = i + 1
            while j < len(lines) and lines[j].strip() and not re.match(r'^\s*([-*]|\d+\.)\s+', lines[j]) \
                    and not lines[j].strip().startswith(('|', '#', '```')) and lines[j].startswith(' '):
                item += ' ' + lines[j].strip()
                j += 1
            out.append(r'\item ' + inline(item))
            i = j
            continue
        if not line.strip():
            close_lists()
            out.append('')
            i += 1
            continue
        if line.strip().startswith('>'):
            close_lists()
            out.append(r'\begin{quote}' + inline(line.strip().lstrip('>').strip()) + r'\end{quote}')
            i += 1
            continue
        close_lists()
        para = [line.strip()]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not re.match(r'^(#{1,6}\s|\s*([-*]|\d+\.)\s|\s*\||```|!\[|>)', lines[j]):
            para.append(lines[j].strip())
            j += 1
        out.append(inline(' '.join(para)))
        i = j
    close_lists()
    return '\n'.join(out) + '\n'
