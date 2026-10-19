"""
Fix 1 of 資料集/03_方針と文献調査/今後の計画/修正の実験方針.md: remove table fragments and boilerplate from a release text
before it is cut to a model's window. Rules are fixed here (regular expressions, no model, no stock data).

Removed, line by line:
  - table fragments: a line without any letter, a single word, or fewer than 4 words that do not end a sentence
    (lone numbers, '(0.9', ')', 'U.S.', 'Int'l', 'n/m', column headings);
  - boilerplate: lines about forward-looking statements / safe harbor / risk factors of the 10-K, conference call or
    webcast notices, contact lines (telephone numbers, e-mail addresses, 'Media:' / 'Investors:'), and the company
    description that follows an 'About <company>' heading.
Everything else (summary bullets, management quotes, segment commentary, outlook and guidance) is kept in order.
"""
import re

LETTER = re.compile(r'[A-Za-z]')
SENTENCE_END = re.compile(r'[.!?:;]["”’)]?\s*$')
BOILERPLATE = re.compile(
    r'forward[- ]looking statement|safe harbor|private securities litigation reform act|undertakes? no (obligation|duty)'
    r'|risk factors|conference call|webcast|dial[- ]in|replay|investor relations website|for further information'
    r'|^\s*(media|investors?|investor contact|media contact|contacts?)\s*:', re.I)
CONTACT = re.compile(r'\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+|www\.\S+\.(com|net|org)\b', re.I)
ABOUT = re.compile(r'^\s*about\s+[A-Z][\w&.,\'’ -]{0,60}:?\s*$', re.I)


def is_fragment(line):
    s = line.strip()
    if not LETTER.search(s):
        return True
    words = len(s.split())
    return words < 2 or (words < 4 and not SENTENCE_END.search(s))


def clean(text):
    """The text with table fragments and boilerplate lines removed (lines joined with newlines)."""
    out, skip_next = [], False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if ABOUT.match(s):
            skip_next = True          # the paragraph after 'About X:' is the standard company description
            continue
        if skip_next:
            skip_next = False
            if not is_fragment(s):
                continue
        if is_fragment(s) or BOILERPLATE.search(s) or CONTACT.search(s):
            continue
        out.append(s)
    return '\n'.join(out)
