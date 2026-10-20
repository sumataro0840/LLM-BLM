"""
Fix 1 of 資料集/03_方針と文献調査/今後の計画/修正の実験方針.md: remove table fragments and boilerplate from a release text
before it is cut to a model's window. Rules are fixed here (regular expressions, no model, no stock data).

Removed, line by line:
  - table fragments: a line without any letter, a single word, or fewer than 4 words that do not end a sentence
    (lone numbers, '(0.9', ')', 'U.S.', 'Int'l', 'n/m', column headings);
  - a safe-harbor paragraph: a line of at most 2,000 characters that mentions forward-looking statements, the safe
    harbor, the Private Securities Litigation Reform Act or 'undertakes no obligation';
  - the company description: the paragraph that follows an 'About <company>' heading;
  - in every other line, only the sentences (or bullet items) that are boilerplate: conference call and webcast
    notices, contact details (telephone numbers, e-mail addresses, web addresses, 'Media:' / 'Investors:'),
    references to the risk factors of the 10-K.
Everything else (summary bullets, management quotes, segment commentary, outlook and guidance) is kept in order.

Revised on 2026-10-20 before any score was evaluated (preregistration.md, 追記 2): the first version dropped a whole
line when any boilerplate word or contact appeared in it, and some releases put a whole summary on one line
(e.g. a press-contact e-mail on the same line as 'raises full year earnings guidance').
"""
import re

LETTER = re.compile(r'[A-Za-z]')
SENTENCE_END = re.compile(r'[.!?:;]["”’)]?\s*$')
SAFE_HARBOR = re.compile(r'forward[- ]looking statement|safe harbor|private securities litigation reform act'
                         r'|undertakes? no (obligation|duty)', re.I)
BOILERPLATE = re.compile(
    r'forward[- ]looking statement|safe harbor|private securities litigation reform act|undertakes? no (obligation|duty)'
    r'|risk factors|conference call|webcast|dial[- ]in|replay|investor relations website|for further information'
    r'|^\s*(media|investors?|investor contact|media contact|contacts?)\s*:', re.I)
CONTACT = re.compile(r'\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+|www\.\S+\.(com|net|org)\b', re.I)
ABOUT = re.compile(r'^\s*about\s+[A-Z][\w&.,\'’ -]{0,60}:?\s*$', re.I)
UNIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9“"(•·▪])|\s+(?=[•·▪]\s)')
MAX_PARAGRAPH = 2000


def is_fragment(line):
    s = line.strip()
    if not LETTER.search(s):
        return True
    words = len(s.split())
    return words < 2 or (words < 4 and not SENTENCE_END.search(s))


def is_boilerplate(unit):
    return bool(BOILERPLATE.search(unit) or CONTACT.search(unit))


def clean_line(s):
    """The line without its boilerplate sentences ('' when nothing is left)."""
    if len(s) <= MAX_PARAGRAPH and SAFE_HARBOR.search(s):
        return ''
    if not is_boilerplate(s):
        return s
    units = [u.strip() for u in UNIT.split(s) if u.strip()]
    return ' '.join(u for u in units if not is_boilerplate(u))


def clean(text):
    """The text with table fragments and boilerplate removed (lines joined with newlines)."""
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
        if is_fragment(s):
            continue
        s = clean_line(s)
        if s and not is_fragment(s):
            out.append(s)
    return '\n'.join(out)
