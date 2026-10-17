"""
Release texts: where they are kept (data/cache/edgar/{cik}/{accession without dashes}.txt) and their hash.
"""
import hashlib
import re

from events import config

MAIN_TYPES = ("EX-99.1", "EX-99", "EX-99.01")


def path_for(cik, accession):
    return config.TEXT_DIR / str(int(cik)) / f"{accession.replace('-', '')}.txt"


def relative(path):
    """Path relative to the repository, with forward slashes (the same on every OS)."""
    return path.resolve().relative_to(config.REPO).as_posix()


def absolute(text_path):
    return config.REPO / text_path


def read(text_path, max_chars=None):
    text = absolute(text_path).read_text(encoding="utf-8")
    return text[:max_chars] if max_chars else text


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def doc_type(text):
    """Exhibit type written at the top of an EDGAR document, e.g. 'EX-99.1' (None when absent)."""
    m = re.match(r"\s*(EX-[\d.A-Z]+)", text[:100])
    return m.group(1) if m else None
