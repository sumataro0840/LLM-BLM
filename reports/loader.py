"""
Find the report files in data/reports and read what is registered in the database.
"""
import hashlib
import io
import re

from pypdf import PdfReader, PdfWriter

FILE_NAME = re.compile(r"(?P<source>.+)_(?P<month>\d{4}-\d{2})\.pdf")


def list_report_files(reports_dir):
    """PDF files named <source>_<YYYY-MM>.pdf, oldest first; any other PDF name is an error."""
    paths = sorted(reports_dir.glob("*.pdf"))
    unknown = [p.name for p in paths if not FILE_NAME.fullmatch(p.name)]
    if unknown:
        raise ValueError(f"rename to <source>_<YYYY-MM>.pdf: {unknown}")
    return paths


def read_report_file(path):
    """Fields of the reports table for one file."""
    name = FILE_NAME.fullmatch(path.name)
    pdf = PdfReader(path)
    return {
        'source': name['source'],
        'report_month': name['month'],
        # the file is created shortly before it is published
        'published_date': pdf.metadata.creation_date.date().isoformat(),
        'file_name': path.name,
        'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'n_pages': len(pdf.pages),
    }


def first_pages(path, n_pages):
    """The first n_pages of the file as a new PDF (bytes): the summary, the forecast tables and the US pages
    come first, and the whole report is too long to send."""
    writer = PdfWriter()
    for page in PdfReader(path).pages[:n_pages]:
        writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
