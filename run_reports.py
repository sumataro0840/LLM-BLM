"""
Register the monthly reports in data/reports in the database.
Settings are read from .env.

    python run_reports.py

Files are named <source>_<YYYY-MM>.pdf. A file registered before is skipped.
(The analysis by the LLM is not implemented yet.)
"""
from reports import config
from reports.loader import list_report_files, read_report_file
from marketdb.db.connection import connect, init_db
from marketdb.db.reports import load_reports, save_report

conn = connect(config.DB_PATH)
init_db(conn)

before = len(load_reports(conn))
for path in list_report_files(config.REPORTS_DIR):
    save_report(conn, **read_report_file(path))

reports = load_reports(conn)
print(f"reports: {len(reports)} (new {len(reports) - before})")
print(reports[['source', 'report_month', 'published_date', 'n_pages']].to_string())
