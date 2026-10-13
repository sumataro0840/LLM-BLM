"""
Register the monthly reports in data/reports and read them with the LLM; the answers are stored in the database.
Settings (months, pages, run to continue) are read from .env.

    python run_reports.py

Files are named <source>_<YYYY-MM>.pdf. What is read from a report is listed in reports/items.py.
Every report is saved as soon as it is answered, so an interrupted run can be continued by setting
REPORT_RUN_ID in .env.
"""
from reports import config
from reports.client import analyse_report, make_client
from reports.items import items_as_dicts
from reports.loader import first_pages, list_report_files, read_report_file
from reports.prompts import SYSTEM_PROMPT, USER_PROMPT, answer_schema, make_user_prompt
from marketdb.db.connection import connect, init_db
from marketdb.db.reports import analysed_reports, create_run, load_run, save_report, save_values

client = make_client(config.OPENAI_API_KEY)
conn = connect(config.DB_PATH)
init_db(conn)

paths = {}
for path in list_report_files(config.REPORTS_DIR):
    fields = read_report_file(path)
    report_id = save_report(conn, **fields)
    if config.REPORT_START_MONTH <= fields['report_month'] <= config.REPORT_END_MONTH:
        paths[report_id] = (path, fields['report_month'])

if config.REPORT_RUN_ID is None:
    run_id = create_run(conn, config.OPENAI_MODEL, SYSTEM_PROMPT + '\n\n' + USER_PROMPT, items_as_dicts(), config.REPORT_TEMPERATURE,
                        config.REPORT_RUN_NOTE)
    print(f"new run: REPORT_RUN_ID={run_id}")
else:
    run_id = config.REPORT_RUN_ID
    run = load_run(conn, run_id)
    # a run must hold answers to one question only
    current = {'model': config.OPENAI_MODEL, 'system_prompt': SYSTEM_PROMPT + '\n\n' + USER_PROMPT, 'items': items_as_dicts(),
               'temperature': config.REPORT_TEMPERATURE}
    changed = [key for key, value in current.items() if run[key] != value]
    if changed:
        raise SystemExit(f"REPORT_RUN_ID={run_id} was started with different settings: {changed}. "
                         "Clear REPORT_RUN_ID to start a new run.")
    print(f"continue run: REPORT_RUN_ID={run_id}")

done = analysed_reports(conn, run_id)
failed = []
for report_id, (path, report_month) in paths.items():
    if report_id in done:
        continue
    try:
        values, model_version, input_tokens = analyse_report(
            client, config.OPENAI_MODEL, SYSTEM_PROMPT, make_user_prompt(report_month), answer_schema(report_month), path.name,
            first_pages(path, config.REPORT_MAX_PAGES), config.REPORT_TEMPERATURE)
    except Exception as e:
        failed.append((path.name, str(e)[:200]))
        continue
    save_values(conn, run_id, report_id, values, model_version)
    answered = sum(value is not None for value in values.values())
    print(f"{path.name}: {answered}/{len(values)} items answered | input tokens {input_tokens}")

print(f"finished: REPORT_RUN_ID={run_id} | reports {len(paths)} | already done {len(done & set(paths))} | failed {failed}")
