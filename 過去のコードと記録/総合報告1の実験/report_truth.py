"""Ground truth for the items that can be read from the PDF text with a fixed pattern (new layout only)."""
import re, sys, sqlite3, glob, pandas as pd
from pypdf import PdfReader
num=lambda x: float(x.replace(',',''))
rows=[]
for f in sorted(glob.glob("data/reports/macro/*.pdf")):
    month=f[-11:-4]; r=PdfReader(f); t={}
    text="\n".join(re.sub(r"[ \t]+"," ",p.extract_text()) for p in r.pages[:10])
    m=re.search(r"S&P500 ([\d,]+) [+\-▲]?\d+% ([\d,]+) [+\-▲]?\d+% ([\d,]+) [+\-▲]?\d+% ([\d,]+) [+\-▲]?\d+% ([\d,]+) [+\-▲]?\d+%",text)
    if m: t.update(sp500_latest=num(m[3]),sp500_target_this_year=num(m[4]),sp500_target_next_year=num(m[5]))
    m=re.search(r"実績値 予想値\n(\d{4})年 (\d{4})年 (\d{4})年 (\d{4})年\n ?米国 (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)",text)
    if m:
        y=dict(zip(map(int,m.groups()[:4]),map(float,m.groups()[4:]))); yr=int(month[:4])
        if yr in y: t['us_gdp_growth_this_year']=y[yr]
        if yr+1 in y: t['us_gdp_growth_next_year']=y[yr+1]
    for k,v in t.items(): rows.append((month,k,v))
truth=pd.DataFrame(rows,columns=['report_month','item','truth'])
conn=sqlite3.connect("file:data/blm.sqlite?mode=ro",uri=True)
got=pd.read_sql("SELECT r.report_month, v.item, v.value_num FROM report_values v JOIN reports r USING(report_id) WHERE report_run_id=?",conn,params=(int(sys.argv[1]),))
d=truth.merge(got,how='inner',on=['report_month','item']); d['ok']=(d.truth-d.value_num).abs()<1e-6
print("reports with truth:",truth.report_month.nunique(),"| analysed:",got.report_month.nunique())
print(d.groupby('item').ok.agg(['sum','count']).to_string())
print(d[~d.ok].to_string())
