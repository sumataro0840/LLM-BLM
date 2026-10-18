import json, sys, sqlite3, numpy as np, pandas as pd
sys.path.insert(0,'.')
from marketdb.db.sector import load_sector_metrics
conn=sqlite3.connect(f"file:{sys.argv[1]}?mode=ro",uri=True)
sc=json.load(open(sys.argv[2]))
rows=[]
for r in sc:
    z=load_sector_metrics(conn,r['as_of'],'z_score'); v=load_sector_metrics(conn,r['as_of'])
    mom=pd.DataFrame(r['tables'][4],index=['3m','6m','12m','total']).T; vol=pd.DataFrame(r['tables'][5],index=['rv','iv','pct','chg']).T
    val=pd.DataFrame(r['tables'][1],index=['pe','fpe','pb','ps','total']).T
    mine_total=z[['momentum_3m','momentum_6m','momentum_12m']].mean(axis=1)
    c=lambda a,b: np.corrcoef(a[mom.index],b[mom.index])[0,1]
    sp=lambda a,b: a[mom.index].rank().corr(b[mom.index].rank())
    rows.append({'as_of':r['as_of'],'mom3':c(z.momentum_3m,mom['3m']),'mom6':c(z.momentum_6m,mom['6m']),'mom12':c(z.momentum_12m,mom['12m']),
                 'mom_total':c(mine_total,mom['total']),'vol':c(v.volatility_30d,vol['rv']),'vol_level':(v.volatility_30d*100/vol['rv']).mean(),
                 'vol_vs_iv':c(v.volatility_30d,vol['iv']),
                 'val_pe(rank)':sp(-v.pe_trailing,val['pe']),'val_pb(rank)':sp(-v.pb,val['pb']),'val_ps(rank)':sp(-v.ps,val['ps'])})
d=pd.DataFrame(rows).set_index('as_of'); print(d.round(2).to_string()); print(d.mean().round(3).to_string())
print(load_sector_metrics(conn,'2026-05-29').round(3).to_string())
print(pd.read_sql("SELECT metric, COUNT(*) n, COUNT(DISTINCT as_of_date) dates, MIN(as_of_date) first, MIN(value) lo, MAX(value) hi FROM sector_metrics GROUP BY 1",conn).to_string())
print(pd.read_sql("SELECT as_of_date, gics_sector, value FROM sector_metrics WHERE metric='valuation_coverage' ORDER BY value LIMIT 5",conn).to_string())
print(pd.read_sql("SELECT metric, ROUND(AVG(z_score),6) m, ROUND(AVG(z_score*z_score),6) v FROM sector_metric_scores WHERE as_of_date='2026-05-29' GROUP BY 1",conn).to_string())
