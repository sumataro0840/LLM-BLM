"""Parse State Street sector scorecards: every table of 11 sector rows, in page order."""
import re, sys, glob, json
from pypdf import PdfReader
JP = {'コミュニケーション・サービス':'Communication Services','一般消費財':'Consumer Discretionary','生活必需品':'Consumer Staples','エネルギー':'Energy','金融':'Financials','ヘルスケア':'Health Care','資本財':'Industrials','情報技術':'Information Technology','素材':'Materials','不動産':'Real Estate','公益事業':'Utilities'}
LINE = re.compile(r'^(.*?)\s*(-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)*)\s*$')
def tables(pdf):
    out, cur = [], {}
    for page in pdf.pages:
        for line in page.extract_text().split('\n'):
            m = LINE.match(line)
            name = re.sub(r'\s', '', m.group(1)) if m else None
            if name in JP:
                cur[JP[name]] = [float(x) for x in m.group(2).split()]
                if len(cur) == 11: out.append(cur); cur = {}
            elif cur and name != 'S&P500':
                cur = {}
    return out
def parse(path):
    pdf = PdfReader(path)
    text = re.sub(r'\s', '', ''.join(p.extract_text() for p in pdf.pages))
    y, m, d = map(int, re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日時点', text).groups())
    return {'file': path.split('/')[-1], 'as_of': f'{y}-{m:02d}-{d:02d}', 'created': pdf.metadata.creation_date.date().isoformat(), 'tables': tables(pdf)}
if __name__ == '__main__':
    out = [parse(f) for f in sorted(glob.glob(sys.argv[1] + '/*.pdf'))]
    out.sort(key=lambda r: r['as_of'])
    for r in out:
        print(r['file'], r['as_of'], r['created'], [len(t['Energy']) for t in r['tables']])
    for t in out[3]['tables']: print('  Energy', t['Energy'], '| IT', t['Information Technology'])
    json.dump(out, open('scorecards.json', 'w'))
