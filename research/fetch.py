"""Yahoo 일봉(수정종가) 전체 기간 받기 → yahoo/<sym>.json"""
import json, sys, urllib.request, time
from pathlib import Path
OUT = Path(sys.argv[1])
syms = sys.argv[2:]
for s in syms:
    y = s + ".KS" if s[0].isdigit() else s
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{y}?period1=0&period2={int(time.time())}&interval=1d&events=split"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=30))["chart"]["result"][0]
        ts = d["timestamp"]; adj = d["indicators"]["adjclose"][0]["adjclose"]
        rows = [(t, c) for t, c in zip(ts, adj) if c]
        json.dump(rows, open(OUT / f"{s}.json", "w"))
        print(s, len(rows), time.strftime("%Y-%m-%d", time.gmtime(rows[0][0])))
    except Exception as e:
        print(s, "ERR", e)
    time.sleep(0.5)
