"""3차 — 레버리지에서 '복리 수익'이 늘었나. 비교 대상은 같은 평균 비중을 매일 유지한 경우.

레버리지 ETF 는 비중만 줄여도(=레버리지 낮춤) 변동성 감쇠가 줄어 수익이 늘 수 있다.
그래서 '보유'가 아니라 '같은 평균 비중 고정'과 비교해야 타이밍 효과만 남는다.
"""
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.argv = sys.argv[:3]
sys.path.insert(0, str(Path(__file__).parent))
import search as S  # noqa: E402

PICK = ["변동성 조절", "200일선 위", "앱 M 20/80", "앱 Z 20/80", "자기Z 역발상 20/80", "1·3·6·12개월 다수결",
        "고변동 회피(60일 vol<중앙값×1.5)", "12개월 수익>0"]


def cagr(sr):
    sr = sr.dropna(); eq = (1 + sr).cumprod()
    return eq.iloc[-1] ** (252 / len(sr)) - 1 if eq.iloc[-1] > 0 else -1.0


def mdd(sr):
    eq = (1 + sr.dropna()).cumprod(); return float((1 - eq / eq.cummax()).max())


def go(universe, lev, title):
    rows = []
    for sym in universe:
        df = S.frame(sym)
        if len(df) < S.WARM + 500:
            continue
        sig = S.signals(df)
        idx = df.index[S.WARM:]
        mid = idx[len(idx) // 2]
        r = df["c"].pct_change()
        if lev != 1.0:
            r = lev * r - 0.015 / 252
        for name in PICK:
            pos = sig[name]
            sr = S.evaluate(df, pos, lev).loc[idx]
            e = float(pos.loc[idx].mean())
            const = (e * r).loc[idx]                   # 같은 평균 비중 고정(매일 리밸런싱)
            for part, a, b in (("앞", sr.loc[:mid], const.loc[:mid]), ("뒤", sr.loc[mid:], const.loc[mid:]),
                               ("전체", sr, const)):
                rows.append(dict(sym=sym, sig=name, part=part, d=cagr(a) - cagr(b),
                                 dd=mdd(b) - mdd(a), expo=e))
    d = pd.DataFrame(rows)
    print(f"\n=== {title} — 연수익 차이(전략 − 같은 비중 고정) 중앙값·이긴 비율, 최대낙폭 개선 ===")
    out = []
    for sig, x in d.groupby("sig"):
        row = {"신호": sig}
        for p in ("앞", "뒤", "전체"):
            y = x[x.part == p]; row[f"{p}"] = y.d.median(); row[f"{p} 승"] = (y.d > 0).mean()
        row["MDD개선"] = x[x.part == "전체"].dd.median(); row["비중"] = x[x.part == "전체"].expo.median()
        out.append(row)
    t = pd.DataFrame(out).sort_values("전체", ascending=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:+.3f}".format):
        print(t.to_string(index=False))
    return d


if __name__ == "__main__":
    go(S.UNDER, 3.0, "3배 합성 (일반 자산 19개)")
    d = go(S.LEV, 1.0, "실제 레버리지 ETF 10개")
    print("\n변동성 조절 — 종목별 (전체 기간 연수익 차이, 최대낙폭 개선)")
    x = d[(d.sig == "변동성 조절") & (d.part == "전체")]
    print(x[["sym", "d", "dd", "expo"]].to_string(index=False))
