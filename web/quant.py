"""퀀트 분석 코어 — `android-toss` 의 `quant/Quant.kt` 를 그대로 옮긴 것.

SPY 대비 로그-로그 회귀 → 잔차 Z-score(expanding std), RSI(Wilder), MACD,
변동성 적응 모멘텀 M.

⚠️ **수식을 고치면 `Quant.kt` 와 `app.py` 도 같이 손봐야 한다.** 세 곳이 같은 값을 내야
안드로이드·웹·스트림릿이 같은 숫자를 보여준다. 여기서는 pandas 없이 직접 구현한다
(`app.py` 를 import 하면 streamlit·FinanceDataReader·plotly 가 전부 딸려 온다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

NAN = float("nan")

# ── Config (app.py Config 미러) ──
M_W_HEIGHT = 0.30
M_W_INFLECT = 0.15
M_W_RSI = 0.55
M_VOL_WINDOW = 120
M_SIGMA_SCALE = 1.5
M_RSI_SCALE = 30.0
EXPANDING_MIN = 30


def _nan(v: float) -> bool:
    return v != v


def z_to_pct(z: float) -> float:
    """Z/M 점수 → 0~100 백분위 (Z=-2.5→0, 0→50, +2.5→100)."""
    if _nan(z):
        return 50.0
    return min(100.0, max(0.0, (z + 2.5) / 5.0 * 100))


def pct_to_signal(pct: float) -> str:
    """백분위(0~100) → 5단계 신호."""
    if pct < 20:
        return "strong_buy"
    if pct < 40:
        return "buy"
    if pct < 60:
        return "hold"
    if pct < 80:
        return "sell"
    return "strong_sell"


# ── 수치 헬퍼 ──

def _ols(x: list[float], y: list[float]) -> tuple[float, float]:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = den = 0.0
    for i in range(n):
        dx = x[i] - mx
        num += dx * (y[i] - my)
        den += dx * dx
    beta = num / den if den else 0.0
    return beta, my - beta * mx


def _ewm(x: list[float], alpha: float) -> list[float]:
    """ewm adjust=false: out[0]=x[0], out[i]=a*x[i]+(1-a)*out[i-1]."""
    if not x:
        return []
    out = [x[0]]
    for i in range(1, len(x)):
        out.append(alpha * x[i] + (1 - alpha) * out[i - 1])
    return out


def _ema(x: list[float], span: int) -> list[float]:
    return _ewm(x, 2.0 / (span + 1))


def _rsi(close: list[float]) -> list[float]:
    n = len(close)
    gain = [0.0] * n
    loss = [0.0] * n
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            gain[i] = d
        elif d < 0:
            loss[i] = -d
    g = _ewm(gain, 1.0 / 14)
    l = _ewm(loss, 1.0 / 14)
    out = []
    for i in range(n):
        if l[i] == 0:
            out.append(NAN if g[i] == 0 else 100.0)
        else:
            out.append(100 - 100 / (1 + g[i] / l[i]))
    return out


def _sample_std(x: list[float], start: int, end: int) -> float:
    """[start, end) 구간의 표본표준편차(ddof=1). NaN 은 무시."""
    vals = [v for v in x[start:end] if not _nan(v)]
    if len(vals) < 2:
        return NAN
    mean = sum(vals) / len(vals)
    ss = sum((v - mean) ** 2 for v in vals)
    return math.sqrt(ss / (len(vals) - 1))


def _rolling_std(x: list[float], window: int, min_periods: int) -> list[float]:
    n = len(x)
    out = [NAN] * n
    for i in range(n):
        start = max(0, i - window + 1)
        if i - start + 1 >= min_periods:
            out[i] = _sample_std(x, start, i + 1)
    return out


def _expanding_std(x: list[float], min_periods: int) -> list[float]:
    n = len(x)
    out = [NAN] * n
    for i in range(n):
        if i + 1 >= min_periods:
            out[i] = _sample_std(x, 0, i + 1)
    return out


def _momentum(macd_pct: float, dmacd_pct: float, rsi: float,
              macd_std: float, dmacd_std: float) -> float:
    """compute_momentum_score_smooth — 변동성으로 정규화한 M 스칼라."""
    mp = 0.0 if _nan(macd_pct) else macd_pct
    dp = 0.0 if _nan(dmacd_pct) else dmacd_pct
    r0 = 50.0 if _nan(rsi) else rsi
    h = mp / (M_SIGMA_SCALE * macd_std) if (not _nan(macd_std) and macd_std > 0) else mp / 2.0
    d = dp / (M_SIGMA_SCALE * dmacd_std) if (not _nan(dmacd_std) and dmacd_std > 0) else dp / 0.5
    h = min(1.0, max(-1.0, h))
    d = min(1.0, max(-1.0, d))
    r = min(1.0, max(-1.0, (r0 - 50) / M_RSI_SCALE))
    return 2.5 * (M_W_HEIGHT * h + M_W_INFLECT * d + M_W_RSI * r)


# ── 공개 API ──

def macd_of(close: list[float]) -> tuple[list[float], list[float]]:
    """종가만으로 MACD·Signal — **봉 주기와 무관**(1분봉 모드에서 재계산).

    일봉 분석([analyze])은 SPY 와 날짜 교집합을 맞춘 뒤 계산하지만, MACD·RSI 는
    그 종목 종가만 있으면 되는 값이라 따로 낼 수 있다. Z·M·회귀는 그렇지 않다.
    """
    e12 = _ema(close, 12)
    e26 = _ema(close, 26)
    macd = [e12[i] - e26[i] for i in range(len(close))]
    return macd, _ema(macd, 9)


def rsi_of(close: list[float]) -> list[float]:
    """종가만으로 RSI(14)."""
    return _rsi(close)


@dataclass
class Result:
    dates: list[int] = field(default_factory=list)      # epoch seconds
    price: list[float] = field(default_factory=list)
    tickerNorm: list[float] = field(default_factory=list)
    spyNorm: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    bandUpper: list[float] = field(default_factory=list)
    bandLower: list[float] = field(default_factory=list)
    zPct: list[float] = field(default_factory=list)     # 0..100
    mPct: list[float] = field(default_factory=list)     # 0..100
    rsi: list[float] = field(default_factory=list)
    macd: list[float] = field(default_factory=list)
    macdSignal: list[float] = field(default_factory=list)
    beta: float = 0.0
    sigmaPct: float = 0.0
    lastPrice: float = 0.0
    lastZpct: float = 50.0
    lastMpct: float = 50.0
    signal: str = "hold"


def analyze(spy: list[tuple[int, float]],
            ticker: list[tuple[int, float]]) -> Result | None:
    """spy/ticker 는 (epochSec, close) 시계열. **날짜 교집합**으로 맞춘다.

    교집합을 쓰는 이유 — 한쪽에만 있는 날(휴장 차이 등)을 남기면 회귀가 어긋난다.
    데이터가 모자라면(EXPANDING_MIN 미만) None.
    """
    spy_map = {t // 86400: c for t, c in spy}
    tk_map = {t // 86400: c for t, c in ticker}
    days = sorted(set(spy_map) & set(tk_map))
    n = len(days)
    if n < EXPANDING_MIN:
        return None

    dates = [d * 86400 for d in days]
    x = [spy_map[d] for d in days]
    y = [tk_map[d] for d in days]
    if x[0] <= 0 or y[0] <= 0:
        return None

    x_norm = [v / x[0] for v in x]
    y_norm = [v / y[0] for v in y]
    log_x = [math.log(v) for v in x_norm]
    log_y = [math.log(v) for v in y_norm]

    beta, intercept = _ols(log_x, log_y)
    predicted = [math.exp(intercept) * (v ** beta) for v in x_norm]

    rsi = _rsi(y)
    ema12 = _ema(y, 12)
    ema26 = _ema(y, 26)
    macd = [ema12[i] - ema26[i] for i in range(n)]
    macd_signal = _ema(macd, 9)

    macd_pct = [macd[i] / ema26[i] * 100 if ema26[i] else NAN for i in range(n)]
    # dMACD = MACD 1차 미분(EMA span=3) → 부호 반전 (매도방향이 양수)
    dmacd = [0.0 if i == 0 else macd[i] - macd[i - 1] for i in range(n)]
    dmacd_smooth = _ema(dmacd, 3)
    dmacd_pct = [-(dmacd_smooth[i] / ema26[i] * 100) if ema26[i] else NAN for i in range(n)]
    macd_pct_std = _rolling_std(macd_pct, M_VOL_WINDOW, EXPANDING_MIN)
    dmacd_pct_std = _rolling_std(dmacd_pct, M_VOL_WINDOW, EXPANDING_MIN)

    log_resid = [log_y[i] - math.log(predicted[i]) for i in range(n)]
    std_resid = _sample_std(log_resid, 0, n)          # 밴드용 전체 std
    exp_std = _expanding_std(log_resid, EXPANDING_MIN)
    z = [NAN if (_nan(exp_std[i]) or exp_std[i] == 0) else log_resid[i] / exp_std[i]
         for i in range(n)]

    m_score = [_momentum(macd_pct[i], dmacd_pct[i], rsi[i],
                         macd_pct_std[i], dmacd_pct_std[i]) for i in range(n)]

    # warmup(Z 미정의) 구간은 NaN 으로 남긴다 — 50 으로 채우면 산점도에서 한 줄로 뭉친다
    z_pct = [NAN if _nan(z[i]) else z_to_pct(z[i]) for i in range(n)]
    m_pct = [z_to_pct(v) for v in m_score]
    band_u = [math.exp(math.log(predicted[i]) + 1.5 * std_resid) for i in range(n)]
    band_l = [math.exp(math.log(predicted[i]) - 1.5 * std_resid) for i in range(n)]

    sigma_unit = std_resid
    for i in range(n - 1, -1, -1):
        if not _nan(exp_std[i]) and exp_std[i] > 0:
            sigma_unit = exp_std[i]
            break
    sigma_pct = (math.exp(sigma_unit) - 1) * 100

    last_z = 50.0 if _nan(z_pct[-1]) else z_pct[-1]
    last_m = 50.0 if _nan(m_pct[-1]) else m_pct[-1]
    return Result(
        dates=dates, price=y, tickerNorm=y_norm, spyNorm=x_norm,
        predicted=predicted, bandUpper=band_u, bandLower=band_l,
        zPct=z_pct, mPct=m_pct, rsi=rsi, macd=macd, macdSignal=macd_signal,
        beta=beta, sigmaPct=sigma_pct, lastPrice=y[-1],
        lastZpct=last_z, lastMpct=last_m, signal=pct_to_signal(last_m),
    )
