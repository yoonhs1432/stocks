package com.quant.dashboard.quant

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * 퀀트 분석 코어 — app.py / backend/analysis.py 수식을 Kotlin으로 포팅.
 *
 * SPY 대비 로그-로그 회귀 → 잔차 Z-score(expanding std),
 * RSI(Wilder), MACD, 변동성 적응 모멘텀 M. pandas 없이 직접 구현.
 * 수식 변경 시 app.py / backend/analysis.py 와 함께 점검할 것.
 */
object Quant {

    // ── Config (app.py Config 미러) ──
    private const val M_W_HEIGHT = 0.30
    private const val M_W_INFLECT = 0.15
    private const val M_W_RSI = 0.55
    private const val M_VOL_WINDOW = 120
    private const val M_SIGMA_SCALE = 1.5
    private const val M_RSI_SCALE = 30.0
    private const val EXPANDING_MIN = 30

    /** Z/M 점수 → 0~100 백분위 (Z=-2.5→0, 0→50, +2.5→100). */
    fun zToPct(z: Double): Double {
        if (z.isNaN()) return 50.0
        return ((z + 2.5) / 5.0 * 100).coerceIn(0.0, 100.0)
    }

    /** 백분위(0~100) → 5단계 신호. */
    fun pctToSignal(pct: Double): String = when {
        pct < 20 -> "strong_buy"
        pct < 40 -> "buy"
        pct < 60 -> "hold"
        pct < 80 -> "sell"
        else -> "strong_sell"
    }

    data class Result(
        val dates: LongArray,        // epoch seconds (일자)
        val price: DoubleArray,
        val tickerNorm: DoubleArray,
        val spyNorm: DoubleArray,
        val predicted: DoubleArray,
        val bandUpper: DoubleArray,
        val bandLower: DoubleArray,
        val zPct: DoubleArray,       // 0..100
        val mPct: DoubleArray,       // 0..100
        val rsi: DoubleArray,
        val macd: DoubleArray,
        val macdSignal: DoubleArray,
        val beta: Double,
        val sigmaPct: Double,
        val lastPrice: Double,
        val lastZpct: Double,
        val lastMpct: Double,
        val signal: String,
    )

    /**
     * 분석 실행. spy/ticker는 (epochSec, close) 시계열. 날짜 교집합으로 정렬.
     * 데이터가 부족하면 null.
     */
    fun analyze(spy: List<Pair<Long, Double>>, ticker: List<Pair<Long, Double>>): Result? {
        // 일자(epoch day) 기준 정렬 + 교집합
        val spyMap = LinkedHashMap<Long, Double>()
        for ((t, c) in spy) spyMap[t / 86400L] = c
        val tkMap = LinkedHashMap<Long, Double>()
        for ((t, c) in ticker) tkMap[t / 86400L] = c
        val days = spyMap.keys.intersect(tkMap.keys).sorted()
        val n = days.size
        if (n < EXPANDING_MIN) return null

        val dates = LongArray(n) { days[it] * 86400L }
        val x = DoubleArray(n) { spyMap[days[it]]!! }   // SPY close
        val y = DoubleArray(n) { tkMap[days[it]]!! }    // ticker close
        if (x[0] <= 0 || y[0] <= 0) return null

        val xNorm = DoubleArray(n) { x[it] / x[0] }
        val yNorm = DoubleArray(n) { y[it] / y[0] }
        val logX = DoubleArray(n) { ln(xNorm[it]) }
        val logY = DoubleArray(n) { ln(yNorm[it]) }

        // OLS (log-log): beta, intercept
        val (beta, intercept) = ols(logX, logY)
        val predicted = DoubleArray(n) { exp(intercept) * xNorm[it].pow(beta) }

        // RSI (Wilder, alpha=1/14)
        val rsi = computeRsi(y)

        // EMA / MACD
        val ema12 = ema(y, 12)
        val ema26 = ema(y, 26)
        val macd = DoubleArray(n) { ema12[it] - ema26[it] }
        val macdSignal = ema(macd, 9)

        val macdPct = DoubleArray(n) {
            if (ema26[it] != 0.0) macd[it] / ema26[it] * 100 else Double.NaN
        }
        // dMACD = MACD 1차 미분(EMA span=3) → 부호 반전 (매도방향 양수)
        val dmacd = DoubleArray(n) { if (it == 0) 0.0 else macd[it] - macd[it - 1] }
        val dmacdSmooth = ema(dmacd, 3)
        val dmacdPct = DoubleArray(n) {
            if (ema26[it] != 0.0) -(dmacdSmooth[it] / ema26[it] * 100) else Double.NaN
        }
        val macdPctStd = rollingStd(macdPct, M_VOL_WINDOW, EXPANDING_MIN)
        val dmacdPctStd = rollingStd(dmacdPct, M_VOL_WINDOW, EXPANDING_MIN)

        // Z-score: log_resid / expanding std
        val logResid = DoubleArray(n) { logY[it] - ln(predicted[it]) }
        val stdResid = sampleStd(logResid, 0, n)          // 밴드용 전체 std
        val expStd = expandingStd(logResid, EXPANDING_MIN)
        val z = DoubleArray(n) {
            val s = expStd[it]
            if (s.isNaN() || s == 0.0) Double.NaN else logResid[it] / s
        }

        // 모멘텀 M
        val mScore = DoubleArray(n) {
            momentum(macdPct[it], dmacdPct[it], rsi[it], macdPctStd[it], dmacdPctStd[it])
        }

        val zPct = DoubleArray(n) { zToPct(z[it]) }
        val mPct = DoubleArray(n) { zToPct(mScore[it]) }
        val bandUpper = DoubleArray(n) { exp(ln(predicted[it]) + 1.5 * stdResid) }
        val bandLower = DoubleArray(n) { exp(ln(predicted[it]) - 1.5 * stdResid) }

        // 현재 시점 expanding σ → σ%
        var sigmaUnit = stdResid
        for (i in n - 1 downTo 0) {
            if (!expStd[i].isNaN() && expStd[i] > 0) { sigmaUnit = expStd[i]; break }
        }
        val sigmaPct = (exp(sigmaUnit) - 1) * 100

        val lastZ = zPct[n - 1]
        val lastM = mPct[n - 1]
        return Result(
            dates = dates, price = y, tickerNorm = yNorm, spyNorm = xNorm,
            predicted = predicted, bandUpper = bandUpper, bandLower = bandLower,
            zPct = zPct, mPct = mPct, rsi = rsi, macd = macd, macdSignal = macdSignal,
            beta = beta, sigmaPct = sigmaPct, lastPrice = y[n - 1],
            lastZpct = lastZ, lastMpct = lastM, signal = pctToSignal(lastM),
        )
    }

    // ── 모멘텀 스칼라 (compute_momentum_score_smooth) ──
    private fun momentum(
        macdPct: Double, dmacdPct: Double, rsi: Double,
        macdStd: Double, dmacdStd: Double,
    ): Double {
        val mp = if (macdPct.isNaN()) 0.0 else macdPct
        val dp = if (dmacdPct.isNaN()) 0.0 else dmacdPct
        val r0 = if (rsi.isNaN()) 50.0 else rsi
        var h = if (!macdStd.isNaN() && macdStd > 0) mp / (M_SIGMA_SCALE * macdStd) else mp / 2.0
        var d = if (!dmacdStd.isNaN() && dmacdStd > 0) dp / (M_SIGMA_SCALE * dmacdStd) else dp / 0.5
        h = h.coerceIn(-1.0, 1.0)
        d = d.coerceIn(-1.0, 1.0)
        val r = ((r0 - 50) / M_RSI_SCALE).coerceIn(-1.0, 1.0)
        return 2.5 * (M_W_HEIGHT * h + M_W_INFLECT * d + M_W_RSI * r)
    }

    // ── 수치 헬퍼 ──
    private fun ols(x: DoubleArray, y: DoubleArray): Pair<Double, Double> {
        val n = x.size
        var mx = 0.0; var my = 0.0
        for (i in 0 until n) { mx += x[i]; my += y[i] }
        mx /= n; my /= n
        var num = 0.0; var den = 0.0
        for (i in 0 until n) {
            val dx = x[i] - mx
            num += dx * (y[i] - my)
            den += dx * dx
        }
        val beta = if (den != 0.0) num / den else 0.0
        return Pair(beta, my - beta * mx)
    }

    /** ewm adjust=false: out[0]=x[0], out[i]=a*x[i]+(1-a)*out[i-1]. */
    private fun ewm(x: DoubleArray, alpha: Double): DoubleArray {
        val out = DoubleArray(x.size)
        if (x.isEmpty()) return out
        out[0] = x[0]
        for (i in 1 until x.size) out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
        return out
    }

    private fun ema(x: DoubleArray, span: Int): DoubleArray = ewm(x, 2.0 / (span + 1))

    private fun computeRsi(close: DoubleArray): DoubleArray {
        val n = close.size
        val gain = DoubleArray(n)
        val loss = DoubleArray(n)
        for (i in 1 until n) {
            val d = close[i] - close[i - 1]
            if (d > 0) gain[i] = d else if (d < 0) loss[i] = -d
        }
        val g = ewm(gain, 1.0 / 14)
        val l = ewm(loss, 1.0 / 14)
        return DoubleArray(n) {
            val ll = l[it]
            val gg = g[it]
            if (ll == 0.0) { if (gg == 0.0) Double.NaN else 100.0 }
            else 100 - 100 / (1 + gg / ll)
        }
    }

    /** rolling 표본표준편차(ddof=1). minPeriods 미만 구간은 NaN. */
    private fun rollingStd(x: DoubleArray, window: Int, minPeriods: Int): DoubleArray {
        val n = x.size
        val out = DoubleArray(n) { Double.NaN }
        for (i in 0 until n) {
            val start = maxOf(0, i - window + 1)
            out[i] = sampleStd(x, start, i + 1)
            if (i - start + 1 < minPeriods) out[i] = Double.NaN
        }
        return out
    }

    private fun expandingStd(x: DoubleArray, minPeriods: Int): DoubleArray {
        val n = x.size
        val out = DoubleArray(n) { Double.NaN }
        for (i in 0 until n) {
            if (i + 1 >= minPeriods) out[i] = sampleStd(x, 0, i + 1)
        }
        return out
    }

    /** [start, end) 구간의 표본표준편차(ddof=1). NaN은 무시. */
    private fun sampleStd(x: DoubleArray, start: Int, end: Int): Double {
        var cnt = 0; var mean = 0.0
        for (i in start until end) {
            val v = x[i]
            if (!v.isNaN()) { cnt++; mean += v }
        }
        if (cnt < 2) return Double.NaN
        mean /= cnt
        var ss = 0.0
        for (i in start until end) {
            val v = x[i]
            if (!v.isNaN()) { val d = v - mean; ss += d * d }
        }
        return sqrt(ss / (cnt - 1))
    }
}
