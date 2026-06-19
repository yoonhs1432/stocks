package com.quant.dashboard.ui

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TextSecondary
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/** 의존성 없는 Compose Canvas 차트. 가격($)·Z·M·RSI를 세로 스택으로. */

/** 차트 위 매매 마커 (x=윈도우 내 인덱스, y=해당 차트 y척도 값, buy 여부). */
data class Mark(val x: Int, val y: Double, val buy: Boolean)

/** 완료 사이클 평균매수→평균매도 화살표 (x=윈도우 인덱스, y=가격, profit=수익여부). */
data class CycleArrow(val x1: Int, val y1: Double, val x2: Int, val y2: Double, val profit: Boolean)

private fun DrawScope.marker(cx: Float, cy: Float, buy: Boolean, r: Float = 7f) {
    val col = if (buy) Color(0xFFDC2626) else Color(0xFF2563EB)
    drawCircle(col, r, Offset(cx, cy))
    drawCircle(Color.White, r, Offset(cx, cy), style = Stroke(1.2f))
    // 내부 화살표 — 원에 꽉 차게 (글리프 높이 ≈ 지름)
    label(if (buy) "↑" else "↓", cx, cy + r * 0.92f, 0xFFFFFFFF.toInt(), r * 2.7f, Paint.Align.CENTER)
}

/** 사이클 화살표 (app.py 평균매수→평균매도 주석 화살표). 수익=녹색/손실=빨강. */
private fun DrawScope.arrow(x1: Float, y1: Float, x2: Float, y2: Float, profit: Boolean) {
    val col = if (profit) Color(0xFF16A34A) else Color(0xFFDC2626)
    drawLine(col, Offset(x1, y1), Offset(x2, y2), 2.5f)
    val ang = atan2(y2 - y1, x2 - x1)
    val len = 16f
    val a1 = ang + 2.618f   // 150°
    val a2 = ang - 2.618f
    drawLine(col, Offset(x2, y2), Offset(x2 + len * cos(a1), y2 + len * sin(a1)), 2.5f)
    drawLine(col, Offset(x2, y2), Offset(x2 + len * cos(a2), y2 + len * sin(a2)), 2.5f)
}

private fun DoubleArray.minNaN(): Double {
    var m = Double.POSITIVE_INFINITY
    for (v in this) if (!v.isNaN() && v < m) m = v
    return if (m.isInfinite()) 0.0 else m
}

private fun DoubleArray.maxNaN(): Double {
    var m = Double.NEGATIVE_INFINITY
    for (v in this) if (!v.isNaN() && v > m) m = v
    return if (m.isInfinite()) 1.0 else m
}

private fun DrawScope.label(text: String, x: Float, y: Float, colorArgb: Int, sizePx: Float, align: Paint.Align = Paint.Align.LEFT) {
    val p = Paint().apply {
        color = colorArgb; textSize = sizePx; textAlign = align; isAntiAlias = true
    }
    drawContext.canvas.nativeCanvas.drawText(text, x, y, p)
}

private fun DrawScope.poly(data: DoubleArray, xAt: (Int) -> Float, yAt: (Double) -> Float, color: Color, stroke: Float) {
    var prev = -1
    for (i in data.indices) {
        if (data[i].isNaN()) { prev = -1; continue }
        if (prev >= 0) drawLine(color, Offset(xAt(prev), yAt(data[prev])), Offset(xAt(i), yAt(data[i])), stroke)
        prev = i
    }
}

/** Turbo 컬러맵 근사 (파랑→청록→초록→노랑→빨강). */
private fun turbo(t: Float): Color {
    val x = t.coerceIn(0f, 1f)
    val stops = listOf(
        Color(0xFF30123B), Color(0xFF28BBEC), Color(0xFFA2FC3C),
        Color(0xFFFB8022), Color(0xFF7A0403),
    )
    val seg = x * (stops.size - 1)
    val i = seg.toInt().coerceIn(0, stops.size - 2)
    return lerp(stops[i], stops[i + 1], seg - i)
}

private val DASH = PathEffect.dashPathEffect(floatArrayOf(8f, 8f))

/** 점선. */
private fun DrawScope.dline(color: Color, x1: Float, y1: Float, x2: Float, y2: Float, w: Float) {
    drawLine(color, Offset(x1, y1), Offset(x2, y2), w, pathEffect = DASH)
}

private val DOT = PathEffect.dashPathEffect(floatArrayOf(2f, 5f))

/** 촘촘한 점선(dot, app.py dash='dot'). */
private fun DrawScope.dotline(color: Color, x1: Float, y1: Float, x2: Float, y2: Float, w: Float) {
    drawLine(color, Offset(x1, y1), Offset(x2, y2), w, pathEffect = DOT)
}

/** 차트 테두리 — app.py 전 축 showline+mirror (#adbac7 1px, 4면). */
private fun DrawScope.chartBorder() {
    drawRect(Color(0xFFADBAC7), topLeft = Offset(0f, 0f), size = size, style = Stroke(1f))
}

/** 현재 위치 별표 (흰 채움 + 검정 테두리). app.py symbol='star' size12 line 1.5. */
private fun DrawScope.star(cx: Float, cy: Float, rOuter: Float) {
    val rInner = rOuter * 0.42f
    val p = Path()
    for (k in 0 until 10) {
        val rr = if (k % 2 == 0) rOuter else rInner
        val ang = (-90.0 + k * 36.0) * Math.PI / 180.0
        val x = cx + (rr * cos(ang)).toFloat()
        val y = cy + (rr * sin(ang)).toFloat()
        if (k == 0) p.moveTo(x, y) else p.lineTo(x, y)
    }
    p.close()
    drawPath(p, Color.White)
    drawPath(p, Color.Black, style = Stroke(1.5f))
}

/** 로그축 1-2-5 눈금 시퀀스 (범위 [lo,hi] 내). */
private fun log125(lo: Double, hi: Double): List<Double> {
    if (lo <= 0 || hi <= lo) return emptyList()
    val out = ArrayList<Double>()
    var e = Math.floor(Math.log10(lo)).toInt()
    while (e <= 9) {
        val dec = Math.pow(10.0, e.toDouble())
        for (m in intArrayOf(1, 2, 5)) {
            val v = m * dec
            if (v in lo..hi) out.add(v)
        }
        if (dec > hi) break
        e++
    }
    return out
}

private fun fmt125(v: Double): String =
    if (v >= 1) "%.0f".format(v) else if (v >= 0.1) "%.1f".format(v) else "%.2f".format(v)

/** 임계값 위쪽 면적 채움 (Z>80, RSI>70 등). */
private fun DrawScope.fillAbove(data: DoubleArray, threshold: Double, xAt: (Int) -> Float, yAt: (Double) -> Float, color: Color) {
    val n = data.size
    val p = Path(); p.moveTo(xAt(0), yAt(threshold))
    for (i in 0 until n) { val v = if (data[i].isNaN()) threshold else maxOf(data[i], threshold); p.lineTo(xAt(i), yAt(v)) }
    for (i in n - 1 downTo 0) p.lineTo(xAt(i), yAt(threshold))
    p.close(); drawPath(p, color)
}

/** 임계값 아래쪽 면적 채움 (RSI<30 등). */
private fun DrawScope.fillBelow(data: DoubleArray, threshold: Double, xAt: (Int) -> Float, yAt: (Double) -> Float, color: Color) {
    val n = data.size
    val p = Path(); p.moveTo(xAt(0), yAt(threshold))
    for (i in 0 until n) { val v = if (data[i].isNaN()) threshold else minOf(data[i], threshold); p.lineTo(xAt(i), yAt(v)) }
    for (i in n - 1 downTo 0) p.lineTo(xAt(i), yAt(threshold))
    p.close(); drawPath(p, color)
}

/**
 * ① 회귀 산점도 (로그-로그). X=SPY_Norm, Y=종목_Norm.
 * Turbo 시간순 점 + 회귀선 + ±1.5σ 밴드 + 가이드 점선 + 현재 위치 ★ + β 라벨.
 */
@Composable
fun RegressionScatter(
    spyNorm: DoubleArray, tickerNorm: DoubleArray, predicted: DoubleArray,
    bandU: DoubleArray, bandL: DoubleArray, beta: Double,
    markIdx: List<Pair<Int, Boolean>> = emptyList(),
    modifier: Modifier = Modifier,
) {
    val n = spyNorm.size
    if (n < 2) return
    val guideN = 4.0
    var xLo = Double.MAX_VALUE; var xHi = -Double.MAX_VALUE
    for (v in spyNorm) if (v > 0 && v.isFinite()) { if (v < xLo) xLo = v; if (v > xHi) xHi = v }
    var yLo = Double.MAX_VALUE; var yHi = -Double.MAX_VALUE
    for (arr in listOf(tickerNorm, bandU, bandL)) for (v in arr) if (v > 0 && v.isFinite()) { if (v < yLo) yLo = v; if (v > yHi) yHi = v }
    if (xHi <= xLo || yHi <= yLo) return
    val lxLo = Math.log10(xLo * 0.98); val lxHi = Math.log10(xHi * 1.02)
    val lyLo = Math.log10(yLo * 0.88); val lyHi = Math.log10(yHi * 1.18)
    val order = (0 until n).sortedBy { spyNorm[it] }
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun sx(v: Double) = (size.width * (Math.log10(v) - lxLo) / (lxHi - lxLo)).toFloat()
        fun sy(v: Double) = (size.height * (1 - (Math.log10(v) - lyLo) / (lyHi - lyLo))).toFloat()
        // 가이드 곡선 (y = c·x^guideN)
        var ecLo = Double.MAX_VALUE; var ecHi = -Double.MAX_VALUE
        for (i in 0 until n) if (spyNorm[i] > 0) {
            val ec = tickerNorm[i] / Math.pow(spyNorm[i], guideN)
            if (ec > 0 && ec.isFinite()) { if (ec < ecLo) ecLo = ec; if (ec > ecHi) ecHi = ec }
        }
        if (ecLo < Double.MAX_VALUE) {
            val lcLo = Math.log10(ecLo) - 1.0; val lcHi = Math.log10(ecHi) + 1.0
            // 가이드 곡선 15개 — app.py rgba(200,200,200,0.6) width1 dash='dot'
            for (g in 0 until 15) {
                val c = Math.pow(10.0, lcLo + (lcHi - lcLo) * g / 14.0)
                val gp = Path(); var started = false
                for (s in 0..24) {
                    val xv = xLo * Math.pow(xHi / xLo, s / 24.0)
                    val yv = c * Math.pow(xv, guideN)
                    if (yv > 0) {
                        val o = Offset(sx(xv), sy(yv))
                        if (!started) { gp.moveTo(o.x, o.y); started = true } else gp.lineTo(o.x, o.y)
                    }
                }
                drawPath(gp, Color(0x99C8C8C8), style = Stroke(1f, pathEffect = DOT))
            }
        }
        // 밴드
        val bp = Path(); bp.moveTo(sx(spyNorm[order[0]]), sy(bandU[order[0]]))
        for (k in 1 until n) { val i = order[k]; bp.lineTo(sx(spyNorm[i]), sy(bandU[i])) }
        for (k in n - 1 downTo 0) { val i = order[k]; bp.lineTo(sx(spyNorm[i]), sy(bandL[i])) }
        bp.close(); drawPath(bp, Color(0x33969696))
        // 회귀선
        var prev: Offset? = null
        for (k in 0 until n) { val i = order[k]; val o = Offset(sx(spyNorm[i]), sy(predicted[i])); if (prev != null) drawLine(Color(0xFFADBAC7), prev, o, 2f); prev = o }
        // Turbo 점
        for (i in 0 until n) if (spyNorm[i] > 0 && tickerNorm[i] > 0) drawCircle(turbo(i.toFloat() / (n - 1)), 4f, Offset(sx(spyNorm[i]), sy(tickerNorm[i])))
        // 매매 마커 (회귀 패널 위 ↑/↓ 풍선) — 점이 작아 마커도 작게(r6)
        for ((i, buy) in markIdx) if (i in 0 until n && spyNorm[i] > 0 && tickerNorm[i] > 0) {
            marker(sx(spyNorm[i]), sy(tickerNorm[i]), buy, 6f)
        }
        // 현재 위치 ★ (별표)
        val li = n - 1
        if (spyNorm[li] > 0 && tickerNorm[li] > 0) star(sx(spyNorm[li]), sy(tickerNorm[li]), 11f)
        // 로그축 눈금 (1·2·5·10·20·50·100 …) — x는 하단, y는 좌측
        val gx = 0xCCADBAC7.toInt()
        for (v in log125(Math.pow(10.0, lxLo), Math.pow(10.0, lxHi))) {
            val xx = sx(v)
            drawLine(Color(0x55ADBAC7), Offset(xx, size.height), Offset(xx, size.height - 4f), 1f)
            label(fmt125(v), xx, size.height - 6f, gx, 17f, Paint.Align.CENTER)
        }
        for (v in log125(Math.pow(10.0, lyLo), Math.pow(10.0, lyHi))) {
            val yy = sy(v)
            drawLine(Color(0x55ADBAC7), Offset(0f, yy), Offset(4f, yy), 1f)
            label(fmt125(v), 5f, yy - 2f, gx, 17f)
        }
        label("β=${"%.2f".format(beta)}", size.width - 4f, 24f, 0xFFADBAC7.toInt(), 20f, Paint.Align.RIGHT)
        chartBorder()
    }
}


/** 가격($) + 회귀선 + ±1.5σ 밴드. 우측에 최고/최저가 라벨. */
@Composable
fun PriceChart(
    priceDollar: DoubleArray,
    predictedDollar: DoubleArray,
    bandUpper: DoubleArray,
    bandLower: DoubleArray,
    markers: List<Mark> = emptyList(),
    arrows: List<CycleArrow> = emptyList(),
    currency: String = "$",
    modifier: Modifier = Modifier,
) {
    val n = priceDollar.size
    if (n < 2) return
    var lo = minOf(priceDollar.minNaN(), bandLower.minNaN())
    var hi = maxOf(priceDollar.maxNaN(), bandUpper.maxNaN())
    if (hi <= lo) return
    val pad = (hi - lo) * 0.05
    lo -= pad; hi += pad

    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()

        val band = Path().apply {
            moveTo(0f, yAt(bandUpper[0]))
            for (i in 1 until n) lineTo(xAt(i), yAt(bandUpper[i]))
            for (i in n - 1 downTo 0) lineTo(xAt(i), yAt(bandLower[i]))
            close()
        }
        drawPath(band, Color(0x22FFFFFF))
        poly(priceDollar, ::xAt, ::yAt, Color(0xFFE6EDF3), 2.5f)

        val gray = 0xFFADBAC7.toInt()
        label("$currency${"%,.0f".format(hi)}", 6f, 24f, gray, 24f)
        label("$currency${"%,.0f".format(lo)}", 6f, size.height - 10f, gray, 24f)

        for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
        chartBorder()
    }
}

/** 캔들(상승=빨강/하락=파랑) + ±1.5σ 밴드 + 흰 종가선 + 매매 마커. 값은 가격($/₩) 단위. */
@Composable
fun CandleChart(
    opens: DoubleArray, highs: DoubleArray, lows: DoubleArray, closes: DoubleArray,
    predicted: DoubleArray, bandUpper: DoubleArray, bandLower: DoubleArray,
    markers: List<Mark> = emptyList(),
    arrows: List<CycleArrow> = emptyList(),
    currency: String = "$",
    topLabel: String = "",
    modifier: Modifier = Modifier,
) {
    val n = closes.size
    if (n < 2) return
    var lo = minOf(lows.minNaN(), bandLower.minNaN())
    var hi = maxOf(highs.maxNaN(), bandUpper.maxNaN())
    if (hi <= lo) return
    val pad = (hi - lo) * 0.05
    lo -= pad; hi += pad
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()
        val band = Path().apply {
            moveTo(0f, yAt(bandUpper[0]))
            for (i in 1 until n) lineTo(xAt(i), yAt(bandUpper[i]))
            for (i in n - 1 downTo 0) lineTo(xAt(i), yAt(bandLower[i]))
            close()
        }
        drawPath(band, Color(0x1AFFFFFF))
        // 흰 종가선 (캔들 뒤)
        poly(closes, ::xAt, ::yAt, Color(0x99E6EDF3), 0.8f)
        // 캔들
        val w = (size.width / n * 0.6f).coerceAtLeast(1.5f)
        for (i in 0 until n) {
            if (opens[i].isNaN() || highs[i].isNaN() || lows[i].isNaN() || closes[i].isNaN()) continue
            val up = closes[i] >= opens[i]
            val col = if (up) Color(0xFFF85149) else Color(0xFF58A6FF)
            val cx = xAt(i)
            drawLine(col, Offset(cx, yAt(highs[i])), Offset(cx, yAt(lows[i])), 1.5f)
            val top = yAt(maxOf(opens[i], closes[i]))
            val bot = yAt(minOf(opens[i], closes[i]))
            drawRect(col, topLeft = Offset(cx - w / 2, top), size = Size(w, maxOf(bot - top, 1f)))
        }
        val gray = 0xFFADBAC7.toInt()
        label("$currency${"%,.0f".format(hi)}", size.width - 6f, 24f, gray, 22f, Paint.Align.RIGHT)
        label("$currency${"%,.0f".format(lo)}", size.width - 6f, size.height - 10f, gray, 22f, Paint.Align.RIGHT)
        if (topLabel.isNotEmpty()) label(topLabel, 6f, 26f, 0xFFE6EDF3.toInt(), 26f)
        for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
        chartBorder()
    }
}

/** Z(흰)·M(주황) 백분위 0~100, 임계선 20/40/60/80, Z>80 빨강 면적. */
@Composable
fun ZmChart(zPct: DoubleArray, mPct: DoubleArray, markers: List<Mark> = emptyList(),
            topLabel: String = "", modifier: Modifier = Modifier) {
    val n = zPct.size
    if (n < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        // Z>80 빨강 면적
        fillAbove(zPct, 80.0, ::xAt, ::yAt, Color(0x40F85149))
        // 임계선 20/40/60/80 흰색 점선 (app.py dash='dot')
        for (t in intArrayOf(20, 40, 60, 80)) {
            dotline(Color(0x88FFFFFF), 0f, yAt(t.toDouble()), size.width, yAt(t.toDouble()), 0.6f)
        }
        // 좌측 Y 눈금 0/50/100
        for (t in intArrayOf(0, 50, 100)) label(t.toString(), 2f, yAt(t.toDouble()) - 2f, 0x77FFFFFF.toInt(), 15f)
        poly(zPct, ::xAt, ::yAt, Color(0xFFE6EDF3), 2f)
        poly(mPct, ::xAt, ::yAt, Color(0xFFF97316), 1.5f)
        if (topLabel.isNotEmpty()) label(topLabel, 6f, 22f, 0xFFE6EDF3.toInt(), 22f)
        for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
        chartBorder()
    }
}

/**
 * Z·M 사분면 산점도. X=Z(0~100), Y=M(0~100). 시간 순서대로 색(파랑→빨강) 궤적 +
 * 현재 위치 별표 + 매매 마커. tradeIdx: (전체 인덱스, 매수여부).
 */
@Composable
fun ZmScatter(
    zPct: DoubleArray, mPct: DoubleArray,
    tradeIdx: List<Pair<Int, Boolean>> = emptyList(),
    modifier: Modifier = Modifier,
) {
    val n = zPct.size
    if (n < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(110.dp)) {
        // 범위 -5~105 (app.py): 별표가 0/100 극단에 가도 안 잘리게 마진
        val lo = -5.0; val hi = 105.0; val span = hi - lo
        fun px(v: Double) = (size.width * ((v - lo) / span)).toFloat()
        fun py(v: Double) = (size.height * (1 - (v - lo) / span)).toFloat()
        // 임계선 20/40/60/80만 — 흰색 점선 (app.py dash='dot', 50 중앙선·축라벨 없음)
        for (t in intArrayOf(20, 40, 60, 80)) {
            dotline(Color(0x88FFFFFF), px(t.toDouble()), 0f, px(t.toDouble()), size.height, 0.6f)
            dotline(Color(0x88FFFFFF), 0f, py(t.toDouble()), size.width, py(t.toDouble()), 0.6f)
        }
        // 좌측 Y / 하단 X 눈금 숫자 (0/50/100) — 크고 또렷하게
        for (t in intArrayOf(0, 50, 100)) {
            label(t.toString(), 3f, py(t.toDouble()) + 8f, 0xCCFFFFFF.toInt(), 24f)
            label(t.toString(), px(t.toDouble()), size.height - 4f, 0xCCFFFFFF.toInt(), 24f, Paint.Align.CENTER)
        }
        // 시간 궤적 점 — Turbo 컬러맵 (파랑→청록→초록→노랑→빨강), size5, 테두리 없음
        for (i in 0 until n) {
            if (zPct[i].isNaN() || mPct[i].isNaN()) continue
            drawCircle(turbo(i.toFloat() / (n - 1)), 5f, Offset(px(zPct[i]), py(mPct[i])))
        }
        // 매매 마커 (circle size8 + ↑/↓, 흰 테두리)
        for ((idx, buy) in tradeIdx) if (idx in 0 until n) {
            if (!zPct[idx].isNaN() && !mPct[idx].isNaN()) marker(px(zPct[idx]), py(mPct[idx]), buy)
        }
        // 현재 위치 별표
        val li = n - 1
        if (!zPct[li].isNaN() && !mPct[li].isNaN()) star(px(zPct[li]), py(mPct[li]), 11f)
        chartBorder()
    }
}

/** RSI 0~100, 70/50/30 임계선 + >70 빨강·<30 파랑 면적. */
@Composable
fun RsiChart(rsi: DoubleArray, topLabel: String = "", modifier: Modifier = Modifier) {
    val n = rsi.size
    if (n < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        fillAbove(rsi, 70.0, ::xAt, ::yAt, Color(0x47F85149))
        fillBelow(rsi, 30.0, ::xAt, ::yAt, Color(0x4758A6FF))
        // 70/50/30 실선 (app.py add_hline solid)
        drawLine(Color(0xFFF85149), Offset(0f, yAt(70.0)), Offset(size.width, yAt(70.0)), 0.7f)
        drawLine(Color(0xFF768390), Offset(0f, yAt(50.0)), Offset(size.width, yAt(50.0)), 0.5f)
        drawLine(Color(0xFF58A6FF), Offset(0f, yAt(30.0)), Offset(size.width, yAt(30.0)), 0.7f)
        for (t in intArrayOf(0, 50, 100)) label(t.toString(), 2f, yAt(t.toDouble()) - 2f, 0x77FFFFFF.toInt(), 15f)
        poly(rsi, ::xAt, ::yAt, Color(0xFF22D3EE), 2f)
        if (topLabel.isNotEmpty()) label(topLabel, 6f, 22f, 0xFF22D3EE.toInt(), 22f)
        chartBorder()
    }
}

/** 자산추이(누적손익) 라인 + 색 마커 + 0 기준 점선. unit 접미사(예: 만원). */
@Composable
fun EquityChart(values: DoubleArray, unit: String = "$", modifier: Modifier = Modifier) {
    val n = values.size
    if (n < 2) return
    var lo = values.minNaN(); var hi = values.maxNaN()
    if (lo > 0) lo = 0.0
    if (hi < 0) hi = 0.0
    if (hi <= lo) hi = lo + 1.0
    val pad = (hi - lo) * 0.08
    lo -= pad; hi += pad
    Canvas(modifier = modifier.fillMaxWidth().height(160.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()
        val y0 = yAt(0.0)
        // 0선 점선
        var dx = 0f
        while (dx < size.width) { drawLine(Color(0x888B949E), Offset(dx, y0), Offset(dx + 6f, y0), 0.8f); dx += 12f }
        poly(values, ::xAt, ::yAt, Color(0xFFF85149), 2f)
        for (i in 0 until n) {
            if (values[i].isNaN()) continue
            val c = if (values[i] >= 0) Color(0xFFDC2626) else Color(0xFF2563EB)
            drawCircle(c, 5f, Offset(xAt(i), yAt(values[i])))
            drawCircle(Color(0xFF0D1117), 5f, Offset(xAt(i), yAt(values[i])), style = Stroke(1f))
        }
        if (unit == "만원") {
            label("${"%,.0f".format(hi)}만원", size.width - 6f, 22f, 0xFFADBAC7.toInt(), 20f, Paint.Align.RIGHT)
            label("${"%,.0f".format(lo)}만원", size.width - 6f, size.height - 8f, 0xFFADBAC7.toInt(), 20f, Paint.Align.RIGHT)
        } else {
            label("$unit${"%,.0f".format(hi)}", 6f, 22f, 0xFFADBAC7.toInt(), 20f)
            label("$unit${"%,.0f".format(lo)}", 6f, size.height - 8f, 0xFFADBAC7.toInt(), 20f)
        }
        chartBorder()
    }
}

/** MACD(보라) + Signal(흰) + 0선 + 교차 마커(▲빨강 상향 / ▼파랑 하향). */
@Composable
fun MacdChart(macd: DoubleArray, signal: DoubleArray, topLabel: String = "", modifier: Modifier = Modifier) {
    val n = macd.size
    if (n < 2) return
    var mx = 0.0
    for (v in macd) if (!v.isNaN() && kotlin.math.abs(v) > mx) mx = kotlin.math.abs(v)
    for (v in signal) if (!v.isNaN() && kotlin.math.abs(v) > mx) mx = kotlin.math.abs(v)
    if (mx <= 0) mx = 1.0
    mx *= 1.15
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v + mx) / (2 * mx))).toFloat()
        // 0 중립선 실선 (app.py)
        drawLine(Color(0xFF768390), Offset(0f, yAt(0.0)), Offset(size.width, yAt(0.0)), 0.5f)
        label("0", 2f, yAt(0.0) - 2f, 0x77FFFFFF.toInt(), 15f)
        poly(macd, ::xAt, ::yAt, Color(0xFF7C3AED), 2f)
        poly(signal, ::xAt, ::yAt, Color(0xFFE6EDF3), 0.9f)
        for (i in 1 until n) {
            if (macd[i].isNaN() || signal[i].isNaN() || macd[i - 1].isNaN() || signal[i - 1].isNaN()) continue
            val prev = macd[i - 1] - signal[i - 1]
            val cur = macd[i] - signal[i]
            if (prev < 0 && cur >= 0) marker(xAt(i), yAt(macd[i]), true)
            else if (prev > 0 && cur <= 0) marker(xAt(i), yAt(macd[i]), false)
        }
        if (topLabel.isNotEmpty()) label(topLabel, 6f, 22f, 0xFFE6EDF3.toInt(), 22f)
        chartBorder()
    }
}

/** 산점도 한 점. */
data class ScatterPt(val x: Double, val y: Double, val label: String, val color: Color)

/** 임계/중앙선. */
data class GridLine(val v: Double, val color: Color, val width: Float)

/**
 * Streamlit 스타일 라벨 산점도 — 큰 흰테두리 점 + 종목명 라벨(8방향 분산으로 겹침 완화) + 임계선.
 * yLog=true면 Y축 로그(σ%용).
 */
@Composable
fun ScatterChart(
    points: List<ScatterPt>,
    xMin: Double, xMax: Double, yMin: Double, yMax: Double,
    yLog: Boolean = false,
    vLines: List<GridLine> = emptyList(),
    hLines: List<GridLine> = emptyList(),
    xAxisLabel: String = "", yAxisLabel: String = "",
    labelTopCenter: Boolean = false,
    height: Dp = 340.dp,
    modifier: Modifier = Modifier,
) {
    // NaN/무한 좌표 제거 (없으면 좌표·라벨 각도 계산에서 크래시)
    val pts = points.filter { it.x.isFinite() && it.y.isFinite() && (!yLog || it.y > 0) }
    if (pts.isEmpty()) return
    val vL = vLines.filter { it.v.isFinite() }
    val hL = hLines.filter { it.v.isFinite() }
    val lyMin = if (yLog) kotlin.math.log10(maxOf(yMin, 1e-6)) else yMin
    val lyMax = if (yLog) kotlin.math.log10(maxOf(yMax, 1e-6)) else yMax
    val xSpan = if (xMax > xMin) xMax - xMin else 1.0
    val ySpan = if (lyMax > lyMin) lyMax - lyMin else 1.0
    Canvas(modifier = modifier.fillMaxWidth().height(height)) {
        val pad = 8f
        fun sx(x: Double) = (pad + (size.width - 2 * pad) * ((x - xMin) / xSpan)).toFloat()
        fun sy(y: Double): Float {
            val yy = if (yLog) kotlin.math.log10(y.coerceAtLeast(1e-6)) else y
            return (pad + (size.height - 2 * pad) * (1 - (yy - lyMin) / ySpan)).toFloat()
        }
        for (g in vL) drawLine(g.color, Offset(sx(g.v), 0f), Offset(sx(g.v), size.height), g.width)
        for (g in hL) drawLine(g.color, Offset(0f, sy(g.v)), Offset(size.width, sy(g.v)), g.width)

        val xs = FloatArray(pts.size) { sx(pts[it].x) }
        val ys = FloatArray(pts.size) { sy(pts[it].y) }
        // 점 (큰 원 + 흰 테두리)
        for (i in pts.indices) {
            drawCircle(pts[i].color, 14f, Offset(xs[i], ys[i]))
            drawCircle(Color.White, 14f, Offset(xs[i], ys[i]), style = Stroke(2f))
        }
        // 라벨 — top-center(라벨 위) 또는 8방향 분산
        val r = 18f
        if (labelTopCenter) {
            for (i in pts.indices) label(pts[i].label, xs[i], ys[i] - 16f, 0xFFE6EDF3.toInt(), 22f, Paint.Align.CENTER)
        } else
        for (i in pts.indices) {
            var best = Float.MAX_VALUE; var nd = -1
            for (j in pts.indices) {
                if (i == j) continue
                val dx = xs[j] - xs[i]; val dy = ys[j] - ys[i]
                val d = dx * dx + dy * dy
                if (d < best) { best = d; nd = j }
            }
            val ax = if (nd >= 0) -(xs[nd] - xs[i]) else 1f
            val ay = if (nd >= 0) -(ys[nd] - ys[i]) else 0f
            val k = if (ax == 0f && ay == 0f) 0
                else (((Math.toDegrees(kotlin.math.atan2(ay, ax).toDouble()) / 45.0).roundToInt()) % 8 + 8) % 8
            // k: 0=오른,1=오른아래,2=아래,3=왼아래,4=왼,5=왼위,6=위,7=오른위
            val (ox, oy, align) = when (k) {
                0 -> Triple(r, 5f, Paint.Align.LEFT)
                1 -> Triple(r, r, Paint.Align.LEFT)
                2 -> Triple(0f, r + 12f, Paint.Align.CENTER)
                3 -> Triple(-r, r, Paint.Align.RIGHT)
                4 -> Triple(-r, 5f, Paint.Align.RIGHT)
                5 -> Triple(-r, -r + 4f, Paint.Align.RIGHT)
                6 -> Triple(0f, -r, Paint.Align.CENTER)
                else -> Triple(r, -r + 4f, Paint.Align.LEFT)
            }
            label(pts[i].label, xs[i] + ox, ys[i] + oy, 0xFFE6EDF3.toInt(), 22f, align)
        }
        if (xAxisLabel.isNotEmpty()) label(xAxisLabel, size.width - 8f, size.height - 8f, 0x88FFFFFF.toInt(), 22f, Paint.Align.RIGHT)
        if (yAxisLabel.isNotEmpty()) label(yAxisLabel, 6f, 22f, 0x88FFFFFF.toInt(), 22f)
        chartBorder()
    }
}


/** 차트 하단 공통 X축 날짜 라벨 (start · mid · end). */
@Composable
fun DateAxis(datesEpochSec: LongArray, modifier: Modifier = Modifier) {
    val n = datesEpochSec.size
    if (n < 2) return
    val fmt = SimpleDateFormat("yy/MM/dd", Locale.US)
    fun d(i: Int) = fmt.format(Date(datesEpochSec[i] * 1000L))
    Row(modifier = modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        androidx.compose.material3.Text(d(0), color = TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.Normal)
        androidx.compose.material3.Text(d(n / 2), color = TextSecondary, fontSize = 10.sp)
        androidx.compose.material3.Text(d(n - 1), color = TextSecondary, fontSize = 10.sp)
    }
}
