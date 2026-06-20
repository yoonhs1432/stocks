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
import androidx.compose.ui.graphics.drawscope.clipRect
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
import kotlin.math.sin

/** 의존성 없는 Compose Canvas 차트. 가격($)·Z·M·RSI를 세로 스택으로. */

/** 차트 위 매매 마커 (x=윈도우 내 인덱스, y=해당 차트 y척도 값, buy 여부). */
data class Mark(val x: Int, val y: Double, val buy: Boolean)

/** 완료 사이클 평균매수→평균매도 화살표 (x=윈도우 인덱스, y=가격, profit=수익여부). */
data class CycleArrow(val x1: Int, val y1: Double, val x2: Int, val y2: Double, val profit: Boolean)

// 모든 차트의 숫자/값 라벨 공통 스타일 — 종목 버튼 글자에 맞춰 sp 기반(밀도 추종, 크게)
private val DrawScope.AX_SIZE: Float get() = 13.sp.toPx()
private val AX_COLOR = 0xFFADBAC7.toInt()

private fun DrawScope.marker(cx: Float, cy: Float, buy: Boolean, r: Float = 13.5f) {
    val col = if (buy) Color(0xFFDC2626) else Color(0xFF2563EB)
    drawCircle(col, r, Offset(cx, cy))
    drawCircle(Color.White, r, Offset(cx, cy), style = Stroke(0.8f))   // 얇은 흰 테두리
    // 내부 ↑/↓ — bold + 흰 외곽선(FILL_AND_STROKE)으로 두껍게, 원에 꽉 차게
    val p = Paint().apply {
        color = 0xFFFFFFFF.toInt(); textSize = r * 2.3f
        textAlign = Paint.Align.CENTER; isAntiAlias = true; isFakeBoldText = true
        style = Paint.Style.FILL_AND_STROKE
        strokeWidth = r * 0.22f
        strokeJoin = Paint.Join.ROUND
    }
    drawContext.canvas.nativeCanvas.drawText(if (buy) "↑" else "↓", cx, cy + r * 0.85f, p)
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

/** 현재 위치 마커 — 마젠타 다이아몬드 + 흰 테두리 (어떤 팔레트 위에서도 또렷). */
private fun DrawScope.currentMarker(cx: Float, cy: Float, r: Float) {
    val p = Path().apply {
        moveTo(cx, cy - r)   // 위
        lineTo(cx + r, cy)   // 오른
        lineTo(cx, cy + r)   // 아래
        lineTo(cx - r, cy)   // 왼
        close()
    }
    drawPath(p, Color(0xFFFF2BD6))                 // 마젠타 채움
    drawPath(p, Color.White, style = Stroke(2.5f)) // 흰 테두리
}

// ── 증권앱 스타일 공용 (우측 축·펜넌트·크로스 화살표·십자선) ──
private const val RIGHT_PAD = 152f   // 우측 축 라벨 + 펜넌트 공간
private const val FLAG_W = 118f
private const val FLAG_TIP = 14f
private val MAGENTA = Color(0xFFFF2BD6)
private val ORANGE = Color(0xFFE8943A)
private val ORANGE2 = Color(0xFFF3C489)

/** 좌향 뾰족 펜넌트(고정 폭) — tip이 plotRight(현재값 위치)를 가리키고 우측 여백에 본체. 텍스트 자동 축소. */
private fun DrawScope.pennant(plotRight: Float, y: Float, lines: List<String>, bg: Color) {
    val tip = plotRight; val x0 = tip + FLAG_TIP; val x1 = x0 + FLAG_W
    val lineH = AX_SIZE * 1.05f
    val half = (lines.size * lineH) / 2f + 4f
    val yy = y.coerceIn(half + 1f, size.height - half - 1f)
    val path = Path().apply {
        moveTo(tip, yy); lineTo(x0, yy - half); lineTo(x1, yy - half)
        lineTo(x1, yy + half); lineTo(x0, yy + half); close()
    }
    drawPath(path, bg)
    val cx = (x0 + x1) / 2f
    val tp = Paint().apply { color = 0xFFFFFFFF.toInt(); textAlign = Paint.Align.CENTER; isAntiAlias = true; isFakeBoldText = true }
    var fs = AX_SIZE; val longest = lines.maxByOrNull { it.length } ?: ""
    tp.textSize = fs
    while (tp.measureText(longest) > FLAG_W - 10f && fs > AX_SIZE * 0.55f) { fs -= 1f; tp.textSize = fs }
    val total = lines.size * (fs * 1.12f); var ty = yy - total / 2f + fs * 0.82f
    for (ln in lines) { drawContext.canvas.nativeCanvas.drawText(ln, cx, ty, tp); ty += fs * 1.12f }
}

/** 우측 축 — 가로 그리드 + 우측정렬 값 라벨. */
private fun DrawScope.rightAxis(ticks: List<Double>, yAt: (Double) -> Float, plotRight: Float, fmt: (Double) -> String) {
    for (t in ticks) {
        val yy = yAt(t)
        drawLine(Color(0x2EADBAC7), Offset(0f, yy), Offset(plotRight, yy), 0.8f)
        label(fmt(t), size.width - 4f, yy + AX_SIZE * 0.34f, AX_COLOR, AX_SIZE, Paint.Align.RIGHT)
    }
}

/** 작은 매매신호 삼각형 (매수=빨강▲ / 매도=파랑▼). */
private fun DrawScope.smallCross(x: Float, y: Float, up: Boolean) {
    val col = if (up) Color(0xFFE84D5E) else Color(0xFF3D7DE0)
    val r = 9f; val p = Path()
    if (up) { p.moveTo(x, y - r); p.lineTo(x - r * 0.8f, y + r * 0.6f); p.lineTo(x + r * 0.8f, y + r * 0.6f) }
    else { p.moveTo(x, y + r); p.lineTo(x - r * 0.8f, y - r * 0.6f); p.lineTo(x + r * 0.8f, y - r * 0.6f) }
    p.close(); drawPath(p, col)
}

/** 산점도 현재 X/Y 값 미니 태그 (마젠타 박스 + 흰 글자). */
private fun DrawScope.miniTag(x: Float, y: Float, text: String, bg: Color, align: Paint.Align) {
    val tp = Paint().apply { textSize = AX_SIZE; textAlign = align; isAntiAlias = true; isFakeBoldText = true }
    val w = tp.measureText(text) + 12f; val h = AX_SIZE + 9f
    val left = when (align) { Paint.Align.CENTER -> x - w / 2; Paint.Align.RIGHT -> x - w; else -> x }
    drawRect(bg, topLeft = Offset(left, y - h / 2f), size = Size(w, h))
    tp.color = 0xFFFFFFFF.toInt()
    drawContext.canvas.nativeCanvas.drawText(text, x, y + tp.textSize * 0.35f, tp)
}

/** 산점도 현재 위치 십자선 + 축 교점 값 태그 + 다이아 마커. */
private fun DrawScope.crosshair(cx: Float, cy: Float, xText: String, yText: String) {
    dotline(MAGENTA, cx, 0f, cx, size.height, 1f)
    dotline(MAGENTA, 0f, cy, size.width, cy, 1f)
    currentMarker(cx, cy, 13f)
    miniTag(cx, size.height - (AX_SIZE + 9f) / 2f - 1f, xText, MAGENTA, Paint.Align.CENTER)
    miniTag(2f, cy, yText, MAGENTA, Paint.Align.LEFT)
}

/** 가격 표기 — 크기에 따라 자릿수 자동(천 단위 콤마/소수). */
private fun priceFmt(v: Double): String = when {
    kotlin.math.abs(v) >= 1000 -> "%,.0f".format(v)
    kotlin.math.abs(v) >= 1 -> "%.2f".format(v)
    else -> "%.4f".format(v)
}

/** 보기 좋은 축 눈금(1·2·5×10ⁿ) — [lo,hi] 안에서 target개 내외. */
private fun niceTicks(lo: Double, hi: Double, target: Int = 5): List<Double> {
    if (hi <= lo) return emptyList()
    val raw = (hi - lo) / target
    val mag = Math.pow(10.0, Math.floor(Math.log10(raw)))
    val norm = raw / mag
    val step = (if (norm < 1.5) 1.0 else if (norm < 3) 2.0 else if (norm < 7) 5.0 else 10.0) * mag
    val out = ArrayList<Double>()
    var t = Math.ceil(lo / step) * step
    var guard = 0
    while (t <= hi + step * 1e-6 && guard < 40) { out.add(t); t += step; guard++ }
    return out
}

/** 캔들 최고/최저 수평 콜아웃 — 점에 화살촉, 바깥쪽으로 짧은 선 + 값 텍스트. */
private fun DrawScope.hCallout(px: Float, py: Float, text: String, colorArgb: Int, textRight: Boolean, plotW: Float) {
    val col = Color(colorArgb)
    val len = 26f
    val ex = if (textRight) px + len else px - len
    drawLine(col, Offset(px, py), Offset(ex, py), 2f)
    // 점을 향한 화살촉
    val s = if (textRight) 1f else -1f
    drawLine(col, Offset(px, py), Offset(px + s * 7f, py - 5f), 2f)
    drawLine(col, Offset(px, py), Offset(px + s * 7f, py + 5f), 2f)
    val tx = (if (textRight) ex + 4f else ex - 4f).coerceIn(2f, plotW - 2f)
    label(text, tx, py + AX_SIZE * 0.32f, colorArgb, AX_SIZE * 0.88f,
        if (textRight) Paint.Align.LEFT else Paint.Align.RIGHT)
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
    Canvas(modifier = modifier.fillMaxWidth().height(120.dp)) {
        fun sx(v: Double) = (size.width * (Math.log10(v) - lxLo) / (lxHi - lxLo)).toFloat()
        fun sy(v: Double) = (size.height * (1 - (Math.log10(v) - lyLo) / (lyHi - lyLo))).toFloat()
        // 패널 밖으로 삐져나가지 않게 클리핑 (가이드 곡선 overflow 버그 수정)
        clipRect(0f, 0f, size.width, size.height) {
            // 가이드 곡선 (y = c·x^guideN)
            var ecLo = Double.MAX_VALUE; var ecHi = -Double.MAX_VALUE
            for (i in 0 until n) if (spyNorm[i] > 0) {
                val ec = tickerNorm[i] / Math.pow(spyNorm[i], guideN)
                if (ec > 0 && ec.isFinite()) { if (ec < ecLo) ecLo = ec; if (ec > ecHi) ecHi = ec }
            }
            if (ecLo < Double.MAX_VALUE) {
                val lcLo = Math.log10(ecLo) - 1.0; val lcHi = Math.log10(ecHi) + 1.0
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
            // Turbo 점 (크게)
            for (i in 0 until n) if (spyNorm[i] > 0 && tickerNorm[i] > 0) drawCircle(turbo(i.toFloat() / (n - 1)), 6f, Offset(sx(spyNorm[i]), sy(tickerNorm[i])))
            // 매매 마커
            for ((i, buy) in markIdx) if (i in 0 until n && spyNorm[i] > 0 && tickerNorm[i] > 0) {
                marker(sx(spyNorm[i]), sy(tickerNorm[i]), buy)
            }
        }
        // 로그축 눈금 (1·2·5·10·20·50·100 …) — x는 하단, y는 좌측
        for (v in log125(Math.pow(10.0, lxLo), Math.pow(10.0, lxHi))) {
            val xx = sx(v)
            drawLine(Color(0x66ADBAC7), Offset(xx, size.height), Offset(xx, size.height - 5f), 1f)
            label(fmt125(v), xx, size.height - 7f, AX_COLOR, AX_SIZE, Paint.Align.CENTER)
        }
        for (v in log125(Math.pow(10.0, lyLo), Math.pow(10.0, lyHi))) {
            val yy = sy(v)
            drawLine(Color(0x66ADBAC7), Offset(0f, yy), Offset(5f, yy), 1f)
            label(fmt125(v), 6f, yy - 2f, AX_COLOR, AX_SIZE)
        }
        label("β=${"%.2f".format(beta)}", size.width - 5f, AX_SIZE + 2f, AX_COLOR, AX_SIZE, Paint.Align.RIGHT)
        // 현재 위치 십자선 + 축 교점 값 (SPY_Norm / 종목_Norm)
        val li = n - 1
        if (spyNorm[li] > 0 && tickerNorm[li] > 0) {
            crosshair(sx(spyNorm[li]), sy(tickerNorm[li]),
                "%.2f".format(spyNorm[li]), "%.2f".format(tickerNorm[li]))
        }
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
    // y범위는 종목 가격에만 맞춤 (밴드 제거)
    var lo = priceDollar.minNaN()
    var hi = priceDollar.maxNaN()
    if (hi <= lo) return
    val pad = (hi - lo) * 0.06
    lo -= pad; hi += pad

    val cur = priceDollar.lastOrNull { !it.isNaN() } ?: return
    Canvas(modifier = modifier.fillMaxWidth().height(110.dp)) {
        val plotW = size.width - RIGHT_PAD
        fun xAt(i: Int) = plotW * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()
        rightAxis(niceTicks(lo, hi, 5), ::yAt, plotW) { priceFmt(it) }
        clipRect(0f, 0f, plotW, size.height) {
            poly(priceDollar, ::xAt, ::yAt, Color(0xFFE6EDF3), 2.5f)
            for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
        }
        pennant(plotW, yAt(cur), listOf("$currency${priceFmt(cur)}"), Color(0xFFE84D5E))
        chartBorder()
    }
}

/** 캔들(상승=빨강/하락=파랑) + 흰 종가선 + 매매 마커 + 증권앱 스타일(우측축·고저 콜아웃·현재가 펜넌트). */
@Composable
fun CandleChart(
    opens: DoubleArray, highs: DoubleArray, lows: DoubleArray, closes: DoubleArray,
    predicted: DoubleArray, bandUpper: DoubleArray, bandLower: DoubleArray,
    markers: List<Mark> = emptyList(),
    arrows: List<CycleArrow> = emptyList(),
    currency: String = "$",
    topLabel: String = "",
    dates: LongArray = LongArray(0),
    dailyChgPct: Double = Double.NaN,
    modifier: Modifier = Modifier,
) {
    val n = closes.size
    if (n < 2) return
    // y범위는 종목 가격(고/저)에만 맞춤
    var lo = lows.minNaN()
    var hi = highs.maxNaN()
    if (hi <= lo) return
    val pad = (hi - lo) * 0.10
    val ymin = lo - pad; val ymax = hi + pad
    // 실제 최고/최저 캔들 위치
    var hiI = -1; var loI = -1
    var hiV = -Double.MAX_VALUE; var loV = Double.MAX_VALUE
    for (i in 0 until n) {
        if (!highs[i].isNaN() && highs[i] > hiV) { hiV = highs[i]; hiI = i }
        if (!lows[i].isNaN() && lows[i] < loV) { loV = lows[i]; loI = i }
    }
    val cur = closes.lastOrNull { !it.isNaN() } ?: return
    val df = SimpleDateFormat("yy.MM.dd", Locale.US)
    fun dlbl(i: Int) = if (i in dates.indices) df.format(Date(dates[i] * 1000L)) else ""
    Canvas(modifier = modifier.fillMaxWidth().height(110.dp)) {
        val plotW = size.width - RIGHT_PAD
        fun xAt(i: Int) = plotW * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - ymin) / (ymax - ymin))).toFloat()
        rightAxis(niceTicks(ymin, ymax, 5), ::yAt, plotW) { priceFmt(it) }
        // 현재가 수평선(옅음)
        drawLine(Color(0x66E84D5E), Offset(0f, yAt(cur)), Offset(plotW, yAt(cur)), 0.8f)
        clipRect(0f, 0f, plotW, size.height) {
            poly(closes, ::xAt, ::yAt, Color(0x99E6EDF3), 1.2f)   // 흰 종가선
            val w = (plotW / n * 0.6f).coerceAtLeast(1.5f)
            for (i in 0 until n) {
                if (opens[i].isNaN() || highs[i].isNaN() || lows[i].isNaN() || closes[i].isNaN()) continue
                val up = closes[i] >= opens[i]
                val col = if (up) Color(0xFFE84D5E) else Color(0xFF3D7DE0)
                val cx = xAt(i)
                drawLine(col, Offset(cx, yAt(highs[i])), Offset(cx, yAt(lows[i])), 1.5f)
                val top = yAt(maxOf(opens[i], closes[i]))
                val bot = yAt(minOf(opens[i], closes[i]))
                drawRect(col, topLeft = Offset(cx - w / 2, top), size = Size(w, maxOf(bot - top, 1f)))
            }
            for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
        }
        // 고/저 수평 콜아웃 (현재가 대비 %)
        if (hiI >= 0) {
            val hp = (cur / hiV - 1) * 100
            hCallout(xAt(hiI), yAt(hiV), "$currency${priceFmt(hiV)} ${dlbl(hiI)} ${"%.1f".format(hp)}%",
                0xFFE84D5E.toInt(), textRight = hiI < n / 2, plotW = plotW)
        }
        if (loI >= 0) {
            val lp = (cur / loV - 1) * 100
            hCallout(xAt(loI), yAt(loV), "$currency${priceFmt(loV)} ${dlbl(loI)} +${"%.1f".format(lp)}%",
                0xFF3D7DE0.toInt(), textRight = loI < n / 2, plotW = plotW)
        }
        if (topLabel.isNotEmpty()) label(topLabel, 6f, AX_SIZE, AX_COLOR, AX_SIZE)
        // 현재가 펜넌트 (2행: 가격 + 일간 등락%)
        val up = dailyChgPct.isNaN() || dailyChgPct >= 0
        val chg = if (dailyChgPct.isNaN()) "" else "${if (dailyChgPct >= 0) "+" else ""}${"%.2f".format(dailyChgPct)}%"
        val lines = if (chg.isEmpty()) listOf("$currency${priceFmt(cur)}") else listOf("$currency${priceFmt(cur)}", chg)
        pennant(plotW, yAt(cur), lines, if (up) Color(0xFFE84D5E) else Color(0xFF3D7DE0))
        chartBorder()
    }
}

/** Z(흰)·M(주황) 백분위 0~100, 임계선 20/40/60/80, Z>80 빨강 면적. */
@Composable
fun ZmChart(zPct: DoubleArray, mPct: DoubleArray, markers: List<Mark> = emptyList(),
            topLabel: String = "", modifier: Modifier = Modifier) {
    val n = zPct.size
    if (n < 2) return
    val mCur = mPct.lastOrNull { !it.isNaN() }
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        val plotW = size.width - RIGHT_PAD
        fun xAt(i: Int) = plotW * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        rightAxis(listOf(0.0, 50.0, 100.0), ::yAt, plotW) { "%.0f".format(it) }
        // 임계선 20/40/60/80 흰색 점선 (app.py dash='dot')
        for (t in intArrayOf(20, 40, 60, 80)) {
            dotline(Color(0x55FFFFFF), 0f, yAt(t.toDouble()), plotW, yAt(t.toDouble()), 0.6f)
        }
        clipRect(0f, 0f, plotW, size.height) {
            poly(zPct, ::xAt, ::yAt, Color(0xFFEEF2F8), 2f)       // Z 흰
            poly(mPct, ::xAt, ::yAt, ORANGE, 1.6f)                // M 주황
            for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
        }
        label("Z·M", 6f, AX_SIZE, 0xFFE8943A.toInt(), AX_SIZE)
        if (mCur != null) pennant(plotW, yAt(mCur), listOf("M ${"%.0f".format(mCur)}"), ORANGE)
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
    Canvas(modifier = modifier.fillMaxWidth().height(130.dp)) {
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
            label(t.toString(), 4f, py(t.toDouble()) + AX_SIZE * 0.38f, AX_COLOR, AX_SIZE)
            label(t.toString(), px(t.toDouble()), size.height - 5f, AX_COLOR, AX_SIZE, Paint.Align.CENTER)
        }
        // 시간 궤적 점 — Turbo 컬러맵 (크게)
        for (i in 0 until n) {
            if (zPct[i].isNaN() || mPct[i].isNaN()) continue
            drawCircle(turbo(i.toFloat() / (n - 1)), 8f, Offset(px(zPct[i]), py(mPct[i])))
        }
        // 매매 마커
        for ((idx, buy) in tradeIdx) if (idx in 0 until n) {
            if (!zPct[idx].isNaN() && !mPct[idx].isNaN()) marker(px(zPct[idx]), py(mPct[idx]), buy)
        }
        // 현재 위치 십자선 + 축 교점 값 (Z / M)
        val li = n - 1
        if (!zPct[li].isNaN() && !mPct[li].isNaN()) {
            crosshair(px(zPct[li]), py(mPct[li]),
                "Z %.0f".format(zPct[li]), "M %.0f".format(mPct[li]))
        }
        chartBorder()
    }
}

/** RSI 0~100, 70/50/30 임계선 + >70 빨강·<30 파랑 면적. */
@Composable
fun RsiChart(rsi: DoubleArray, topLabel: String = "", modifier: Modifier = Modifier) {
    val n = rsi.size
    if (n < 2) return
    // RSI 시그널 = EMA(rsi, 9) (워밍업 NaN 무시)
    val sig = DoubleArray(n) { Double.NaN }
    run {
        val a = 2.0 / (9 + 1); var e = Double.NaN
        for (i in 0 until n) { val v = rsi[i]; if (v.isNaN()) continue; e = if (e.isNaN()) v else a * v + (1 - a) * e; sig[i] = e }
    }
    val rsiCur = rsi.lastOrNull { !it.isNaN() }
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        val plotW = size.width - RIGHT_PAD
        fun xAt(i: Int) = plotW * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        rightAxis(listOf(30.0, 50.0, 70.0), ::yAt, plotW) { "%.0f".format(it) }
        clipRect(0f, 0f, plotW, size.height) {
            poly(rsi, ::xAt, ::yAt, ORANGE, 2f)
            poly(sig, ::xAt, ::yAt, ORANGE2, 1.3f)
            for (i in 1 until n) {
                if (rsi[i].isNaN() || sig[i].isNaN() || rsi[i - 1].isNaN() || sig[i - 1].isNaN()) continue
                val prev = rsi[i - 1] - sig[i - 1]
                val cur = rsi[i] - sig[i]
                if (prev < 0 && cur >= 0) smallCross(xAt(i), yAt(rsi[i]) + 15f, true)
                else if (prev > 0 && cur <= 0) smallCross(xAt(i), yAt(rsi[i]) - 15f, false)
            }
        }
        label("RSI·Signal", 6f, AX_SIZE, 0xFFE8943A.toInt(), AX_SIZE)
        if (rsiCur != null) pennant(plotW, yAt(rsiCur), listOf("%.2f".format(rsiCur)), ORANGE)
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
            label("${"%,.0f".format(hi)}만원", size.width - 6f, AX_SIZE, AX_COLOR, AX_SIZE, Paint.Align.RIGHT)
            label("${"%,.0f".format(lo)}만원", size.width - 6f, size.height - 8f, AX_COLOR, AX_SIZE, Paint.Align.RIGHT)
        } else {
            label("$unit${"%,.0f".format(hi)}", 6f, AX_SIZE, AX_COLOR, AX_SIZE)
            label("$unit${"%,.0f".format(lo)}", 6f, size.height - 8f, AX_COLOR, AX_SIZE)
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
    val macdCur = macd.lastOrNull { !it.isNaN() }
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        val plotW = size.width - RIGHT_PAD
        fun xAt(i: Int) = plotW * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v + mx) / (2 * mx))).toFloat()
        rightAxis(listOf(0.0), ::yAt, plotW) { "%.2f".format(it) }
        clipRect(0f, 0f, plotW, size.height) {
            poly(macd, ::xAt, ::yAt, ORANGE, 2f)
            poly(signal, ::xAt, ::yAt, ORANGE2, 1.3f)
            for (i in 1 until n) {
                if (macd[i].isNaN() || signal[i].isNaN() || macd[i - 1].isNaN() || signal[i - 1].isNaN()) continue
                val prev = macd[i - 1] - signal[i - 1]
                val cur = macd[i] - signal[i]
                if (prev < 0 && cur >= 0) smallCross(xAt(i), yAt(macd[i]) + 15f, true)   // 매수 ▲ 선 아래
                else if (prev > 0 && cur <= 0) smallCross(xAt(i), yAt(macd[i]) - 15f, false) // 매도 ▼ 선 위
            }
        }
        label("MACD·Signal", 6f, AX_SIZE, 0xFFE8943A.toInt(), AX_SIZE)
        if (macdCur != null) pennant(plotW, yAt(macdCur), listOf("%.2f".format(macdCur)), ORANGE)
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
        val rb = 18f
        // 점 (큰 원 + 흰 테두리)
        for (i in pts.indices) {
            drawCircle(pts[i].color, rb, Offset(xs[i], ys[i]))
            drawCircle(Color.White, rb, Offset(xs[i], ys[i]), style = Stroke(2.5f))
        }
        // ── 라벨 겹침 회피 배치 (그리디 8방향: 원·기존라벨과 겹침 최소 위치 선택) ──
        val ts = 33f
        val tp = Paint().apply { textSize = ts; isAntiAlias = true }
        val occ = ArrayList<FloatArray>(pts.size * 2)
        for (i in pts.indices) occ.add(floatArrayOf(xs[i] - rb, ys[i] - rb, xs[i] + rb, ys[i] + rb))
        fun ovl(a: FloatArray, b: FloatArray): Float {
            val dx = minOf(a[2], b[2]) - maxOf(a[0], b[0])
            val dy = minOf(a[3], b[3]) - maxOf(a[1], b[1])
            return (if (dx > 0) dx else 0f) * (if (dy > 0) dy else 0f)
        }
        val ddx = floatArrayOf(0f, 0f, 1f, -1f, 1f, -1f, 1f, -1f)
        val ddy = floatArrayOf(-1f, 1f, 0f, 0f, -1f, -1f, 1f, 1f)
        val dal = arrayOf(
            Paint.Align.CENTER, Paint.Align.CENTER, Paint.Align.LEFT, Paint.Align.RIGHT,
            Paint.Align.LEFT, Paint.Align.RIGHT, Paint.Align.LEFT, Paint.Align.RIGHT,
        )
        val lblX = FloatArray(pts.size); val lblY = FloatArray(pts.size)
        val lblA = arrayOfNulls<Paint.Align>(pts.size)
        // 위(y큰)·바깥 점부터 배치 → 중앙 밀집부는 남은 자리로
        for (i in pts.indices.sortedByDescending { ys[it] }) {
            val w = tp.measureText(pts[i].label); val h = ts
            var bestSc = Float.MAX_VALUE; var bx0 = 0f; var by0 = 0f; var bal = Paint.Align.CENTER
            for (k in 0 until 8) {
                val d = rb + 6f
                val cx = xs[i] + ddx[k] * d; val cy = ys[i] + ddy[k] * d
                val al = dal[k]
                val x0 = when (al) { Paint.Align.LEFT -> cx; Paint.Align.RIGHT -> cx - w; else -> cx - w / 2 }
                val y1 = if (ddy[k] < 0) cy else if (ddy[k] > 0) cy + h else cy + h / 2
                val y0 = y1 - h
                val rect = floatArrayOf(x0, y0, x0 + w, y1)
                var sc = 0.01f * k
                for (b in occ) sc += ovl(rect, b)
                if (x0 < 1f || x0 + w > size.width - 1f || y0 < 1f || y1 > size.height - 1f) sc += 1e5f
                if (sc < bestSc) { bestSc = sc; bx0 = x0; by0 = y0; bal = al }
            }
            occ.add(floatArrayOf(bx0, by0, bx0 + w, by0 + h))
            lblA[i] = bal
            lblX[i] = when (bal) { Paint.Align.LEFT -> bx0; Paint.Align.RIGHT -> bx0 + w; else -> bx0 + w / 2 }
            lblY[i] = by0 + ts * 0.8f
        }
        for (i in pts.indices) label(pts[i].label, lblX[i], lblY[i], 0xFFE6EDF3.toInt(), ts, lblA[i]!!)
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
