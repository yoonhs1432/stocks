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
import kotlin.math.roundToInt

/** 의존성 없는 Compose Canvas 차트. 가격($)·Z·M·RSI를 세로 스택으로. */

/** 차트 위 매매 마커 (x=윈도우 내 인덱스, y=해당 차트 y척도 값, buy 여부). */
data class Mark(val x: Int, val y: Double, val buy: Boolean)

private fun DrawScope.marker(cx: Float, cy: Float, buy: Boolean) {
    val col = if (buy) Color(0xFFDC2626) else Color(0xFF2563EB)
    drawCircle(col, 9f, Offset(cx, cy))
    drawCircle(Color.White, 9f, Offset(cx, cy), style = Stroke(1.5f))
    label(if (buy) "↑" else "↓", cx, cy + 7f, 0xFFFFFFFF.toInt(), 20f, Paint.Align.CENTER)
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

/** 가격($) + 회귀선 + ±1.5σ 밴드. 우측에 최고/최저가 라벨. */
@Composable
fun PriceChart(
    priceDollar: DoubleArray,
    predictedDollar: DoubleArray,
    bandUpper: DoubleArray,
    bandLower: DoubleArray,
    markers: List<Mark> = emptyList(),
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

    Canvas(modifier = modifier.fillMaxWidth().height(190.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()

        val band = Path().apply {
            moveTo(0f, yAt(bandUpper[0]))
            for (i in 1 until n) lineTo(xAt(i), yAt(bandUpper[i]))
            for (i in n - 1 downTo 0) lineTo(xAt(i), yAt(bandLower[i]))
            close()
        }
        drawPath(band, Color(0x22FFFFFF))
        poly(predictedDollar, ::xAt, ::yAt, Color(0xFFADBAC7), 1.5f)
        poly(priceDollar, ::xAt, ::yAt, Color(0xFFE6EDF3), 2.5f)

        val gray = 0xFFADBAC7.toInt()
        label("$currency${"%,.0f".format(hi)}", 6f, 24f, gray, 24f)
        label("$currency${"%,.0f".format(lo)}", 6f, size.height - 10f, gray, 24f)

        for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
    }
}

/** 캔들(상승=빨강/하락=파랑) + 회귀선 + ±1.5σ 밴드 + 매매 마커. 값은 가격($/₩) 단위. */
@Composable
fun CandleChart(
    opens: DoubleArray, highs: DoubleArray, lows: DoubleArray, closes: DoubleArray,
    predicted: DoubleArray, bandUpper: DoubleArray, bandLower: DoubleArray,
    markers: List<Mark> = emptyList(),
    currency: String = "$",
    modifier: Modifier = Modifier,
) {
    val n = closes.size
    if (n < 2) return
    var lo = minOf(lows.minNaN(), bandLower.minNaN())
    var hi = maxOf(highs.maxNaN(), bandUpper.maxNaN())
    if (hi <= lo) return
    val pad = (hi - lo) * 0.05
    lo -= pad; hi += pad
    Canvas(modifier = modifier.fillMaxWidth().height(210.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()
        // 밴드
        val band = Path().apply {
            moveTo(0f, yAt(bandUpper[0]))
            for (i in 1 until n) lineTo(xAt(i), yAt(bandUpper[i]))
            for (i in n - 1 downTo 0) lineTo(xAt(i), yAt(bandLower[i]))
            close()
        }
        drawPath(band, Color(0x1AFFFFFF))
        poly(predicted, ::xAt, ::yAt, Color(0xFFADBAC7), 1.5f)
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
        label("$currency${"%,.0f".format(hi)}", 6f, 24f, gray, 24f)
        label("$currency${"%,.0f".format(lo)}", 6f, size.height - 10f, gray, 24f)
        for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
    }
}

/** Z(흰)·M(주황) 백분위 0~100, 임계선 20/40/60/80. */
@Composable
fun ZmChart(zPct: DoubleArray, mPct: DoubleArray, markers: List<Mark> = emptyList(), modifier: Modifier = Modifier) {
    val n = zPct.size
    if (n < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(120.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        for (t in intArrayOf(20, 40, 60, 80)) {
            val y = yAt(t.toDouble())
            drawLine(Color(0x33FFFFFF), Offset(0f, y), Offset(size.width, y), 1f)
            label(t.toString(), 4f, y - 3f, 0x66FFFFFF, 20f)
        }
        poly(zPct, ::xAt, ::yAt, Color(0xFFE6EDF3), 2f)
        poly(mPct, ::xAt, ::yAt, Color(0xFFF97316), 1.5f)
        for (m in markers) if (m.x in 0 until n) marker(xAt(m.x), yAt(m.y), m.buy)
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
    Canvas(modifier = modifier.fillMaxWidth().height(220.dp)) {
        fun px(v: Double) = (size.width * (v / 100.0)).toFloat()
        fun py(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        // 임계선 20/40/60/80
        for (t in intArrayOf(20, 40, 60, 80)) {
            drawLine(Color(0x22FFFFFF), Offset(px(t.toDouble()), 0f), Offset(px(t.toDouble()), size.height), 1f)
            drawLine(Color(0x22FFFFFF), Offset(0f, py(t.toDouble())), Offset(size.width, py(t.toDouble())), 1f)
        }
        // 중앙선 50
        drawLine(Color(0x55FFFFFF), Offset(px(50.0), 0f), Offset(px(50.0), size.height), 1.2f)
        drawLine(Color(0x55FFFFFF), Offset(0f, py(50.0)), Offset(size.width, py(50.0)), 1.2f)
        // 시간 궤적 점 (파랑→빨강, turbo 느낌) — 또렷하게
        val cold = Color(0xFF2563EB); val hot = Color(0xFFF85149)
        for (i in 0 until n) {
            if (zPct[i].isNaN() || mPct[i].isNaN()) continue
            val c = lerp(cold, hot, i.toFloat() / (n - 1))
            drawCircle(c, 3.2f, Offset(px(zPct[i]), py(mPct[i])))
        }
        // 매매 마커
        for ((idx, buy) in tradeIdx) if (idx in 0 until n) {
            if (!zPct[idx].isNaN() && !mPct[idx].isNaN()) marker(px(zPct[idx]), py(mPct[idx]), buy)
        }
        // 현재 위치 (흰 별 대용: 큰 흰 원 + 검정 테두리)
        val li = n - 1
        if (!zPct[li].isNaN() && !mPct[li].isNaN()) {
            drawCircle(Color.White, 7f, Offset(px(zPct[li]), py(mPct[li])))
            drawCircle(Color.Black, 7f, Offset(px(zPct[li]), py(mPct[li])), style = Stroke(2f))
        }
        // 축 라벨
        label("Z->", size.width - 40f, size.height - 8f, 0x88FFFFFF.toInt(), 22f)
        label("M^", 6f, 22f, 0x88FFFFFF.toInt(), 22f)
    }
}

/** RSI 0~100, 30/70 임계선. */
@Composable
fun RsiChart(rsi: DoubleArray, modifier: Modifier = Modifier) {
    val n = rsi.size
    if (n < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        drawLine(Profit.copy(alpha = 0.5f), Offset(0f, yAt(70.0)), Offset(size.width, yAt(70.0)), 1f)
        drawLine(Loss.copy(alpha = 0.5f), Offset(0f, yAt(30.0)), Offset(size.width, yAt(30.0)), 1f)
        label("70", 4f, yAt(70.0) - 3f, 0x66FFFFFF, 20f)
        label("30", 4f, yAt(30.0) - 3f, 0x66FFFFFF, 20f)
        poly(rsi, ::xAt, ::yAt, Color(0xFF22D3EE), 2f)
    }
}

/** 자산추이(누적손익 $) 라인 + 0 기준선. */
@Composable
fun EquityChart(values: DoubleArray, modifier: Modifier = Modifier) {
    val n = values.size
    if (n < 2) return
    var lo = values.minNaN(); var hi = values.maxNaN()
    if (lo > 0) lo = 0.0
    if (hi < 0) hi = 0.0
    if (hi <= lo) hi = lo + 1.0
    val pad = (hi - lo) * 0.08
    lo -= pad; hi += pad
    Canvas(modifier = modifier.fillMaxWidth().height(140.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()
        val y0 = yAt(0.0)
        drawLine(Color(0x55FFFFFF), Offset(0f, y0), Offset(size.width, y0), 1f)
        poly(values, ::xAt, ::yAt, Color(0xFFF85149), 2f)
        label("$%,.0f".format(hi), 6f, 24f, 0xFFADBAC7.toInt(), 22f)
        label("$%,.0f".format(lo), 6f, size.height - 8f, 0xFFADBAC7.toInt(), 22f)
    }
}

/** MACD(보라) + Signal(흰) + 0선 + 교차 마커(▲빨강 상향 / ▼파랑 하향). */
@Composable
fun MacdChart(macd: DoubleArray, signal: DoubleArray, modifier: Modifier = Modifier) {
    val n = macd.size
    if (n < 2) return
    var mx = 0.0
    for (v in macd) if (!v.isNaN() && kotlin.math.abs(v) > mx) mx = kotlin.math.abs(v)
    for (v in signal) if (!v.isNaN() && kotlin.math.abs(v) > mx) mx = kotlin.math.abs(v)
    if (mx <= 0) mx = 1.0
    mx *= 1.15
    Canvas(modifier = modifier.fillMaxWidth().height(110.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v + mx) / (2 * mx))).toFloat()
        drawLine(Color(0x55FFFFFF), Offset(0f, yAt(0.0)), Offset(size.width, yAt(0.0)), 1f)
        poly(macd, ::xAt, ::yAt, Color(0xFF7C3AED), 2f)
        poly(signal, ::xAt, ::yAt, Color(0xFFE6EDF3), 1f)
        for (i in 1 until n) {
            if (macd[i].isNaN() || signal[i].isNaN() || macd[i - 1].isNaN() || signal[i - 1].isNaN()) continue
            val prev = macd[i - 1] - signal[i - 1]
            val cur = macd[i] - signal[i]
            if (prev < 0 && cur >= 0) marker(xAt(i), yAt(macd[i]), true)
            else if (prev > 0 && cur <= 0) marker(xAt(i), yAt(macd[i]), false)
        }
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
    height: Dp = 340.dp,
    modifier: Modifier = Modifier,
) {
    if (points.isEmpty()) return
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
        for (g in vLines) drawLine(g.color, Offset(sx(g.v), 0f), Offset(sx(g.v), size.height), g.width)
        for (g in hLines) drawLine(g.color, Offset(0f, sy(g.v)), Offset(size.width, sy(g.v)), g.width)

        val xs = FloatArray(points.size) { sx(points[it].x) }
        val ys = FloatArray(points.size) { sy(points[it].y) }
        // 점 (큰 원 + 흰 테두리)
        for (i in points.indices) {
            drawCircle(points[i].color, 13f, Offset(xs[i], ys[i]))
            drawCircle(Color.White, 13f, Offset(xs[i], ys[i]), style = Stroke(2f))
        }
        // 라벨 — 가장 가까운 점 반대 방향(8방향)에 배치
        val r = 17f
        for (i in points.indices) {
            var best = Float.MAX_VALUE; var nd = -1
            for (j in points.indices) {
                if (i == j) continue
                val dx = xs[j] - xs[i]; val dy = ys[j] - ys[i]
                val d = dx * dx + dy * dy
                if (d < best) { best = d; nd = j }
            }
            val ax = if (nd >= 0) -(xs[nd] - xs[i]) else 1f
            val ay = if (nd >= 0) -(ys[nd] - ys[i]) else 0f
            val k = (((Math.toDegrees(kotlin.math.atan2(ay, ax).toDouble()) / 45.0).roundToInt()) % 8 + 8) % 8
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
            label(points[i].label, xs[i] + ox, ys[i] + oy, 0xFFE6EDF3.toInt(), 22f, align)
        }
        if (xAxisLabel.isNotEmpty()) label(xAxisLabel, size.width - 8f, size.height - 8f, 0x88FFFFFF.toInt(), 22f, Paint.Align.RIGHT)
        if (yAxisLabel.isNotEmpty()) label(yAxisLabel, 6f, 22f, 0x88FFFFFF.toInt(), 22f)
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
