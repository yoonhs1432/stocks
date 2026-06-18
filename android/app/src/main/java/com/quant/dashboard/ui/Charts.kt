package com.quant.dashboard.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.unit.dp
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Profit

/**
 * 의존성 없는 Compose Canvas 라인 차트.
 * 외부 차트 라이브러리 없이 가격/밴드·Z·M·RSI를 세로 스택으로 그림.
 */

private fun DoubleArray.minIgnoringNaN(): Double {
    var m = Double.POSITIVE_INFINITY
    for (v in this) if (!v.isNaN() && v < m) m = v
    return if (m.isInfinite()) 0.0 else m
}

private fun DoubleArray.maxIgnoringNaN(): Double {
    var m = Double.NEGATIVE_INFINITY
    for (v in this) if (!v.isNaN() && v > m) m = v
    return if (m.isInfinite()) 1.0 else m
}

/** 가격 + 회귀선 + ±1.5σ 밴드. */
@Composable
fun PriceChart(
    price: DoubleArray,
    predicted: DoubleArray,
    bandUpper: DoubleArray,
    bandLower: DoubleArray,
    tickerNorm: DoubleArray,   // 가격 스케일 정규화에 사용 (norm 기준 동일 축)
    modifier: Modifier = Modifier,
) {
    // predicted/band는 norm 스케일, price는 달러 스케일 → norm 축으로 통일
    val n = tickerNorm.size
    if (n < 2) return
    val ys = ArrayList<DoubleArray>()
    ys.add(tickerNorm); ys.add(predicted); ys.add(bandUpper); ys.add(bandLower)
    var lo = Double.POSITIVE_INFINITY; var hi = Double.NEGATIVE_INFINITY
    for (arr in ys) {
        val a = arr.minIgnoringNaN(); val b = arr.maxIgnoringNaN()
        if (a < lo) lo = a; if (b > hi) hi = b
    }
    if (lo.isInfinite() || hi.isInfinite() || hi <= lo) return
    val pad = (hi - lo) * 0.05
    lo -= pad; hi += pad

    Canvas(modifier = modifier.fillMaxWidth().height(180.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()

        // 밴드 영역
        val band = Path()
        band.moveTo(0f, yAt(bandUpper[0]))
        for (i in 1 until n) band.lineTo(xAt(i), yAt(bandUpper[i]))
        for (i in n - 1 downTo 0) band.lineTo(xAt(i), yAt(bandLower[i]))
        band.close()
        drawPath(band, Color(0x22FFFFFF))

        drawLine(predicted, ::xAt, ::yAt, Color(0xFFADBAC7), 1.5f, n)
        drawLine(tickerNorm, ::xAt, ::yAt, Color(0xFFE6EDF3), 2.5f, n)
    }
}

/** Z·M 백분위(0~100) — 임계선 20/40/60/80. */
@Composable
fun ZmChart(zPct: DoubleArray, mPct: DoubleArray, modifier: Modifier = Modifier) {
    val n = zPct.size
    if (n < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(120.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        for (t in intArrayOf(20, 40, 60, 80)) {
            val y = yAt(t.toDouble())
            drawLine(Color(0x33FFFFFF), Offset(0f, y), Offset(size.width, y), 1f)
        }
        drawLine(zPct, ::xAt, ::yAt, Color(0xFFE6EDF3), 2f, n)   // Z 흰색
        drawLine(mPct, ::xAt, ::yAt, Color(0xFFF97316), 1.5f, n) // M 주황
    }
}

/** RSI(0~100) — 30/70 임계선. */
@Composable
fun RsiChart(rsi: DoubleArray, modifier: Modifier = Modifier) {
    val n = rsi.size
    if (n < 2) return
    Canvas(modifier = modifier.fillMaxWidth().height(90.dp)) {
        fun xAt(i: Int) = size.width * i / (n - 1)
        fun yAt(v: Double) = (size.height * (1 - v / 100.0)).toFloat()
        drawLine(Profit.copy(alpha = 0.5f), Offset(0f, yAt(70.0)), Offset(size.width, yAt(70.0)), 1f)
        drawLine(Loss.copy(alpha = 0.5f), Offset(0f, yAt(30.0)), Offset(size.width, yAt(30.0)), 1f)
        drawLine(rsi, ::xAt, ::yAt, Color(0xFF22D3EE), 2f, n)
    }
}

/** NaN 구간을 건너뛰며 폴리라인을 그림. */
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLine(
    data: DoubleArray,
    xAt: (Int) -> Float,
    yAt: (Double) -> Float,
    color: Color,
    stroke: Float,
    n: Int,
) {
    var prev = -1
    for (i in 0 until n) {
        if (data[i].isNaN()) { prev = -1; continue }
        if (prev >= 0) {
            drawLine(color, Offset(xAt(prev), yAt(data[prev])), Offset(xAt(i), yAt(data[i])), stroke)
        }
        prev = i
    }
}
