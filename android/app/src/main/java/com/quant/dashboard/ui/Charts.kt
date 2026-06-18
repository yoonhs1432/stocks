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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TextSecondary
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** 의존성 없는 Compose Canvas 차트. 가격($)·Z·M·RSI를 세로 스택으로. */

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
        label("$%,.0f".format(hi), 6f, 24f, gray, 24f)
        label("$%,.0f".format(lo), 6f, size.height - 10f, gray, 24f)
    }
}

/** Z(흰)·M(주황) 백분위 0~100, 임계선 20/40/60/80. */
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
            label(t.toString(), 4f, y - 3f, 0x66FFFFFF, 20f)
        }
        poly(zPct, ::xAt, ::yAt, Color(0xFFE6EDF3), 2f)
        poly(mPct, ::xAt, ::yAt, Color(0xFFF97316), 1.5f)
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
