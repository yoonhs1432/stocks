package com.quant.dashboard.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TextMuted

/*
 * 리밸런싱 화면용 작은 그래프들 — 목업 research/mockup_v4_I.png 의 모양.
 * 색: 상승·매수 = 빨강, 하락·매도 = 파랑(한국식). 자산군 5색은 dataviz 검증기로 색약 구분을 통과한 순서.
 */

/** 자산군 색·이름 — 순서가 곧 색약 안전 순서(청록·주황·보라·노랑·자홍). */
val GROUP_ORDER = listOf("us", "intl", "sector", "real", "bond", "etc")
val GROUP_COLOR = mapOf(
    "us" to Color(0xFF199E70), "intl" to Color(0xFFD95926), "sector" to Color(0xFF9085E9),
    "real" to Color(0xFFC98500), "bond" to Color(0xFFD55181), "etc" to Color(0xFF6B7684),
)
val GROUP_LABEL = mapOf(
    "us" to "미국 지수", "intl" to "해외", "sector" to "업종·테마", "real" to "원자재·코인", "bond" to "채권", "etc" to "기타",
)
val CASH_COLOR = Color(0xFF2A2A33)
val CARD = Color(0xFF17171C)
val CARD2 = Color(0xFF1F1F26)
val OK = Color(0xFF2EA078)

/** 총자산(빨강 영역) + 원금(점선 계단) — 한 축. */
@Composable
fun AssetChart(total: DoubleArray, principal: DoubleArray, height: Dp, modifier: Modifier = Modifier) {
    Canvas(modifier.fillMaxWidth().height(height)) {
        if (total.size < 2) return@Canvas
        val vals = (total.filter { it.isFinite() } + principal.filter { it.isFinite() && it > 0 })
        val mn = vals.minOrNull() ?: return@Canvas
        val mx = (vals.maxOrNull() ?: mn).let { if (it - mn < 1e-6) mn + 1 else it }
        fun x(i: Int) = size.width * i / (total.size - 1)
        fun y(v: Double) = (size.height - 4f) - ((v - mn) / (mx - mn)).toFloat() * (size.height - 8f)
        val line = Path()
        total.forEachIndexed { i, v -> if (i == 0) line.moveTo(x(i), y(v)) else line.lineTo(x(i), y(v)) }
        val area = Path().apply { addPath(line); lineTo(size.width, size.height); lineTo(0f, size.height); close() }
        drawPath(area, Brush.verticalGradient(listOf(Profit.copy(alpha = 0.30f), Profit.copy(alpha = 0f))))
        if (principal.size == total.size) {
            val p = Path()
            var started = false
            principal.forEachIndexed { i, v ->
                if (!v.isFinite() || v <= 0) return@forEachIndexed
                if (!started) { p.moveTo(x(i), y(v)); started = true } else p.lineTo(x(i), y(v))
            }
            drawPath(p, TextMuted, style = Stroke(width = 3f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 10f))))
        }
        drawPath(line, Profit, style = Stroke(width = 4.5f, cap = StrokeCap.Round))
    }
}

/** 작은 추이 — 오르면 빨강, 내리면 파랑. */
@Composable
fun Sparkline(values: DoubleArray, width: Dp, height: Dp) {
    Canvas(Modifier.size(width, height)) {
        if (values.size < 2) return@Canvas
        val mn = values.min(); val mx = values.max().let { if (it - mn < 1e-9) mn + 1 else it }
        val col = if (values.last() >= values.first()) Profit else Loss
        val p = Path()
        values.forEachIndexed { i, v ->
            val x = size.width * i / (values.size - 1)
            val y = size.height - ((v - mn) / (mx - mn)).toFloat() * size.height
            if (i == 0) p.moveTo(x, y) else p.lineTo(x, y)
        }
        drawPath(p, col, style = Stroke(width = 3.5f, cap = StrokeCap.Round))
    }
}

/** 가로 쌓은 막대 — 조각 사이 2dp 틈. */
@Composable
fun StackBar(parts: List<Pair<Double, Color>>, height: Dp, modifier: Modifier = Modifier) {
    val shown = parts.filter { it.first > 0.0005 }
    Row(modifier.fillMaxWidth().height(height).clip(RoundedCornerShape(height / 2))) {
        shown.forEachIndexed { i, (v, c) ->
            Box(Modifier.weight(v.toFloat()).fillMaxHeight().background(c))
            if (i < shown.size - 1) Box(Modifier.width(2.dp).fillMaxHeight())
        }
    }
}

/** 반원 게이지 0..1. */
@Composable
fun SemiGauge(p: Double, width: Dp, color: Color = Profit) {
    Canvas(Modifier.size(width, width / 2 + 8.dp)) {
        val sw = 28f
        val d = size.width - sw
        val tl = Offset(sw / 2, sw / 2)
        drawArc(CARD2, 180f, 180f, false, tl, Size(d, d), style = Stroke(sw, cap = StrokeCap.Round))
        drawArc(color, 180f, (180f * p.coerceIn(0.0, 1.0)).toFloat(), false, tl, Size(d, d), style = Stroke(sw, cap = StrokeCap.Round))
    }
}
