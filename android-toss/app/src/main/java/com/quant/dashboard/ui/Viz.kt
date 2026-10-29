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
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.sp
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.quant.dashboard.ui.theme.Gold
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

/**
 * 종목 상세 가격 차트 — 종가 영역 + 금색 점선 평단 + 매매 점(빨강 B = 매수, 파랑 S = 매도).
 * [marks] 는 (봉 인덱스, 매수 여부). 평단이 범위 밖이면 축을 넓혀 늘 보이게 한다.
 */
@Composable
fun PriceChart(closes: DoubleArray, avg: Double?, avgLabel: String?, marks: List<Pair<Int, Boolean>>, height: Dp, modifier: Modifier = Modifier) {
    val tm = rememberTextMeasurer()
    Canvas(modifier.fillMaxWidth().height(height)) {
        if (closes.size < 2) return@Canvas
        val a = avg?.takeIf { it.isFinite() && it > 0 }
        var mn = closes.min(); var mx = closes.max()
        if (a != null) { mn = minOf(mn, a); mx = maxOf(mx, a) }
        val pad = (mx - mn).coerceAtLeast(mx * 1e-4) * 0.08
        mn -= pad; mx += pad
        val top = 14.dp.toPx(); val bot = size.height - 12.dp.toPx()
        fun x(i: Int) = size.width * i / (closes.size - 1)
        fun y(v: Double) = bot - ((v - mn) / (mx - mn)).toFloat() * (bot - top)
        val up = closes.last() >= closes.first()
        val col = if (up) Profit else Loss
        val line = Path()
        closes.forEachIndexed { i, v -> if (i == 0) line.moveTo(x(i), y(v)) else line.lineTo(x(i), y(v)) }
        val area = Path().apply { addPath(line); lineTo(size.width, size.height); lineTo(0f, size.height); close() }
        drawPath(area, Brush.verticalGradient(listOf(col.copy(alpha = 0.28f), col.copy(alpha = 0f))))
        drawPath(line, col, style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round))
        if (a != null) {
            val ya = y(a)
            drawLine(Gold, Offset(0f, ya), Offset(size.width, ya), strokeWidth = 1.5.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 8f)))
            avgLabel?.let {
                val lay = tm.measure(it, TextStyle(color = Gold, fontSize = 10.sp, fontWeight = FontWeight.Bold))
                val ly = if (ya - lay.size.height - 2f < 0) ya + 2f else ya - lay.size.height - 2f
                drawText(lay, topLeft = Offset(4f, ly))
            }
        }
        // 매매 점 — 원 + 글자, 바깥에 배경색 테두리(겹쳐도 구분되게)
        val r = 8.dp.toPx()
        marks.forEach { (i, buy) ->
            if (i !in closes.indices) return@forEach
            val c = Offset(x(i).coerceIn(r, size.width - r), y(closes[i]))
            drawCircle(CARD, r + 2.dp.toPx(), c)
            drawCircle(if (buy) Profit else Loss, r, c)
            val lay = tm.measure(if (buy) "B" else "S", TextStyle(color = Color.White, fontSize = 9.sp, fontWeight = FontWeight.ExtraBold))
            drawText(lay, topLeft = Offset(c.x - lay.size.width / 2f, c.y - lay.size.height / 2f))
        }
    }
}

/**
 * 목표 대비 막대 — 가운데 흰 선 = 목표(0%). 오른쪽 파랑 = 많음(팔기), 왼쪽 빨강 = 모자람(사기).
 * 눈금 ±10·20%, ±[lim]% 를 넘으면 끝에 흰 화살표. [muted] 면 회색(목표 근처 — 할 일 없음).
 */
@Composable
fun DevBar(d: Double, modifier: Modifier = Modifier, height: Dp = 10.dp, lim: Double = 30.0, muted: Boolean = false) {
    Canvas(modifier.fillMaxWidth().height(height + 6.dp)) {
        val half = size.width / 2
        val cy = size.height / 2
        val h = height.toPx()
        drawRoundRect(CARD2, Offset(0f, cy - 2.dp.toPx()), Size(size.width, 4.dp.toPx()),
            androidx.compose.ui.geometry.CornerRadius(2.dp.toPx()))
        listOf(-20.0, -10.0, 10.0, 20.0).forEach { v ->
            val x = half + (v / lim).toFloat() * half
            drawLine(Color(0xFF3A3A44), Offset(x, cy - 3.dp.toPx()), Offset(x, cy + 3.dp.toPx()), strokeWidth = 1.dp.toPx())
        }
        val f = (minOf(kotlin.math.abs(d), lim) / lim).toFloat() * half
        val col = if (muted) TextMuted else if (d >= 0) Loss else Profit
        val x0 = if (d >= 0) half else half - f
        drawRoundRect(col, Offset(x0, cy - h / 2), Size(maxOf(f, 1.5f), h), androidx.compose.ui.geometry.CornerRadius(3.dp.toPx()))
        if (kotlin.math.abs(d) > lim) {
            val a = 6.dp.toPx()
            val p = Path().apply {
                if (d > 0) { moveTo(size.width, cy); lineTo(size.width - a, cy - a * 0.8f); lineTo(size.width - a, cy + a * 0.8f) }
                else { moveTo(0f, cy); lineTo(a, cy - a * 0.8f); lineTo(a, cy + a * 0.8f) }
                close()
            }
            drawPath(p, Color.White)
        }
        drawLine(Color.White, Offset(half, 0f), Offset(half, size.height), strokeWidth = 2.5.dp.toPx())
    }
}
