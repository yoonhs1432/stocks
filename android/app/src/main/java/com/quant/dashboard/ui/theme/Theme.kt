package com.quant.dashboard.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// 한국식: 매수/수익 = 빨강, 매도/손실 = 파랑
val Profit = Color(0xFFDC2626)
val Loss = Color(0xFF2563EB)
val Neutral = Color(0xFF9CA3AF)
val BgApp = Color(0xFF0D1117)
val BgCard = Color(0xFF161B22)
val TextPrimary = Color(0xFFF0F6FC)
val TextSecondary = Color(0xFFC9D1D9)
val BorderColor = Color(0xFF30363D)

/** 5단계 신호 라벨 → 색. */
fun signalColor(signal: String): Color = when (signal) {
    "strong_buy" -> Color(0xFFDC2626)
    "buy" -> Color(0xFFFCA5A5)
    "hold" -> Neutral
    "sell" -> Color(0xFF93C5FD)
    "strong_sell" -> Color(0xFF2563EB)
    else -> Neutral
}

/** 백분위(0~100) → 색 (20/40/60/80 5단계). */
fun pctColor(pct: Double): Color = when {
    pct < 20 -> Color(0xFFDC2626)
    pct < 40 -> Color(0xFFFCA5A5)
    pct < 60 -> Neutral
    pct < 80 -> Color(0xFF93C5FD)
    else -> Color(0xFF2563EB)
}

private val DarkColors = darkColorScheme(
    primary = Profit,
    background = BgApp,
    surface = BgCard,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
)

@Composable
fun QuantTheme(content: @Composable () -> Unit) {
    // 항상 다크 (대시보드 톤 고정)
    MaterialTheme(colorScheme = DarkColors, content = content)
}
