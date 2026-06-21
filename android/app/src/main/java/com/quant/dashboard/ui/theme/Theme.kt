package com.quant.dashboard.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.font.FontFamily

// ─────────────────────────────────────────────────────────────
// 디자인 토큰 (다크 리디자인 Direction A — 퀀트 터미널)
// 한국식: 상승/매수/이익 = 빨강, 하락/매도/손실 = 파랑
// ─────────────────────────────────────────────────────────────

// 배경/표면
val BgApp = Color(0xFF0C0E11)          // 화면 배경(거의 검정)
val BgElevated = Color(0xFF0A0C0F)     // 탭바 배경
val BgCard = Color(0xFF15181D)         // 카드·차트 컨테이너
val SurfaceInput = Color(0xFF0C0E11)   // 입력칸·세그먼트 비활성
val SegmentOn = Color(0xFF262B32)      // 세그먼트 활성
val ChipOn = Color(0xFF33373F)         // 선택형 칩 활성
val BorderColor = Color(0x0DFFFFFF)    // 카드 헤어라인 (white 5%)
val DividerColor = Color(0x14FFFFFF)   // 구분선 (white 8%)

// 텍스트
val TextPrimary = Color(0xFFEEF1F4)
val TextSecondary = Color(0xFFAEB6BF)
val TextMuted = Color(0xFF727B85)

// 시그널 컬러 (한국식)
val Profit = Color(0xFFEF6066)         // 상승/매수/이익 (선·텍스트)
val ProfitBtn = Color(0xFFEF4D57)      // 매수 버튼
val Loss = Color(0xFF5B9BF2)           // 하락/매도/손실
val LossBtn = Color(0xFF4D8DF0)        // 매도 버튼
val Neutral = Color(0xFF9AA3AD)

// 액센트
val Gold = Color(0xFFE0A24A)           // 보유 표식·금리·M 컬럼
val Teal = Color(0xFF37B6C4)           // RSI 선
val Violet = Color(0xFF9B8CFF)         // MACD 선
val SignalGrey = Color(0xFFC9C5BB)     // Signal 선·M 오실레이터

// 탭바
val TabActive = Color(0xFFEF6066)
val TabInactive = Color(0xFF5F6873)

// 보유 박스 (녹색 테두리)
val HoldingBorder = Color(0x662EA078)  // rgba(46,160,120,0.4)
val HoldingBg = Color(0x122EA078)      // rgba(46,160,120,0.07)

// 포트폴리오 비중/식별 색 팔레트 (6)
val WeightPalette = listOf(
    Color(0xFFE0A24A), Color(0xFFD9694E), Color(0xFFCF5D7F),
    Color(0xFF8A6FD0), Color(0xFF4D8DF0), Color(0xFF37A48C),
)

/** 모든 숫자/티커/지표 — 모노스페이스 (IBM Plex Mono 대용: 시스템 모노). */
val Mono = FontFamily.Monospace

/** 5단계 신호 라벨 → 색. */
fun signalColor(signal: String): Color = when (signal) {
    "strong_buy" -> Profit
    "buy" -> Color(0xFFEF8A8E)
    "hold" -> Neutral
    "sell" -> Color(0xFF8FB8F5)
    "strong_sell" -> Loss
    else -> Neutral
}

/** 백분위(0~100) → 텍스트/포인트 색 (저=매수 빨강 / 고=매도 파랑). */
fun pctColor(pct: Double): Color = when {
    pct < 20 -> Profit
    pct < 40 -> Color(0xFFEF8A8E)
    pct < 60 -> Neutral
    pct < 80 -> Color(0xFF8FB8F5)
    else -> Loss
}

/** 워치리스트 M 히트맵 배경 — 저 M(매수권)=빨강 틴트 → 고 M(매도권)=파랑 틴트 (카드 위 은은하게). */
fun mHeat(pct: Double): Color {
    val t = (pct / 100.0).coerceIn(0.0, 1.0).toFloat()
    val hue = lerp(Profit, Loss, t)
    return lerp(BgCard, hue, 0.28f)
}

private val DarkColors = darkColorScheme(
    primary = ProfitBtn,
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
