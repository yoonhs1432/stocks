package com.quant.dashboard.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.font.FontFamily

// ─────────────────────────────────────────────────────────────
// 디자인 토큰 — A-1 "토스 블루" (미니멀 다크)
//  · 카드 없음. 배경 한 톤 + 1px 구분선으로만 나눈다
//  · 액센트는 파랑 하나(버튼·선택·탭). 빨강/파랑은 손익 숫자에만
//  · 한국식: 상승/매수/이익 = 빨강, 하락/매도/손실 = 파랑
// ─────────────────────────────────────────────────────────────

// 배경/표면
val BgApp = Color(0xFF101013)          // 화면 배경
val BgElevated = Color(0xFF101013)     // 탭바 배경 (배경과 같음 — 구분선으로만)
val BgCard = Color(0xFF101013)         // (카드 제거) 배경과 같음. 남은 참조 호환용
val SurfaceInput = Color(0xFF1B1B20)   // 입력칸·고스트 버튼
val SegmentOn = Color(0xFF1F2024)      // (호환) 세그먼트 활성 배경 — 새 UI 는 밑줄 방식
val ChipOn = Color(0xFF1F2024)         // (호환) 선택형 칩 활성
val BorderColor = Color(0xFF24242A)    // 구분선
val DividerColor = Color(0xFF24242A)   // 구분선

// 텍스트
val TextPrimary = Color(0xFFF2F4F6)
val TextSecondary = Color(0xFF8B95A1)
val TextMuted = Color(0xFF6B7684)

// 액센트 — UI 전용 (버튼·선택·탭·포커스)
val Accent = Color(0xFF3182F6)
val OnAccent = Color(0xFFFFFFFF)
val Ghost = Color(0xFF1F2024)          // 보조 버튼 배경

// 시그널 컬러 (한국식) — 손익 숫자·차트에만
val Profit = Color(0xFFF04452)         // 상승/매수/이익
val ProfitBtn = Profit                 // (호환)
val Loss = Color(0xFF3182F6)           // 하락/매도/손실
val LossBtn = Loss                     // (호환)
val Neutral = Color(0xFF8B95A1)

// 차트 전용 액센트
val Gold = Color(0xFFE0A24A)           // 보유 표식·M 컬럼
val Teal = Color(0xFF37B6C4)           // RSI 선
val Violet = Color(0xFF9B8CFF)         // MACD 선
val SignalGrey = Color(0xFFC9C5BB)     // Signal 선

// 탭바
val TabActive = Accent
val TabInactive = Color(0xFF6B7684)

// 보유 박스 (호환)
val HoldingBorder = Color(0x662EA078)
val HoldingBg = Color(0x122EA078)

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
    primary = Accent,            // M3 Button·TextField 포커스 = 파랑
    onPrimary = OnAccent,
    background = BgApp,
    surface = BgApp,
    onBackground = TextPrimary,
    onSurface = TextPrimary,
    outline = BorderColor,
)

@Composable
fun QuantTheme(content: @Composable () -> Unit) {
    // 항상 다크 (대시보드 톤 고정)
    MaterialTheme(colorScheme = DarkColors, content = content)
}
