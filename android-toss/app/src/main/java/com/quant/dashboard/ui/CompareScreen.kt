package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Text
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.MarketHours
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.Gold
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.Teal
import com.quant.dashboard.ui.theme.TextMuted
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.roundToInt
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import com.quant.dashboard.ui.theme.pctColor
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.ui.graphics.lerp
import com.quant.dashboard.data.Store

// 한국식: 상승=빨강 / 하락=파랑 / 0=회색
private val TableGray = Color(0xFFA4ADB8)
private fun pnColor(v: Double) = when {
    v > 0 -> Profit; v < 0 -> Loss; else -> TableGray
}
// Z 셀: ≥80 파랑(매도) / ≤20 빨강(매수) / 그 외 회색
private fun zCellColor(pct: Double) = when {
    pct >= 80 -> Loss; pct <= 20 -> Profit; else -> TableGray
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CompareScreen(vm: CompareViewModel = viewModel(), onOpenAnalysis: (String) -> Unit = {}) {
    val s = vm.state
    LaunchedEffect(AppState.dataVersion) { vm.sync(AppState.dataVersion) }
    // 자동 새로고침 — 화면 켜진 비교 탭 + 장중에만, 60초 (조용히, 명단은 5분 캐시)
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(60_000)
            if (MarketHours.anyOpen()) vm.autoRefresh()
        }
    }

    Column(modifier = Modifier.fillMaxSize().background(BgApp)) {
        // 표는 위, 조작 버튼은 아래 — 한 손으로 엄지가 닿는 곳에 둔다
        Box(Modifier.fillMaxWidth().weight(1f)) {
            PullToRefreshBox(isRefreshing = s.loading, onRefresh = { AppState.bump() },
                modifier = Modifier.fillMaxSize()) {
                Column(
                    modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                        .padding(horizontal = 12.dp).padding(top = 8.dp, bottom = 4.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    TickStatus(s.market)

                    val rows = vm.sorted()
                    val err = s.error
                    when {
                        rows.isEmpty() && err != null -> Text("⚠️ $err", color = Loss)
                        s.rows.isEmpty() -> Text("불러오는 중…", color = TextSecondary,
                            modifier = Modifier.padding(24.dp))
                        else -> {
                            // 표는 **간격 없는** 안쪽 Column 에 넣는다. 바깥 Column 의 spacedBy 가
                            // 행마다·구분선마다 붙으면 행 하나당 12dp 가 더 생겨 오히려 벌어진다.
                            Column(Modifier.fillMaxWidth()) {
                                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(6.dp)).background(SurfaceInput)
                                    .padding(vertical = 3.dp, horizontal = 2.dp)) {
                                    HCell(vm, "종목", SortKey.NAME, 2.4f, TextAlign.Start)
                                    HCell(vm, "현재가", SortKey.PRICE, 2f)
                                    HCell(vm, "일", SortKey.DAY, 1.4f)
                                    HCell(vm, "Z", SortKey.Z, 1f)
                                    HCell(vm, "M", SortKey.M, 1f)
                                }
                                if (rows.isEmpty()) {
                                    Text("표시할 종목이 없습니다", color = TextSecondary, fontSize = 13.sp,
                                        modifier = Modifier.padding(12.dp))
                                }
                                rows.forEachIndexed { idx, r ->
                                    QuoteRow(vm, r, onOpenAnalysis)
                                    if (idx < rows.lastIndex)
                                        Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                                }
                            }
                            Text("● 보유 / ○ 이력 · 행 탭=분석 이동 · 헤더 탭=정렬 · 흐림=이번 장 체결 없음(직전 종가)",
                                color = TextSecondary, fontSize = 11.sp)
                        }
                    }
                }
            }
        }
        BottomBar(vm)
    }
}

/** 하단 고정 조작 바 — 시장 전환과 보유 필터. 탭바 바로 위라 한 손으로 닿는다. */
@Composable
private fun BottomBar(vm: CompareViewModel) {
    val s = vm.state
    Row(
        Modifier.fillMaxWidth().background(BgApp).padding(horizontal = 12.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        // 한쪽 시장 종목만 있으면 전환 버튼이 의미가 없다
        if (vm.hasBothMarkets()) {
            listOf("US" to "미국", "KR" to "국내").forEach { (id, label) ->
                val on = s.market == id
                Box(
                    Modifier.clip(RoundedCornerShape(8.dp))
                        .background(if (on) Teal else SurfaceInput)
                        .clickable { vm.setMarket(id) }
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                ) {
                    Text(label, color = if (on) Color(0xFF0C0E11) else TextSecondary,
                        fontSize = 13.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
        Spacer(Modifier.weight(1f))
        Box(
            Modifier.clip(RoundedCornerShape(8.dp))
                .background(if (s.holdingsOnly) Gold else SurfaceInput)
                .clickable { vm.toggleHoldings() }
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            Text("● 보유", color = if (s.holdingsOnly) Color(0xFF0C0E11) else TextSecondary,
                fontSize = 13.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun RowScope.HCell(vm: CompareViewModel, text: String, key: SortKey, weight: Float, align: TextAlign = TextAlign.End) {
    val s = vm.state
    val on = s.sortKey == key
    val mark = if (on) (if (s.sortDesc) " ▼" else " ▲") else ""
    Text(
        text + mark, color = if (on) TextPrimary else TextSecondary, fontSize = 14.5.sp,
        fontWeight = FontWeight.SemiBold, textAlign = align,
        modifier = Modifier.weight(weight).clickable { vm.setSort(key) },
    )
}

@Composable
private fun RowScope.Cell(text: String, weight: Float, color: Color, align: TextAlign = TextAlign.End,
                          fw: FontWeight = FontWeight.Normal, bg: Color = Color.Transparent) {
    Text(text, color = color, fontSize = 15.sp, textAlign = align, fontWeight = fw,
        fontFamily = Mono, maxLines = 1,
        modifier = Modifier.weight(weight).clip(RoundedCornerShape(4.dp)).background(bg))
}

/** 종목 셀 — 상태 점(보유=금채움/이력=금링) + 티커(모노). */
@Composable
private fun RowScope.DotName(state: Int, name: String, weight: Float, alpha: Float = 1f) {
    Row(Modifier.weight(weight), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        if (state == 0) Spacer(Modifier.size(6.dp)) else Box(
            Modifier.size(6.dp).clip(RoundedCornerShape(50))
                .then(if (state == 2) Modifier.background(Gold.copy(alpha = alpha))
                else Modifier.border(1.3.dp, Gold.copy(alpha = alpha), RoundedCornerShape(50))),
        )
        Text(name, color = TextPrimary.copy(alpha = alpha), fontSize = 15.sp, maxLines = 1,
            fontWeight = FontWeight.SemiBold, fontFamily = Mono)
    }
}

private fun signed(v: Double) = (if (v >= 0) "+" else "") + "%.1f%%".format(v)


/**
 * 실시간 틱 상태 — 갱신될 때마다 점이 밝아졌다 사그라들고, **안 도는 경우 그 사유**를 보여준다.
 * 조용히 멈추면 고장과 구분이 안 되므로 사유 표시가 핵심이다.
 */
@Composable
private fun TickStatus(market: String) {
    val seq = LivePrices.tickSeq
    val note = LivePrices.note
    val at = LivePrices.updatedAt
    // 상단 시장 헤더를 없앴으므로 지금 어느 장이 열려 있는지를 여기서 알린다
    val session = MarketHours.labelFor(market) ?: "장 마감"
    // 설정 파일을 매 틱마다 읽지 않도록 — 설정 변경은 dataVersion 을 올린다
    val sec = remember(AppState.dataVersion) { Store.tickSeconds() }

    // 틱이 들어올 때마다 1 → 0 으로 감쇠 (Compose 가 seq 변화를 보고 다시 실행)
    val glow = remember { Animatable(0f) }
    LaunchedEffect(seq) {
        if (seq > 0) { glow.snapTo(1f); glow.animateTo(0f, tween(700)) }
    }
    val dot = if (note != null) TextMuted else lerp(Color(0xFF2EA078), Color(0xFF7BFFCB), glow.value)

    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
            .padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(dot))
        Text(
            (note ?: if (sec > 0) "실시간 ${sec}초" else "갱신 꺼짐") + " · $session",
            color = if (note != null) Gold else TextSecondary, fontSize = 11.sp,
        )
        Spacer(Modifier.weight(1f))
        if (at > 0) {
            Text(
                remember(at) { SimpleDateFormat("HH:mm:ss", Locale.US).format(Date(at)) },
                color = TextMuted, fontSize = 11.sp,
            )
        }
    }
}

/**
 * 비교 표 한 줄. 값이 실제로 바뀐 종목은 현재가 칸이 잠깐 물들었다 사라진다(체결 플래시).
 * 이번 장에 체결이 없는 종목(직전 종가)은 흐리게 표시한다.
 */
@Composable
private fun QuoteRow(vm: CompareViewModel, r: CompareRow, onOpenAnalysis: (String) -> Unit) {
    // 표시값과 정렬값이 어긋나지 않도록 같은 함수를 쓴다
    val px = vm.shownPrice(r)
    val day = vm.shownDay(r)
    val stale = LivePrices.isStale(r.ticker, if (Tickers.isKrw(r.ticker)) "KR" else "US")

    // 가격이 바뀐 틱에서만 배경을 깔았다 지운다
    val flash = remember { Animatable(0f) }
    val seq = LivePrices.tickSeq
    LaunchedEffect(seq) {
        if (r.ticker in LivePrices.changed) { flash.snapTo(1f); flash.animateTo(0f, tween(600)) }
    }
    val flashBg = (if (day >= 0) Profit else Loss).copy(alpha = 0.22f * flash.value)
    val alpha = if (stale) 0.45f else 1f

    Row(
        // 여백을 줄여 한 화면에 더 많은 종목이 들어가게
        Modifier.fillMaxWidth().clickable { onOpenAnalysis(r.ticker) }
            .padding(horizontal = 2.dp, vertical = 1.dp),
    ) {
        DotName((if (r.holding) 2 else if (r.hasHistory) 1 else 0), r.name, 2.4f, alpha = alpha)
        Cell(Tickers.priceLabel(r.ticker, px), 2f, pnColor(day).copy(alpha = alpha), bg = flashBg)
        Cell(signed(day), 1.4f, pnColor(day).copy(alpha = alpha))
        Cell("%.0f".format(r.zPct), 1f, pctColor(r.zPct).copy(alpha = alpha), fw = FontWeight.Bold)
        Cell("%.0f".format(r.mPct), 1f, pctColor(r.mPct).copy(alpha = alpha), fw = FontWeight.Bold)
    }
}
