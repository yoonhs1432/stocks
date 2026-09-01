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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Slider
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
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.MarketHours
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.Rankings
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
        // 당겨서 새로고침 → 전체 탭 새로고침 (dataVersion bump로 모든 탭이 재로드)
        PullToRefreshBox(isRefreshing = s.loading || s.topLoading, onRefresh = { AppState.bump() },
            modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(if (s.showTop) Rankings.titleOf(s.rankMarket, s.rankType, s.rankDuration, s.rankNote != null) else "종목 비교",
                        color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold,
                        modifier = Modifier.weight(1f))
                    // 워치리스트 ↔ 미국 시총 상위 30개 전환
                    Box(
                        Modifier.clip(RoundedCornerShape(8.dp))
                            .background(if (s.showTop) Teal else SurfaceInput)
                            .clickable { vm.toggleTop() }
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                    ) {
                        Text(if (s.showTop) "내 목록" else "TOP ${Rankings.COUNT}",
                            color = if (s.showTop) Color(0xFF0C0E11) else TextSecondary,
                            fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }
                    // 보유종목만 보기 토글
                    Box(
                        Modifier.clip(RoundedCornerShape(8.dp))
                            .background(if (s.holdingsOnly) Gold else SurfaceInput)
                            .clickable { vm.toggleHoldings() }
                            .padding(horizontal = 12.dp, vertical = 6.dp),
                    ) {
                        Text("● 보유", color = if (s.holdingsOnly) Color(0xFF0C0E11) else TextSecondary,
                            fontSize = 13.sp, fontWeight = FontWeight.Bold)
                    }
                }
                TickStatus()
                // ── 랭킹 기준 선택 (미장 TOP 목록에서만) ──
                if (s.showTop) {
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Rankings.MARKETS.forEach { (id, label) ->
                            RankChip(label, s.rankMarket == id) { vm.setRankMarket(id) }
                        }
                        Spacer(Modifier.size(6.dp))
                        Rankings.TYPES.forEach { (id, label) ->
                            RankChip(label, s.rankType == id) { vm.setRankType(id) }
                        }
                    }
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Rankings.durationsFor(s.rankType).forEach { (id, label) ->
                            RankChip(label, s.rankDuration == id, small = true) { vm.setRankDuration(id) }
                        }
                    }
                    s.rankNote?.let { Text("⚠️ $it", color = Gold, fontSize = 11.sp) }
                }
                val active = vm.activeRows()
                val err = if (s.showTop) s.topError else s.error
                when {
                    active.isEmpty() && err != null -> Text("⚠️ $err", color = Loss)
                    active.isEmpty() -> Text(
                        if (s.showTop) "${Rankings.marketLabel(s.rankMarket)} TOP ${Rankings.COUNT} 불러오는 중…" else "불러오는 중…",
                        color = TextSecondary, modifier = Modifier.padding(24.dp))
                    else -> {
                Column(Modifier.fillMaxWidth()) {
                    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                        .padding(vertical = 6.dp, horizontal = 2.dp)) {
                        HCell(vm, "종목", SortKey.NAME, 2.4f, TextAlign.Start)
                        HCell(vm, "현재가", SortKey.PRICE, 2f)
                        HCell(vm, "일", SortKey.DAY, 1.4f)
                        HCell(vm, "Z", SortKey.Z, 1f)
                        HCell(vm, "M", SortKey.M, 1f)
                    }
                    val rows = vm.sorted()
                    if (rows.isEmpty())
                        Text("보유 종목이 없습니다", color = TextSecondary, fontSize = 13.sp,
                            modifier = Modifier.padding(12.dp))
                    // 미장·국장을 섞어 놓으면 장 상태가 달라 등락률을 나란히 읽기 어렵다
                    val us = rows.filter { !Tickers.isKrw(it.ticker) }
                    val kr = rows.filter { Tickers.isKrw(it.ticker) }
                    listOf("미국" to us, "국내" to kr).forEach { (label, list) ->
                        if (list.isEmpty()) return@forEach
                        if (us.isNotEmpty() && kr.isNotEmpty()) {
                            Text(label, color = TextMuted, fontSize = 11.sp, fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(top = 8.dp, bottom = 2.dp))
                        }
                        list.forEachIndexed { idx, r ->
                            QuoteRow(vm, r, s.showTop, vm.rankOf(r.ticker) + 1, onOpenAnalysis)
                            if (idx < list.lastIndex)
                                Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                        }
                    }
                }
                Text(
                    if (s.showTop) "숫자=랭킹 순위 · 행 탭=분석 이동 · 헤더 탭=정렬 (토스는 시총 랭킹을 주지 않아 거래대금 기준이 기본)"
                    else "● 보유 / ○ 이력 · 행 탭=분석 이동 · 헤더 탭=정렬 · 흐림=이번 장 체결 없음(직전 종가)",
                    color = TextSecondary, fontSize = 11.sp)

                // ── Z·M 사분면 (주간 날짜 스크럽) ──
                val weekDates = OverviewRepo.weekDates()
                val wCount = weekDates.size
                var weekIdx by remember(wCount) { mutableStateOf(wCount - 1) }
                val idx = weekIdx.coerceIn(0, maxOf(wCount - 1, 0))
                val df = remember { SimpleDateFormat("yy.MM.dd", Locale.US) }
                val isLatest = wCount == 0 || idx == wCount - 1
                val dateStr = if (wCount > 0) df.format(Date(weekDates[idx] * 1000L)) else ""
                Text("🎯 종목별 Z·M 위치 — ${if (isLatest) "현재" else dateStr}", color = TextPrimary,
                    fontSize = 13.sp, fontWeight = FontWeight.Bold)
                val gMid = Color(0x998B949E); val gBuy = Color(0x66DC2626); val gSell = Color(0x661D4ED8)
                val zmLines = listOf(GridLine(50.0, gMid, 1.0f), GridLine(10.0, gBuy, 0.8f), GridLine(90.0, gSell, 0.8f))
                ScatterChart(
                    points = vm.visibleRows().map { row ->
                        val z = if (idx < row.zHist.size) row.zHist[idx] else row.zPct
                        val m = if (idx < row.mHist.size) row.mHist[idx] else row.mPct
                        ScatterPt(z, m, row.name, pctColor(if (m.isNaN()) 50.0 else m))
                    },
                    xMin = -5.0, xMax = 105.0, yMin = -5.0, yMax = 105.0,
                    vLines = zmLines, hLines = zmLines,
                    xAxisLabel = "Z (가격 위치 0~100)", yAxisLabel = "M (모멘텀 0~100)", labelTopCenter = true, height = 360.dp,
                )
                // 주간 날짜 슬라이더 (스냅 + 햅틱)
                if (wCount > 1) {
                    val haptic = LocalHapticFeedback.current
                    Slider(
                        value = idx.toFloat(),
                        onValueChange = { v ->
                            val ni = v.roundToInt().coerceIn(0, wCount - 1)
                            if (ni != weekIdx) { weekIdx = ni; haptic.performHapticFeedback(HapticFeedbackType.LongPress) }
                        },
                        valueRange = 0f..(wCount - 1).toFloat(),
                        steps = (wCount - 2).coerceAtLeast(0),
                    )
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(df.format(Date(weekDates[0] * 1000L)), color = TextSecondary, fontSize = 10.sp)
                        Text("← 일별 (최근 6개월) →", color = TextMuted, fontSize = 10.sp)
                        Text("현재", color = TextSecondary, fontSize = 10.sp)
                    }
                }
                Text("X=Z(가격 위치), Y=M(모멘텀) · Q1↑↑=강세 / Q3↓↓=약세 · 임계선 10/50/90",
                    color = TextSecondary, fontSize = 11.sp)

                // ── σ·β 산점도 (변동성·시장민감도) ──
                Text("📊 변동성(σ) · 시장민감도(β)", color = TextPrimary,
                    fontSize = 13.sp, fontWeight = FontWeight.Bold)
                val finite = vm.visibleRows().filter { it.beta.isFinite() && it.sigmaPct.isFinite() && it.sigmaPct > 0 }
                val betas = finite.map { it.beta }
                val sigmas = finite.map { it.sigmaPct }
                val medB = if (betas.isNotEmpty()) betas.sorted()[betas.size / 2] else 0.0
                val medS = if (sigmas.isNotEmpty()) sigmas.sorted()[sigmas.size / 2] else 1.0
                ScatterChart(
                    points = finite.map { ScatterPt(it.beta, it.sigmaPct, it.name, pctColor(it.mPct)) },
                    xMin = (betas.minOrNull() ?: -1.0) - 0.8,
                    xMax = (betas.maxOrNull() ?: 1.0) + 1.0,
                    yMin = maxOf((sigmas.minOrNull() ?: 1.0) * 0.8, 0.5),
                    yMax = (sigmas.maxOrNull() ?: 10.0) * 1.25,
                    yLog = true,
                    vLines = listOf(GridLine(medB, Color(0x88E5E7EB), 1.2f), GridLine(0.0, Color(0x99768390), 1.2f)),
                    hLines = listOf(GridLine(medS, Color(0x88E5E7EB), 1.2f)),
                    xAxisLabel = "β", yAxisLabel = "σ% (변동성)", height = 360.dp,
                )
                Text("X=β·SPY · Y=σ%(로그) · 색=모멘텀 · 점선=중앙값",
                    color = TextSecondary, fontSize = 11.sp)
                    }
                }
            }
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
    Text(text, color = color, fontSize = 17.sp, textAlign = align, fontWeight = fw,
        fontFamily = Mono,
        modifier = Modifier.weight(weight).clip(RoundedCornerShape(4.dp)).background(bg))
}

/** 종목 셀 — 상태 점(보유=금채움/이력=금링) + 티커(모노). */
@Composable
private fun RowScope.DotName(state: Int, name: String, weight: Float, rank: Int? = null, alpha: Float = 1f) {
    Row(Modifier.weight(weight), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        // 랭킹 순위 (미장 TOP에서만) — 정렬을 바꿔도 원래 순위를 알 수 있게
        if (rank != null) Text("$rank", color = TextMuted.copy(alpha = alpha), fontSize = 11.sp, fontFamily = Mono)
        if (state == 0) Spacer(Modifier.size(6.dp)) else Box(
            Modifier.size(6.dp).clip(RoundedCornerShape(50))
                .then(if (state == 2) Modifier.background(Gold.copy(alpha = alpha))
                else Modifier.border(1.3.dp, Gold.copy(alpha = alpha), RoundedCornerShape(50))),
        )
        Text(name, color = TextPrimary.copy(alpha = alpha), fontSize = 17.sp,
            fontWeight = FontWeight.SemiBold, fontFamily = Mono)
    }
}

private fun signed(v: Double) = (if (v >= 0) "+" else "") + "%.1f%%".format(v)

/** 랭킹 기준 선택 칩. */
@Composable
private fun RankChip(text: String, on: Boolean, small: Boolean = false, onClick: () -> Unit) {
    Box(
        Modifier.clip(RoundedCornerShape(8.dp))
            .background(if (on) Teal else SurfaceInput)
            .clickable(onClick = onClick)
            .padding(horizontal = if (small) 9.dp else 11.dp, vertical = 5.dp),
    ) {
        Text(text, color = if (on) Color(0xFF0C0E11) else TextSecondary,
            fontSize = if (small) 11.sp else 12.sp,
            fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
    }
}

/**
 * 실시간 틱 상태 — 갱신될 때마다 점이 밝아졌다 사그라들고, **안 도는 경우 그 사유**를 보여준다.
 * 조용히 멈추면 고장과 구분이 안 되므로 사유 표시가 핵심이다.
 */
@Composable
private fun TickStatus() {
    val seq = LivePrices.tickSeq
    val note = LivePrices.note
    val at = LivePrices.updatedAt
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
            note ?: if (sec > 0) "실시간 ${sec}초 간격" else "실시간 갱신 꺼짐",
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
private fun QuoteRow(vm: CompareViewModel, r: CompareRow, showTop: Boolean, rank: Int,
                     onOpenAnalysis: (String) -> Unit) {
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
        Modifier.fillMaxWidth().clickable { onOpenAnalysis(r.ticker) }
            .padding(horizontal = 2.dp, vertical = 6.dp),
    ) {
        DotName((if (r.holding) 2 else if (r.hasHistory) 1 else 0), r.name, 2.4f,
            rank = if (showTop) rank else null, alpha = alpha)
        Cell(Tickers.priceLabel(r.ticker, px), 2f, pnColor(day).copy(alpha = alpha), bg = flashBg)
        Cell(signed(day), 1.4f, pnColor(day).copy(alpha = alpha))
        Cell("%.0f".format(r.zPct), 1f, pctColor(r.zPct).copy(alpha = alpha), fw = FontWeight.Bold)
        Cell("%.0f".format(r.mPct), 1f, pctColor(r.mPct).copy(alpha = alpha), fw = FontWeight.Bold)
    }
}
