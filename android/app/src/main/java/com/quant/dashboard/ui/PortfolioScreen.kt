package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.quant.Portfolio
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgCard
import com.quant.dashboard.ui.theme.DividerColor
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Neutral
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.SegmentOn
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.TextMuted
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import com.quant.dashboard.ui.theme.WeightPalette

private fun pc(v: Double) = if (v > 0) Profit else if (v < 0) Loss else Neutral
private fun ident(i: Int) = WeightPalette[i % WeightPalette.size]

/** 자산추이 누적손익을 일/주/월 단위로 리샘플 (각 버킷의 마지막 값). equity는 시간 오름차순. */
private fun resampleEquity(equity: List<Pair<Long, Double>>, unit: String): List<Double> {
    if (unit == "일") return equity.map { it.second }
    val buckets = LinkedHashMap<Long, Double>()
    for ((sec, v) in equity) {
        val day = sec / 86400L
        val key = if (unit == "주") day / 7 else {
            val d = java.time.Instant.ofEpochSecond(sec).atZone(java.time.ZoneOffset.UTC).toLocalDate()
            d.year * 12L + d.monthValue
        }
        buckets[key] = v
    }
    return buckets.values.toList()
}
private fun won(usd: Double, rate: Double) =
    (if (usd >= 0) "+" else "-") + "%,.0f원".format(kotlin.math.abs(usd * rate))
private fun wonAbs(usd: Double, rate: Double) = "%,.0f원".format(usd * rate)

@Composable
fun PortfolioScreen(vm: PortfolioViewModel = viewModel()) {
    val s = vm.state
    LaunchedEffect(AppState.dataVersion) { vm.sync(AppState.dataVersion) }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Text("포트폴리오", color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.weight(1f))
            Box(
                Modifier.clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                    .clickable { vm.load() }.padding(horizontal = 10.dp, vertical = 6.dp),
            ) { Text("🔄", fontSize = 13.sp) }
        }

        when {
            s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                CircularProgressIndicator()
            }
            s.empty -> Text(
                "매매 기록이 없습니다.\n분석 탭에서 종목을 보고 ‘매매 기록’으로 입력하세요.",
                color = TextSecondary, fontSize = 14.sp,
            )
            s.result != null -> ResultBody(s.result, s.rate)
        }
    }
}

@Composable
private fun ResultBody(r: Portfolio.Result, rate: Double) {
    // ── 평가금액 히어로 카드 (그라데이션) ──
    val evalSum = r.holdings.sumOf { it.eval }
    val pnlSum = r.holdings.sumOf { it.pnl }
    val rp = if (evalSum - pnlSum != 0.0) pnlSum / (evalSum - pnlSum) * 100 else 0.0
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp))
            .background(Brush.linearGradient(listOf(Color(0xFF1C2330), BgCard)))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Text("평가금액", color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        Text(wonAbs(evalSum, rate), color = TextPrimary, fontSize = 32.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
        Text("${won(pnlSum, rate)} · ${if (rp >= 0) "+" else ""}${"%.2f".format(rp)}%",
            color = pc(pnlSum), fontSize = 13.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono)

        // 보유 비중 100% 스택바
        if (evalSum > 0 && r.holdings.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Row(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp))) {
                r.holdings.forEachIndexed { i, h ->
                    Box(Modifier.weight((h.eval / evalSum).toFloat().coerceAtLeast(0.001f))
                        .fillMaxHeight().background(ident(i)))
                }
            }
            Text("색 = 종목별 비중", color = TextMuted, fontSize = 10.sp, modifier = Modifier.padding(top = 4.dp))
        }

        // 보유 목록
        r.holdings.forEachIndexed { i, h ->
            Row(Modifier.fillMaxWidth().padding(top = 7.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(ident(i)))
                Spacer(Modifier.size(7.dp))
                Text(h.name, color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono)
                Spacer(Modifier.size(6.dp))
                Text("${h.qty}주", color = TextMuted, fontSize = 11.sp, fontFamily = Mono, modifier = Modifier.weight(1f))
                Column(horizontalAlignment = Alignment.End) {
                    Text(wonAbs(h.eval, rate), color = TextPrimary, fontSize = 12.5.sp, fontFamily = Mono)
                    Text("${if (h.retPct >= 0) "+" else ""}${"%.2f".format(h.retPct)}%",
                        color = pc(h.pnl), fontSize = 11.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono)
                }
            }
        }
    }

    // ── 손익 종합 카드 ──
    val total = r.seed + r.totalPnl
    val retPct = if (r.seed > 0) r.totalPnl / r.seed * 100 else 0.0
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BgCard).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text("손익 종합 (시드+실현)", color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        Text(wonAbs(total, rate), color = TextPrimary, fontSize = 24.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
        Text("${won(r.totalPnl, rate)} · ${if (retPct >= 0) "+" else ""}${"%.2f".format(retPct)}%",
            color = pc(r.totalPnl), fontSize = 13.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono)
        // 보조 칩: 고점대비 / MDD
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(top = 4.dp)) {
            StatChip("고점대비", "${"%.1f".format(r.currentDd)}%")
            StatChip("MDD", "${"%.1f".format(r.mdd)}%")
        }

        // 종목별 실현손익 — 다이버징 막대 (이익 오른쪽 빨강 / 손실 왼쪽 파랑)
        if (r.realized.isNotEmpty()) {
            Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("종목별 실현손익", color = TextMuted, fontSize = 11.sp, modifier = Modifier.weight(1f))
                Text("◀ 손실", color = Loss, fontSize = 10.sp)
                Spacer(Modifier.size(8.dp))
                Text("이익 ▶", color = Profit, fontSize = 10.sp)
            }
            val maxAbs = r.realized.maxOf { kotlin.math.abs(it.realized) }.coerceAtLeast(1e-9)
            r.realized.forEach { rz ->
                DivergingBar(rz.name, rz.realized, (kotlin.math.abs(rz.realized) / maxAbs).toFloat(),
                    won(rz.realized, rate))
            }
        }
    }

    // ── 자산 추이 ──
    if (r.equity.size >= 2) {
        var unit by remember { mutableStateOf(Store.equityUnit()) }
        val months = Store.equityMonths()
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
            Text("자산 추이 (누적손익) · ${months}개월", color = TextSecondary, fontSize = 12.sp,
                modifier = Modifier.weight(1f))
            Row(Modifier.clip(RoundedCornerShape(8.dp)).background(SurfaceInput).padding(2.dp),
                horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                listOf("일", "주", "월").forEach { u ->
                    val on = unit == u
                    Box(Modifier.clip(RoundedCornerShape(6.dp))
                        .background(if (on) SegmentOn else Color.Transparent)
                        .clickable { unit = u }.padding(horizontal = 10.dp, vertical = 4.dp)) {
                        Text(u, color = if (on) TextPrimary else TextMuted, fontSize = 12.sp,
                            fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
                    }
                }
            }
        }
        val cut = r.equity.last().first - months.toLong() * 30 * 86400
        val windowed = r.equity.filter { it.first >= cut }
        val src = if (windowed.size >= 2) windowed else r.equity
        val series = resampleEquity(src, unit).map { it * rate / 10000.0 }.toDoubleArray()
        EquityChart(series, unit = "만원")
    }

    TradeJournal()
}

/** 보조 통계 칩. */
@Composable
private fun StatChip(label: String, value: String) {
    Row(
        Modifier.clip(RoundedCornerShape(7.dp)).background(SurfaceInput)
            .padding(horizontal = 8.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(label, color = TextMuted, fontSize = 10.sp)
        Text(value, color = Loss, fontSize = 10.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
    }
}

/** 다이버징 막대 — 중앙 제로선 기준 이익 오른쪽(빨강)·손실 왼쪽(파랑). */
@Composable
private fun DivergingBar(name: String, amount: Double, frac: Float, amountText: String) {
    val profit = amount >= 0
    Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(name, color = TextPrimary, fontSize = 11.sp, fontFamily = Mono,
            maxLines = 1, modifier = Modifier.weight(1.2f))
        // 바 영역 (좌:손실 / 중앙선 / 우:이익)
        Row(Modifier.weight(3f).height(13.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.weight(1f).fillMaxHeight(), contentAlignment = Alignment.CenterEnd) {
                if (!profit && frac > 0f) Box(Modifier.fillMaxWidth(frac).height(9.dp)
                    .clip(RoundedCornerShape(2.dp)).background(Loss))
            }
            Box(Modifier.width(1.dp).fillMaxHeight().background(DividerColor))
            Box(Modifier.weight(1f).fillMaxHeight(), contentAlignment = Alignment.CenterStart) {
                if (profit && frac > 0f) Box(Modifier.fillMaxWidth(frac).height(9.dp)
                    .clip(RoundedCornerShape(2.dp)).background(Profit))
            }
        }
        Text(amountText, color = pc(amount), fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
            fontFamily = Mono, textAlign = TextAlign.End, modifier = Modifier.weight(1.6f))
    }
}

/** 📒 매매 일지 — 전 종목 매매 기록을 표 형식(최신순)으로. */
@Composable
private fun TradeJournal() {
    data class Entry(val date: String, val name: String, val type: String, val qty: Int, val price: Double, val memo: String?)
    val entries = remember {
        Store.loadTrades().flatMap { (tk, list) ->
            list.map { Entry(it.date, Tickers.displayName(tk), it.type, it.qty, it.price, it.memo) }
        }.sortedByDescending { it.date }
    }
    if (entries.isEmpty()) return
    var open by remember { mutableStateOf(false) }
    Text("매매 일지 (${entries.size}건) ${if (open) "▲" else "▼"}",
        color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold,
        modifier = Modifier.fillMaxWidth().clickable { open = !open })
    if (open) {
        Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.Top) {
            JCell("날짜", 1.9f, TextSecondary, FontWeight.SemiBold)
            JCell("종목", 1.3f, TextSecondary, FontWeight.SemiBold)
            JCell("구분", 1.0f, TextSecondary, FontWeight.SemiBold, TextAlign.Center)
            JCell("수량", 0.8f, TextSecondary, FontWeight.SemiBold, TextAlign.End)
            JCell("단가", 1.5f, TextSecondary, FontWeight.SemiBold, TextAlign.End)
            JCell("메모", 3.2f, TextSecondary, FontWeight.SemiBold)
        }
        entries.forEach { e ->
            val buy = e.type == "buy"
            Row(Modifier.fillMaxWidth().padding(vertical = 1.dp), verticalAlignment = Alignment.Top) {
                JCell(e.date, 1.9f, TextSecondary)
                JCell(e.name, 1.3f, TextPrimary, FontWeight.SemiBold)
                JCell(if (buy) "매수" else "매도", 1.0f, if (buy) Profit else Loss, FontWeight.SemiBold, TextAlign.Center)
                JCell("${e.qty}", 0.8f, TextPrimary, align = TextAlign.End)
                JCell("$${"%.2f".format(e.price)}", 1.5f, TextPrimary, align = TextAlign.End)
                JCell(e.memo ?: "", 3.2f, TextSecondary, maxLines = Int.MAX_VALUE)
            }
        }
    }
}

@Composable
private fun RowScope.JCell(text: String, weight: Float, color: Color,
                           fw: FontWeight = FontWeight.Normal, align: TextAlign = TextAlign.Start,
                           maxLines: Int = 1) {
    Text(text, color = color, fontSize = 11.sp, fontWeight = fw, textAlign = align,
        fontFamily = Mono, maxLines = maxLines, modifier = Modifier.weight(weight).padding(horizontal = 2.dp))
}
