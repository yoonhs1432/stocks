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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.quant.dashboard.data.BrokerCreds
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.Snapshots
import com.quant.dashboard.data.TossSync
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

/** 소수비율(0.0141) → "+1.41%". 토스 API 는 손익률을 전부 소수비율로 준다. */
private fun signPct(r: Double): String = (if (r >= 0) "+" else "") + "%.2f%%".format(r * 100)
private fun ident(i: Int) = WeightPalette[i % WeightPalette.size]

/**
 * 자산추이 x축 라벨 (epoch 초 → yy.MM.dd).
 *
 * 예전에는 일/주/월 리샘플 토글이 있었는데, 같은 기간을 성기게 그릴 뿐이라 없앴다 —
 * 기간은 "자산추이 기간" 설정으로 조절하고 그래프는 항상 일 단위로 그린다.
 */
private fun equityLabels(equity: List<Pair<Long, Double>>): List<String> {
    val f = java.text.SimpleDateFormat("yy.MM.dd", java.util.Locale.US)
    return equity.map { f.format(java.util.Date(it.first * 1000L)) }
}
private fun won(usd: Double, rate: Double) =
    (if (usd >= 0) "+" else "-") + "%,.0f원".format(kotlin.math.abs(usd * rate))
private fun wonAbs(usd: Double, rate: Double) = "%,.0f원".format(usd * rate)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PortfolioScreen(vm: PortfolioViewModel = viewModel(), onOpenAnalysis: (String) -> Unit = {}) {
    val s = vm.state
    LaunchedEffect(AppState.dataVersion) { vm.sync(AppState.dataVersion) }

    Column(modifier = Modifier.fillMaxSize().background(BgApp)) {
        // 당겨서 새로고침 → 전체 탭 새로고침 (dataVersion bump로 모든 탭이 재로드)
        PullToRefreshBox(isRefreshing = s.loading, onRefresh = { AppState.bump() },
            modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text(if (Store.tossMode() && BrokerCreds.isLinked()) "포트폴리오 (토스)" else "포트폴리오",
                    color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold)

                val toss = Store.tossMode() && BrokerCreds.isLinked()
                when {
                    toss -> TossBody(onOpenAnalysis)
                    s.result != null -> ResultBody(s.result, s.rate, onOpenAnalysis)
                    s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                        CircularProgressIndicator()
                    }
                    s.empty -> Text(
                        "매매 기록이 없습니다.\n분석 탭에서 종목을 보고 ‘매매 기록’으로 입력하세요.",
                        color = TextSecondary, fontSize = 14.sp,
                    )
                }
            }
        }
    }
}

/**
 * 토스 기반 포트폴리오 본문 — 증권사 실측 잔고가 진실이다.
 *
 * 매매기록으로 역산하지 않고 `/holdings` + `/buying-power` 를 그대로 쓴다.
 * 자산추이는 토스에 과거 잔고 API 가 없어, 앱이 열릴 때 남긴 일별 스냅샷으로 그린다
 * (= 전환 시점부터 쌓이며, 앱을 안 연 날은 비어 있다).
 */
@Composable
private fun TossBody(onOpenAnalysis: (String) -> Unit) {
    var acct by remember { mutableStateOf(TossSync.cachedAccount()) }
    var err by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(AppState.dataVersion) {
        val fresh = withContext(Dispatchers.IO) {
            try { TossSync.account() } catch (e: Exception) { err = e.message; null }
        }
        if (fresh != null) { acct = fresh; err = null }
    }
    val a = acct
    if (a == null) {
        Text(err?.let { "⚠️ $it" } ?: "계좌 정보를 불러오는 중…",
            color = if (err != null) Loss else TextSecondary, fontSize = 13.sp)
        return
    }

    // ── 총자산 히어로 (평가금액 + 예수금) ──
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp))
            .background(Brush.linearGradient(listOf(Color(0xFF1C2330), BgCard)))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("총자산 (토스)", color = TextSecondary, fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            Text(BrokerCreds.maskedAccount(), color = TextMuted, fontSize = 10.sp, fontFamily = Mono)
        }
        Text("%,.0f원".format(a.totalKrw()), color = TextPrimary, fontSize = 32.sp,
            fontWeight = FontWeight.Bold, fontFamily = Mono)
        // 당일 손익 — 토스 API 가 계산해 준 값을 그대로 쓴다 (증권사 앱과 같은 기준)
        val hs = a.holdings
        Text(
            "오늘 ${won(a.dailyPnlKrw(), 1.0)} (${signPct(hs.dailyPnlRate)})",
            color = pc(a.dailyPnlKrw()), fontSize = 15.sp,
            fontWeight = FontWeight.Bold, fontFamily = Mono,
        )
        Text(
            "평가손익 ${won(a.pnlKrw(), 1.0)} (${signPct(hs.pnlRate)})" +
                " · 비용차감 ${won(a.pnlAfterCostKrw(), 1.0)} (${signPct(hs.pnlRateAfterCost)})",
            color = pc(a.pnlKrw()), fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold, fontFamily = Mono,
        )

        Spacer(Modifier.height(8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            StatChip("평가금액", "%,.0f원".format(a.krwEval + a.usdEval * a.rate))
            StatChip("예수금", "%,.0f원".format(a.krwCash + a.usdCash * a.rate))
            StatChip("환율", "%,.1f".format(a.rate))
        }
        if (a.usdEval != 0.0 || a.usdCash != 0.0) {
            Text("달러 자산 $${"%,.2f".format(a.usdEval)} + 예수금 $${"%,.2f".format(a.usdCash)} (환율 환산 포함)",
                color = TextMuted, fontSize = 10.sp, modifier = Modifier.padding(top = 4.dp))
        }

        // 보유 비중 100% 스택바
        val evalSum = a.holdings.items.sumOf { if (it.currency == "USD") it.evalAmount * a.rate else it.evalAmount }
        if (evalSum > 0 && a.holdings.items.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Row(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp))) {
                a.holdings.items.forEachIndexed { i, h ->
                    val w = (if (h.currency == "USD") h.evalAmount * a.rate else h.evalAmount) / evalSum
                    Box(Modifier.weight(w.toFloat().coerceAtLeast(0.001f)).fillMaxHeight().background(ident(i)))
                }
            }
        }

        // 보유 목록 — 행 탭 시 분석 이동
        if (a.holdings.items.isNotEmpty()) {
            Text("종목을 누르면 분석 탭으로 이동", color = TextMuted, fontSize = 10.sp,
                modifier = Modifier.padding(top = 6.dp))
        }
        a.holdings.items.forEachIndexed { i, h ->
            val cur = if (h.currency == "KRW") "₩" else "$"
            // 실시간 틱이 있으면 현재가로 다시 계산한다. 기준은 API 와 동일하게 맞춘다 —
            // 누적 손익률 = 현재가/평단 - 1, 당일 등락률 = 현재가/전일기준가 - 1
            // (전일기준가는 API 의 당일 손익률에서 역산). 안 맞으면 증권사 앱과 숫자가 어긋난다.
            val lp = LivePrices.price(h.symbol)
            val evalAmt = if (lp != null) lp * h.quantity else h.evalAmount
            val rate2 = if (lp != null && h.avgPrice > 0) lp / h.avgPrice - 1.0 else h.pnlRate
            val dayRate = if (lp != null && h.basePrice > 0) lp / h.basePrice - 1.0 else h.dailyPnlRate
            Row(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                    .clickable { onOpenAnalysis(h.symbol) }
                    .padding(top = 7.dp, bottom = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(ident(i)))
                Spacer(Modifier.size(7.dp))
                Text(h.name, color = TextPrimary, fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold, fontFamily = Mono)
                Spacer(Modifier.size(6.dp))
                Text(qtyLabel(h.quantity), color = TextMuted, fontSize = 11.sp,
                    fontFamily = Mono, modifier = Modifier.weight(1f))
                Column(horizontalAlignment = Alignment.End) {
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        Text("$cur${"%,.2f".format(evalAmt)}", color = TextPrimary,
                            fontSize = 12.5.sp, fontFamily = Mono)
                        Text("오늘 ${signPct(dayRate)}", color = pc(dayRate),
                            fontSize = 11.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
                    }
                    Text("평단 $cur${"%,.2f".format(h.avgPrice)} · ${signPct(rate2)}",
                        color = pc(rate2), fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold, fontFamily = Mono)
                }
                Text(" ›", color = TextMuted, fontSize = 15.sp)
            }
        }
    }

    // ── 자산 추이 (스냅샷) ──
    val snaps = remember(AppState.dataVersion) { Snapshots.totals() }
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BgCard).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text("자산 추이 (총자산) · 핀치=확대 · 탭=그 시점 금액",
            color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        if (snaps.size < 2) {
            Text("기록이 ${snaps.size}일치뿐입니다. 앱을 열 때마다 하루 1회 잔고를 저장하므로\n" +
                "며칠 지나면 그래프가 그려집니다.", color = TextMuted, fontSize = 11.sp)
        } else {
            EquityChart(
                snaps.map { it.second / 10000.0 }.toDoubleArray(),
                unit = "만원", labels = snaps.map { it.first },
            )
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(snaps.first().first, color = TextSecondary, fontSize = 10.sp)
                Text("${snaps.size}일 기록", color = TextMuted, fontSize = 10.sp)
                Text(snaps.last().first, color = TextSecondary, fontSize = 10.sp)
            }
            Snapshots.drawdown()?.let { dd ->
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.padding(top = 4.dp)) {
                    StatChip("고점대비", "${"%.1f".format(dd.current)}%")
                    StatChip("MDD", "${"%.1f".format(dd.max)}%" + (dd.maxDate?.let { " ($it)" } ?: ""))
                }
            }
        }
    }

    TradeJournal()
}

/** 수량 표시 — 소수점 보유(미국 소수점 매매)는 필요한 자리까지만. */
private fun qtyLabel(q: Double): String =
    if (q == Math.floor(q)) "%,.0f주".format(q)
    else "%,.4f".format(q).trimEnd('0').trimEnd('.') + "주"

@Composable
private fun ResultBody(r: Portfolio.Result, rate: Double, onOpenAnalysis: (String) -> Unit = {}) {
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

        // 보유 목록 — 행을 누르면 해당 종목 분석 탭으로 이동
        if (r.holdings.isNotEmpty()) {
            Text("종목을 누르면 분석 탭으로 이동", color = TextMuted, fontSize = 10.sp,
                modifier = Modifier.padding(top = 6.dp))
        }
        r.holdings.forEachIndexed { i, h ->
            Row(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                    .clickable { onOpenAnalysis(h.ticker) }
                    .padding(top = 7.dp, bottom = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(ident(i)))
                Spacer(Modifier.size(7.dp))
                Text(h.name, color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono)
                Spacer(Modifier.size(6.dp))
                Text("${h.qty}주", color = TextMuted, fontSize = 11.sp, fontFamily = Mono, modifier = Modifier.weight(1f))
                Column(horizontalAlignment = Alignment.End) {
                    Text(wonAbs(h.eval, rate), color = TextPrimary, fontSize = 12.5.sp, fontFamily = Mono)
                    Text("${won(h.pnl, rate)} · ${if (h.retPct >= 0) "+" else ""}${"%.2f".format(h.retPct)}%",
                        color = pc(h.pnl), fontSize = 11.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono)
                }
                Text(" ›", color = TextMuted, fontSize = 15.sp)
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
        val months = Store.equityMonths()
        Text("자산 추이 (누적손익) · ${months}개월 · 핀치=확대 · 탭=그 시점 금액",
            color = TextSecondary, fontSize = 12.sp)
        val cut = r.equity.last().first - months.toLong() * 30 * 86400
        val windowed = r.equity.filter { it.first >= cut }
        val src = if (windowed.size >= 2) windowed else r.equity
        EquityChart(
            src.map { it.second * rate / 10000.0 }.toDoubleArray(),
            unit = "만원", labels = equityLabels(src), baseZero = true,
        )
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
    // ticker 를 함께 들고 있어야 단가를 원화/달러 중 맞는 단위로 찍을 수 있다
    data class Entry(val ticker: String, val date: String, val name: String,
                     val type: String, val qty: Int, val price: Double, val memo: String?)
    val entries = remember {
        Store.visibleTrades().flatMap { (tk, list) ->
            list.map { Entry(tk, it.date, Tickers.displayName(tk), it.type, it.qty, it.price, it.memo) }
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
                JCell(Tickers.priceLabel(e.ticker, e.price), 1.5f, TextPrimary, align = TextAlign.End)
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
