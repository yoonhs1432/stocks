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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.quant.dashboard.data.BrokerCreds
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.Snapshots
import com.quant.dashboard.data.TossSync
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgCard
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

/** 부호를 앞에 붙인 금액 — `-$12.34` 처럼 통화기호 뒤가 아니라 앞에 부호가 오게. */
private fun signedAmt(v: Double, cur: String, dec: Int): String =
    (if (v >= 0) "+" else "-") + cur + "%,.${dec}f".format(kotlin.math.abs(v))

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PortfolioScreen(onOpenAnalysis: (String) -> Unit = {}) {
    Column(modifier = Modifier.fillMaxSize().background(BgApp)) {
        // 당겨서 새로고침 → 전체 탭 새로고침 (dataVersion bump로 모든 탭이 재로드)
        PullToRefreshBox(isRefreshing = false, onRefresh = { AppState.bump() },
            modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Text("포트폴리오", color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold)

                if (BrokerCreds.isLinked()) TossBody(onOpenAnalysis)
                else Text("설정 탭에서 토스증권을 연결하면 계좌가 표시됩니다.",
                    color = TextSecondary, fontSize = 14.sp)
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
            "평가손익 ${won(a.pnlKrw(), 1.0)} (${signPct(hs.pnlRate)})",
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
            // 실시간 틱이 있으면 현재가로 다시 계산한다. 기준은 API 와 동일 —
            // 누적 손익률 = 현재가/평단 - 1 (안 맞추면 증권사 앱과 숫자가 어긋난다).
            // 한 줄에 담으려고 **현재가·평단은 빼고** 수익률 · 수익금 · 평가금액만 남겼다.
            val lp = LivePrices.price(h.symbol)
            val evalAmt = if (lp != null) lp * h.quantity else h.evalAmount
            val rate2 = if (lp != null && h.avgPrice > 0) lp / h.avgPrice - 1.0 else h.pnlRate
            val pnlAmt = if (lp != null) (lp - h.avgPrice) * h.quantity else h.pnlAmount
            val dec = if (h.currency == "KRW") 0 else 2
            Row(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                    .clickable { onOpenAnalysis(h.symbol) }
                    .padding(top = 6.dp, bottom = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(ident(i)))
                Spacer(Modifier.size(6.dp))
                Text(h.name, color = TextPrimary, fontSize = 13.sp, maxLines = 1,
                    fontWeight = FontWeight.SemiBold, fontFamily = Mono)
                Spacer(Modifier.size(5.dp))
                Text(qtyLabel(h.quantity), color = TextMuted, fontSize = 10.sp,
                    maxLines = 1, fontFamily = Mono, modifier = Modifier.weight(1f))
                Text(signPct(rate2), color = pc(rate2), fontSize = 12.sp,
                    fontWeight = FontWeight.Bold, fontFamily = Mono, maxLines = 1)
                Spacer(Modifier.size(6.dp))
                Text(signedAmt(pnlAmt, cur, dec), color = pc(pnlAmt), fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold, fontFamily = Mono, maxLines = 1)
                Spacer(Modifier.size(6.dp))
                Text(cur + "%,.${dec}f".format(evalAmt), color = TextPrimary,
                    fontSize = 12.sp, fontFamily = Mono, maxLines = 1)
                Text(" ›", color = TextMuted, fontSize = 14.sp)
            }
        }
    }

    // ── 자산 추이 (스냅샷) ──
    //
    // ⚠️ 총자산 곡선은 **입금하면 그대로 올라간다** — 수익률이 아니다.
    //    그래서 기본을 [수익률](입출금을 걷어낸 TWR 지수)로 두고, 총자산은 참고용으로 남긴다.
    var mode by remember { mutableStateOf(Store.equityMode()) }
    val snaps = remember(AppState.dataVersion) { Snapshots.totals() }
    val pnlSeries = remember(AppState.dataVersion) { Snapshots.investPnl() }
    val twrSeries = remember(AppState.dataVersion) { Snapshots.twr() }

    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BgCard).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)) {

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("return" to "수익률", "pnl" to "누적손익", "total" to "총자산").forEach { (id, label) ->
                val on = mode == id
                Box(
                    Modifier.weight(1f).clip(RoundedCornerShape(8.dp))
                        .background(if (on) SegmentOn else SurfaceInput)
                        .clickable { mode = id; Store.setEquityMode(id) }
                        .padding(vertical = 6.dp),
                ) {
                    Text(label, color = if (on) Color.White else TextSecondary, fontSize = 12.sp,
                        fontWeight = FontWeight.Bold, textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth())
                }
            }
        }

        val series = when (mode) { "total" -> snaps; "pnl" -> pnlSeries; else -> twrSeries }
        val caption = when (mode) {
            "total" -> "총자산 (입출금 포함)"
            "pnl" -> "누적 투자손익 (평가 + 실현) · 입출금 무관"
            else -> "수익률 지수 (입출금 제거, 시작 = 100)"
        }
        Text(caption + " · 핀치=확대 · 탭=그 시점 값",
            color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)

        if (series.size < 2) {
            Text(
                if (mode == "total")
                    "기록이 ${snaps.size}일치뿐입니다. 앱을 열 때마다 하루 1회 잔고를 저장하므로\n" +
                        "며칠 지나면 그래프가 그려집니다."
                else
                    "손익까지 남긴 기록이 ${series.size}일치뿐입니다. 입출금을 걷어내려면 평가손익·매입금액이\n" +
                        "같이 필요한데 이전 기록에는 없어서, 이 곡선은 오늘부터 새로 쌓입니다.",
                color = TextMuted, fontSize = 11.sp,
            )
        } else {
            val values = (
                if (mode == "return") series.map { it.second }
                else series.map { it.second / 10000.0 }
            ).toDoubleArray()
            EquityChart(
                values,
                unit = if (mode == "return") "" else "만원",
                labels = series.map { it.first },
                baseZero = mode == "pnl",   // 누적손익은 0선이 기준
            )
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(series.first().first, color = TextSecondary, fontSize = 10.sp)
                Text(
                    if (mode == "total") "${series.size}일 기록"
                    else "구간 " + signPct(values.last() / values.first() - 1.0),
                    color = if (mode == "total") TextMuted else pc(values.last() - values.first()),
                    fontSize = 10.sp, fontWeight = FontWeight.Bold,
                )
                Text(series.last().first, color = TextSecondary, fontSize = 10.sp)
            }
            // 낙폭은 지금 보고 있는 곡선 기준. 총자산 낙폭은 입금 타이밍에 휘둘리므로 참고용이다.
            Snapshots.drawdown(series)?.let { dd ->
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.padding(top = 4.dp)) {
                    StatChip("고점대비", "${"%.1f".format(dd.current)}%")
                    StatChip("MDD", "${"%.1f".format(dd.max)}%" + (dd.maxDate?.let { " ($it)" } ?: ""))
                }
            }
            if (mode == "total") {
                Text("입금·출금이 그대로 반영되는 곡선입니다. 순수 매매 성과는 [수익률] 을 보세요.",
                    color = TextMuted, fontSize = 10.sp, modifier = Modifier.padding(top = 3.dp))
            }
        }
    }


    TradeJournal()
}

/** 수량 표시 — 소수점 보유(미국 소수점 매매)는 필요한 자리까지만. */
private fun qtyLabel(q: Double): String =
    if (q == Math.floor(q)) "%,.0f주".format(q)
    else "%,.4f".format(q).trimEnd('0').trimEnd('.') + "주"

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
