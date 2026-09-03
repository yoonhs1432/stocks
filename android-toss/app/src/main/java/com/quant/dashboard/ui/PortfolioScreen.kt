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
import androidx.compose.runtime.rememberCoroutineScope
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
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.quant.dashboard.data.BrokerCreds
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.Snapshots
import com.quant.dashboard.data.TossSync
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgCard
import com.quant.dashboard.ui.theme.ChipOn
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Neutral
import com.quant.dashboard.ui.theme.Profit
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
 * 포트폴리오 표시 통화 — 원/달러 토글.
 *
 * 계산은 전부 **원화로 모아 놓고** 표시 직전에만 환산한다. 종목마다 거래 통화가 달라
 * (국내 ₩ / 해외 $) 화면에 섞여 나오면 합계와 비교가 안 됐다.
 */
private class Money(val usd: Boolean, val rate: Double) {
    /** 원화 금액 → 표시 문자열. */
    fun of(krw: Double): String =
        if (usd) "$" + "%,.2f".format(krw / rate) else "%,.0f원".format(krw)

    /** 부호를 앞에 붙인 금액 (`-$12.34` / `-1,688,798원`). */
    fun signed(krw: Double): String =
        (if (krw >= 0) "+" else "-") + of(kotlin.math.abs(krw))

    /** 거래 통화(KRW|USD) 기준 금액을 원화로. */
    fun krwOf(v: Double, currency: String): Double = if (currency == "USD") v * rate else v
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PortfolioScreen(onOpenAnalysis: (String) -> Unit = {}) {
    // 계좌 상태를 여기서 들고 있어야 당겨서 새로고침 스피너가 실제 조회와 맞물린다.
    // (예전에는 isRefreshing=false 로 박혀 있어 스피너가 즉시 사라졌고,
    //  account() 도 5분 캐시라 당겨도 아무 일이 일어나지 않았다)
    var acct by remember { mutableStateOf(TossSync.cachedAccount()) }
    var err by remember { mutableStateOf<String?>(null) }
    var refreshing by remember { mutableStateOf(false) }
    var usdMode by remember { mutableStateOf(Store.portfolioUsd()) }
    val scope = rememberCoroutineScope()

    suspend fun reload(force: Boolean) {
        val fresh = withContext(Dispatchers.IO) {
            try { TossSync.account(force = force) } catch (e: Exception) { err = e.message; null }
        }
        if (fresh != null) { acct = fresh; err = null }
    }

    LaunchedEffect(AppState.dataVersion) { reload(force = false) }

    Column(modifier = Modifier.fillMaxSize().background(BgApp)) {
        PullToRefreshBox(
            isRefreshing = refreshing,
            onRefresh = {
                refreshing = true
                scope.launch {
                    reload(force = true)   // 5분 캐시를 건너뛰고 계좌를 다시 받는다
                    AppState.bump()        // 다른 탭도 갱신
                    refreshing = false
                }
            },
            modifier = Modifier.fillMaxSize(),
        ) {
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(14.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("포트폴리오", color = TextPrimary, fontSize = 19.sp,
                        fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                    // 표시 통화 — 계산은 원화로 하고 표시만 바꾼다
                    listOf(false to "원", true to "$").forEach { (isUsd, label) ->
                        val on = usdMode == isUsd
                        Box(
                            Modifier.padding(start = 6.dp).clip(RoundedCornerShape(8.dp))
                                .background(if (on) ChipOn else SurfaceInput)
                                .clickable { usdMode = isUsd; Store.setPortfolioUsd(isUsd) }
                                .padding(horizontal = 14.dp, vertical = 6.dp),
                        ) {
                            Text(label, color = if (on) TextPrimary else TextMuted,
                                fontSize = 13.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }

                val a = acct
                when {
                    !BrokerCreds.isLinked() -> Text("설정 탭에서 토스증권을 연결하면 계좌가 표시됩니다.",
                        color = TextSecondary, fontSize = 14.sp)
                    a != null -> TossBody(a, usdMode, onOpenAnalysis)
                    else -> Text(err?.let { "⚠️ $it" } ?: "계좌 정보를 불러오는 중…",
                        color = if (err != null) Loss else TextSecondary, fontSize = 13.sp)
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
private fun TossBody(a: TossSync.Account, usdMode: Boolean, onOpenAnalysis: (String) -> Unit) {
    val m = Money(usdMode, a.rate)
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
        Text(m.of(a.totalKrw()), color = TextPrimary, fontSize = 32.sp,
            fontWeight = FontWeight.Bold, fontFamily = Mono)
        // 당일 손익 — 토스 API 가 계산해 준 값을 그대로 쓴다 (증권사 앱과 같은 기준)
        val hs = a.holdings
        Text(
            "오늘 ${m.signed(a.dailyPnlKrw())} (${signPct(hs.dailyPnlRate)})",
            color = pc(a.dailyPnlKrw()), fontSize = 15.sp,
            fontWeight = FontWeight.Bold, fontFamily = Mono,
        )
        Text(
            "평가손익 ${m.signed(a.pnlKrw())} (${signPct(hs.pnlRate)})",
            color = pc(a.pnlKrw()), fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold, fontFamily = Mono,
        )

        Spacer(Modifier.height(8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            StatChip("평가금액", m.of(a.krwEval + a.usdEval * a.rate))
            StatChip("예수금", m.of(a.krwCash + a.usdCash * a.rate))
            StatChip("환율", "%,.1f".format(a.rate))
        }
        if (a.usdEval != 0.0 || a.usdCash != 0.0) {
            Text("달러 자산 $${"%,.2f".format(a.usdEval)} + 예수금 $${"%,.2f".format(a.usdCash)} (환율 환산 포함)",
                color = TextMuted, fontSize = 10.sp, modifier = Modifier.padding(top = 4.dp))
        }

        // 평가금액(원화 환산) 내림차순. 스택바와 목록이 **같은 목록**을 써야 색과 순서가 맞는다
        fun krwOf(h: com.quant.dashboard.data.TossApi.Holding) = m.krwOf(h.evalAmount, h.currency)
        val items = remember(a) { a.holdings.items.sortedByDescending { krwOf(it) } }

        // 보유 비중 100% 스택바
        val evalSum = items.sumOf { krwOf(it) }
        if (evalSum > 0 && items.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Row(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp))) {
                items.forEachIndexed { i, h ->
                    Box(Modifier.weight((krwOf(h) / evalSum).toFloat().coerceAtLeast(0.001f))
                        .fillMaxHeight().background(ident(i)))
                }
            }
        }

        // 보유 목록 — 행 탭 시 분석 이동
        if (items.isNotEmpty()) {
            Text("종목을 누르면 분석 탭으로 이동", color = TextMuted, fontSize = 10.sp,
                modifier = Modifier.padding(top = 6.dp))
        }
        items.forEachIndexed { i, h ->
            // 실시간 틱이 있으면 현재가로 다시 계산한다. 기준은 API 와 동일 —
            // 누적 손익률 = 현재가/평단 - 1 (안 맞추면 증권사 앱과 숫자가 어긋난다).
            val lp = LivePrices.price(h.symbol)
            val evalKrw = m.krwOf(if (lp != null) lp * h.quantity else h.evalAmount, h.currency)
            val rate2 = if (lp != null && h.avgPrice > 0) lp / h.avgPrice - 1.0 else h.pnlRate
            val pnlKrw = m.krwOf(if (lp != null) (lp - h.avgPrice) * h.quantity else h.pnlAmount, h.currency)
            // 한 줄에 다 넣었더니 글자가 작고 빽빽해 읽기 어려웠다 → 이름 / 수량·금액·손익 2줄로
            Column(
                Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                    .clickable { onOpenAnalysis(h.symbol) }
                    .padding(top = 10.dp, bottom = 2.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(9.dp).clip(RoundedCornerShape(50)).background(ident(i)))
                    Spacer(Modifier.size(8.dp))
                    Text(h.name, color = TextPrimary, fontSize = 17.sp, maxLines = 1,
                        fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                    Text("›", color = TextMuted, fontSize = 16.sp)
                }
                Row(Modifier.fillMaxWidth().padding(top = 2.dp), verticalAlignment = Alignment.Bottom) {
                    Text(qtyLabel(h.quantity), color = TextSecondary, fontSize = 14.sp,
                        maxLines = 1, modifier = Modifier.weight(1f))
                    Column(horizontalAlignment = Alignment.End) {
                        Text(m.of(evalKrw), color = TextPrimary, fontSize = 18.sp,
                            fontWeight = FontWeight.Bold, fontFamily = Mono, maxLines = 1)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(m.signed(pnlKrw), color = pc(pnlKrw), fontSize = 14.sp,
                                fontWeight = FontWeight.SemiBold, fontFamily = Mono, maxLines = 1)
                            Text("  |  ", color = TextMuted, fontSize = 13.sp)
                            Text(signPct(rate2), color = pc(rate2), fontSize = 14.sp,
                                fontWeight = FontWeight.SemiBold, fontFamily = Mono, maxLines = 1)
                        }
                    }
                }
            }
        }
    }

    // ── 자산 추이 ──
    //
    // 토스가 준 값을 그대로 그린다. 예전에는 매매기록으로 입출금을 역산해 TWR 수익률을 냈는데,
    // 매매기록이 불완전하면 값이 어긋나서 걷어냈다.
    val sr = remember(AppState.dataVersion, usdMode) { Snapshots.series(usdMode) }
    val pnlSeries = remember(AppState.dataVersion, usdMode) { Snapshots.pnls(usdMode) }
    // 원화는 만원 단위로 접어야 축이 읽힌다. 달러는 그대로
    val div = if (usdMode) 1.0 else 10000.0
    val unit = if (usdMode) "$" else "만원"

    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BgCard).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text("자산 · 핀치=확대 · 탭=그 시점 값", color = TextSecondary,
            fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        if (sr.dates.size < 2) {
            Text("기록이 ${sr.dates.size}일치뿐입니다. 앱을 열 때마다 하루 1회 잔고를 저장하므로\n" +
                "며칠 지나면 그래프가 그려집니다.", color = TextMuted, fontSize = 11.sp)
        } else {
            AssetStackChart(
                eval = DoubleArray(sr.eval.size) { sr.eval[it] / div },
                cash = DoubleArray(sr.cash.size) { sr.cash[it] / div },
                unit = unit, labels = sr.dates,
            )
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(sr.dates.first(), color = TextSecondary, fontSize = 10.sp)
                Text("${sr.dates.size}일 기록", color = TextMuted, fontSize = 10.sp)
                Text(sr.dates.last(), color = TextSecondary, fontSize = 10.sp)
            }
            Text("총자산 = 평가금액 + 예수금. 입금·출금도 그대로 반영되는 곡선입니다.",
                color = TextMuted, fontSize = 10.sp, modifier = Modifier.padding(top = 3.dp))
        }
    }

    // ── 평가손익 추이 ──
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BgCard).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp)) {
        Text("평가손익 · 핀치=확대 · 탭=그 시점 값", color = TextSecondary,
            fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        if (pnlSeries.size < 2) {
            Text("평가손익까지 남긴 기록이 ${pnlSeries.size}일치뿐입니다 — 이전 기록에는 없어서\n" +
                "이 곡선은 새로 쌓입니다.", color = TextMuted, fontSize = 11.sp)
        } else {
            EquityChart(
                pnlSeries.map { it.second / div }.toDoubleArray(),
                unit = unit, labels = pnlSeries.map { it.first },
                baseZero = true,   // 손익은 0선이 기준
            )
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(pnlSeries.first().first, color = TextSecondary, fontSize = 10.sp)
                Text("${pnlSeries.size}일 기록", color = TextMuted, fontSize = 10.sp)
                Text(pnlSeries.last().first, color = TextSecondary, fontSize = 10.sp)
            }
            Text("보유 중인 종목의 미실현 손익입니다 — 전량 매도하면 0으로 떨어집니다.",
                color = TextMuted, fontSize = 10.sp, modifier = Modifier.padding(top = 3.dp))
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
