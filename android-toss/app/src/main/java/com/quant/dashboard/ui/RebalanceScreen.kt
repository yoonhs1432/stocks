package com.quant.dashboard.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import kotlinx.coroutines.withContext
import kotlinx.coroutines.Dispatchers
import com.quant.dashboard.data.AnalysisCache
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Server
import com.quant.dashboard.data.Snapshots
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.TossLink
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.Gold
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.TextMuted
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import kotlinx.coroutines.delay
import kotlin.math.abs
import kotlin.math.roundToLong

/*
 * 리밸런싱 중심 화면 — 홈·바구니·더보기(손익·세금). 계산은 PC 가 한다(web/rebalance.py).
 * 이 앱은 주문을 내지 않고 토스 주문 화면으로 건네준다. 색은 한국식: 매수·상승=빨강, 매도·하락=파랑.
 * 목업: research/mockup_v4_I.png (홈·바구니). 계좌 탭은 홈과 겹쳐 없앴다 — 총자산 추이는 홈,
 * 보유 금액·손익은 바구니 타일, 평단·세금은 더보기 › 손익·세금.
 */

private val PERIOD_LABEL = mapOf("monthly" to "매월", "quarterly" to "분기", "yearly" to "매년")
private val PERIOD_SUB = mapOf("monthly" to "매월", "quarterly" to "분기마다", "yearly" to "매년")

/** 원화를 짧게 — 1억 2,480만 / 475만 / 8,200원. */
internal fun manwon(v: Double): String {
    val a = abs(v)
    val sign = if (v < 0) "−" else ""
    return when {
        a >= 1e8 -> {
            val eok = (a / 1e8).toLong()
            val man = ((a - eok * 1e8) / 1e4).roundToLong()
            sign + if (man > 0) "${eok}억 ${"%,d".format(man)}만" else "${eok}억"
        }
        a >= 1e4 -> sign + "%,d만".format((a / 1e4).roundToLong())
        else -> sign + "%,d원".format(a.roundToLong())
    }
}

private fun signed(v: Double) = (if (v > 0) "+" else "") + manwon(v)
private fun pct(v: Double, d: Int = 1) = "%+.${d}f%%".format(v)

private fun mmdd(iso: String): String = iso.takeIf { it.length >= 10 }?.let {
    "${it.substring(5, 7).toInt()}/${it.substring(8, 10).toInt()}"
} ?: iso

private fun qtyLabel(q: Double): String =
    if (q == kotlin.math.floor(q)) "%,d주".format(q.toLong()) else "%.2f주".format(q)

/** 매도 먼저(현금을 먼저 만든다, 큰 것부터) → 매수(큰 것부터). */
private fun ordersOf(p: Server.Rebalance): List<Server.RebalRow> =
    p.rows.filter { it.action == "sell" }.sortedByDescending { it.amountKrw } +
        p.rows.filter { it.action == "buy" }.sortedByDescending { it.amountKrw }

private fun levOf(desc: String?): String? = desc?.let { Regex("(\\d(?:\\.\\d+)?)배").find(it)?.groupValues?.get(1) }

@Composable
private fun Card(modifier: Modifier = Modifier, brush: Brush? = null, pad: Dp = 14.dp, content: @Composable () -> Unit) {
    Column(
        modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp))
            .then(if (brush != null) Modifier.background(brush) else Modifier.background(CARD))
            .padding(pad),
    ) { content() }
}

@Composable
private fun Pill(text: String, color: Color = TextSecondary, bg: Color = CARD2) {
    Box(Modifier.clip(RoundedCornerShape(99.dp)).background(bg).padding(horizontal = 10.dp, vertical = 4.dp)) {
        Text(text, color = color, fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 1)
    }
}

@Composable
private fun Label(t: String) = Text(t, color = TextSecondary, fontSize = 12.sp)

// ══════════════════════ ① 홈 ══════════════════════

private val RANGES = listOf("1M" to 22, "3M" to 66, "6M" to 130, "1Y" to 260, "전체" to Int.MAX_VALUE)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(vm: RebalanceViewModel = viewModel()) {
    LaunchedEffect(AppState.dataVersion) { vm.load() }
    LaunchedEffect(Unit) { vm.autoRefresh() }
    val ctx = LocalContext.current
    var range by rememberSaveable { mutableStateOf("6M") }
    val snapRev = Snapshots.rev
    val sr = remember(snapRev, AppState.dataVersion) { Snapshots.series(false) }

    PullToRefreshBox(isRefreshing = vm.loading, onRefresh = { vm.load(force = true) },
        modifier = Modifier.fillMaxSize().background(BgApp)) {
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)) {
            val p = vm.plan
            val orders = p?.let { ordersOf(it) }.orEmpty()
            val doneN = orders.count { it.ticker in vm.checked }

            // 머리 — 지금 상태 한 줄
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    when {
                        p == null -> "홈"
                        p.due && orders.isNotEmpty() -> "맞출 때예요"
                        p.due -> "모두 목표 근처예요"
                        else -> "다음 맞추기 ${mmdd(p.nextEnd)}"
                    },
                    color = TextPrimary, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f),
                )
                when {
                    p == null -> {}
                    p.due && orders.isNotEmpty() -> Pill("할 일 ${orders.size - doneN}", Profit, Profit.copy(alpha = 0.15f))
                    else -> Pill("D-${p.daysLeft}")
                }
            }
            vm.error?.let { Text("⚠️ $it", color = Loss, fontSize = 12.sp) }

            // 총자산 히어로 — 추이(빨강) + 원금(점선)
            val n = RANGES.first { it.first == range }.second
            val from = (sr.total.size - n).coerceAtLeast(0)
            val tot = sr.total.copyOfRange(from, sr.total.size)
            val prin = if (sr.principal.size == sr.total.size) sr.principal.copyOfRange(from, sr.total.size) else DoubleArray(0)
            val nowTotal = p?.accountTotalKrw?.takeIf { it > 0 } ?: sr.total.lastOrNull() ?: 0.0
            val principal = sr.principal.lastOrNull()?.takeIf { it > 0 }
            Card(brush = Brush.linearGradient(listOf(Color(0xFF1D1F2B), Color(0xFF141418))), pad = 16.dp) {
                Label("총자산")
                Text(manwon(nowTotal), color = TextPrimary, fontSize = 30.sp, fontWeight = FontWeight.ExtraBold)
                if (principal != null) {
                    val gain = nowTotal - principal
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(signed(gain), color = if (gain >= 0) Profit else Loss, fontSize = 15.sp, fontWeight = FontWeight.ExtraBold)
                        Spacer(Modifier.width(8.dp))
                        Text("원금 ${manwon(principal)} 대비 ${pct(gain / principal * 100)}", color = TextMuted, fontSize = 12.sp)
                    }
                }
                if (tot.size >= 2) {
                    Spacer(Modifier.height(10.dp))
                    AssetChart(tot, prin, 120.dp)
                    Row(Modifier.padding(top = 4.dp)) {
                        Text("— 총자산  ┄ 원금", color = TextMuted, fontSize = 11.sp, modifier = Modifier.weight(1f))
                        Text("${sr.dates.getOrNull(from).orEmpty()} ~", color = TextMuted, fontSize = 11.sp)
                    }
                } else {
                    Text("자산 기록이 2일 이상 쌓이면 추이가 보입니다", color = TextMuted, fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp))
                }
                Row(Modifier.fillMaxWidth().padding(top = 8.dp)) {
                    RANGES.forEach { (k, _) ->
                        val on = k == range
                        Box(Modifier.weight(1f).clip(RoundedCornerShape(10.dp)).background(if (on) CARD2 else Color.Transparent)
                            .clickable { range = k }.padding(vertical = 7.dp), contentAlignment = Alignment.Center) {
                            Text(k, color = if (on) TextPrimary else TextSecondary, fontSize = 12.sp,
                                fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
                        }
                    }
                }
            }

            if (p == null) {
                Text("PC 에서 계산하는 중…", color = TextSecondary, fontSize = 13.sp)
                return@Column
            }

            // 현금 · 고점 대비
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                val cashPct = if (p.accountTotalKrw > 0) p.cashKrw / p.accountTotalKrw * 100 else p.cashPctNow
                Card(Modifier.weight(1f), pad = 12.dp) {
                    Label("현금")
                    Text("%.0f%%".format(cashPct), color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold)
                    Spacer(Modifier.height(4.dp))
                    StackBar(listOf(cashPct to OK, (100 - cashPct).coerceAtLeast(0.0) to CARD2), 6.dp)
                    Text("목표 ${p.cashPct}%", color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
                }
                val peak = sr.total.maxOrNull() ?: 0.0
                val dd = if (peak > 0 && sr.total.isNotEmpty()) (sr.total.last() / peak - 1) * 100 else 0.0
                val worst = p.cashTable?.firstOrNull { it.key.toIntOrNull() == p.cashPct }?.mdd
                Card(Modifier.weight(1f), pad = 12.dp) {
                    Label("고점 대비")
                    Text(pct(dd), color = if (dd < -0.05) Loss else TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold)
                    Spacer(Modifier.height(4.dp))
                    StackBar(listOf(abs(dd) to Loss, (60 - abs(dd)).coerceAtLeast(1.0) to CARD2), 6.dp)
                    Text(worst?.let { "과거 최악 −%.0f%%".format(it * 100) } ?: "과거 최악 계산 중", color = TextMuted, fontSize = 11.sp,
                        modifier = Modifier.padding(top = 4.dp))
                }
            }

            // 비중 — 지금 / 목표 (자산군)
            WeightCard(p)

            // 맞추기 — 가로로 넘기는 카드
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
                Text("맞추기", color = TextPrimary, fontSize = 17.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f))
                if (orders.isNotEmpty()) Pill("$doneN / ${orders.size}", if (p.due) Profit else TextSecondary,
                    if (p.due) Profit.copy(alpha = 0.15f) else CARD2)
            }
            if (orders.isEmpty()) {
                Card(pad = 14.dp) {
                    Text("모든 종목이 목표에서 20만 원 안쪽이에요", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    Text("다음 맞추기 ${mmdd(p.nextEnd)} · 그 사이 목표의 2배/절반을 넘으면 알려 드려요", color = TextMuted, fontSize = 12.sp)
                }
            } else {
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    orders.forEach { r ->
                        OrderCard(r, checked = r.ticker in vm.checked, onCheck = { vm.toggle(r.ticker) }) {
                            vm.check(r.ticker); TossLink.openOrder(ctx, r.ticker)
                        }
                    }
                }
                val next = orders.firstOrNull { it.ticker !in vm.checked }
                PrimaryButton(
                    if (next != null) "다음: ${Tickers.displayName(next.ticker)} ${if (next.action == "sell") "매도" else "매수"} " +
                        (if (next.outside) "전량" else qtyLabel(next.qty)) else "모두 체크했어요",
                    enabled = next != null,
                ) { next?.let { vm.check(it.ticker); TossLink.openOrder(ctx, it.ticker) } }
                Text(
                    "매도 ${p.sellCount}건 ${manwon(p.sellKrw)} · 매수 ${p.buyCount}건 ${manwon(p.buyKrw)} · 비용 약 ${manwon(p.costKrw)}" +
                        if (p.taxThis > 0) " · 양도세 약 ${manwon(p.taxThis)}" else "",
                    color = TextMuted, fontSize = 11.sp,
                )
            }
            if (p.due) GhostButton("다 맞췄어요", modifier = Modifier.fillMaxWidth()) { vm.done() }
        }
    }
}

/** 자산군별 비중 두 줄(지금/목표) + 범례. 현금은 맨 끝 회색. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun WeightCard(p: Server.Rebalance) {
    val tot = p.accountTotalKrw.takeIf { it > 0 } ?: return
    val nowBy = HashMap<String, Double>()
    val tgtBy = HashMap<String, Double>()
    p.rows.forEach { r ->
        val g = p.groups[r.ticker] ?: "etc"
        nowBy[g] = (nowBy[g] ?: 0.0) + r.valueKrw / tot * 100
        tgtBy[g] = (tgtBy[g] ?: 0.0) + r.targetKrw / tot * 100
    }
    if (!p.sellOutside) p.outside.forEach { (t, v) ->       // 정리하지 않는 바구니 밖은 지금=목표
        val g = p.groups[t] ?: "etc"
        nowBy[g] = (nowBy[g] ?: 0.0) + v / tot * 100; tgtBy[g] = (tgtBy[g] ?: 0.0) + v / tot * 100
    }
    val cashNow = p.cashKrw / tot * 100
    val cashTgt = (100 - tgtBy.values.sum()).coerceAtLeast(0.0)
    val keys = GROUP_ORDER.filter { (nowBy[it] ?: 0.0) + (tgtBy[it] ?: 0.0) > 0.05 }
    Card {
        Row {
            Text("비중", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f))
            Text("지금 / 목표", color = TextMuted, fontSize = 11.sp)
        }
        Spacer(Modifier.height(10.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("지금", color = TextSecondary, fontSize = 11.sp, modifier = Modifier.width(30.dp))
            StackBar(keys.map { (nowBy[it] ?: 0.0) to GROUP_COLOR[it]!! } + (cashNow to CASH_COLOR), 14.dp, Modifier.weight(1f))
        }
        Spacer(Modifier.height(6.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("목표", color = TextSecondary, fontSize = 11.sp, modifier = Modifier.width(30.dp))
            StackBar(keys.map { (tgtBy[it] ?: 0.0) to GROUP_COLOR[it]!! } + (cashTgt to CASH_COLOR), 14.dp, Modifier.weight(1f))
        }
        Spacer(Modifier.height(10.dp))
        // 범례 — 자산군마다 지금 → 목표 숫자
        (keys.map { Triple(GROUP_LABEL[it]!!, GROUP_COLOR[it]!!, (nowBy[it] ?: 0.0) to (tgtBy[it] ?: 0.0)) } +
            Triple("현금", CASH_COLOR, cashNow to cashTgt)).chunked(2).forEach { pair ->
            Row(Modifier.padding(vertical = 2.dp)) {
                pair.forEach { (l, c, v) ->
                    Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(c).border(1.dp, if (c == CASH_COLOR) TextMuted else c, RoundedCornerShape(50)))
                        Spacer(Modifier.width(5.dp))
                        Text(l, color = TextSecondary, fontSize = 11.sp, modifier = Modifier.weight(1f))
                        Text("%.0f%% → %.0f%%".format(v.first, v.second), color = TextPrimary, fontSize = 11.sp, fontFamily = Mono,
                            modifier = Modifier.padding(end = 10.dp))
                    }
                }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun OrderCard(r: Server.RebalRow, checked: Boolean, onCheck: () -> Unit, onToss: () -> Unit) {
    val sell = r.action == "sell"
    Column(
        Modifier.width(128.dp).clip(RoundedCornerShape(16.dp)).background(CARD).clickable { onToss() }
            .padding(12.dp).then(if (checked) Modifier else Modifier),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(Tickers.displayName(r.ticker), color = if (checked) TextMuted else TextPrimary, fontSize = 14.sp,
                fontWeight = FontWeight.ExtraBold, fontFamily = Mono, maxLines = 1, modifier = Modifier.weight(1f))
            Box(Modifier.size(20.dp).clip(RoundedCornerShape(6.dp)).background(if (checked) OK else Color.Transparent)
                .border(1.5.dp, if (checked) OK else TextMuted, RoundedCornerShape(6.dp)).clickable { onCheck() },
                contentAlignment = Alignment.Center) {
                if (checked) Text("✓", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }
        }
        Text("${if (sell) "매도" else "매수"} ${if (r.outside) "전량" else qtyLabel(r.qty)}",
            color = if (checked) TextMuted else if (sell) Loss else Profit, fontSize = 17.sp, fontWeight = FontWeight.ExtraBold,
            modifier = Modifier.padding(top = 6.dp))
        Text(if (r.outside) "바구니 밖 정리" else if (!r.held) "새로 담기" else "목표 대비 ${pct(r.devPct, 0)}",
            color = TextMuted, fontSize = 11.sp)
        Text(manwon(r.amountKrw), color = TextSecondary, fontSize = 11.sp)
    }
}

// ══════════════════════ ② 바구니 ══════════════════════

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BasketScreen(vm: RebalanceViewModel = viewModel()) {
    LaunchedEffect(AppState.dataVersion) { if (vm.plan == null) vm.load() }
    LaunchedEffect(Unit) { vm.autoRefresh() }
    // 과거 성과·종목 숫자는 PC 가 백그라운드로 계산한다 — 아직이면 잠깐씩 다시 묻는다
    LaunchedEffect(vm.plan?.cashTable == null) {
        while (vm.plan != null && vm.plan?.cashTable == null) { delay(4_000); vm.load(quiet = true) }
    }
    var seg by rememberSaveable { mutableStateOf("items") }
    // 타일 아래 가격 추이의 기간(거래일) — 모든 타일에 같이
    var priceDays by rememberSaveable { mutableStateOf(63) }
    Column(Modifier.fillMaxSize().background(BgApp).padding(horizontal = 16.dp)) {
        val p = vm.plan
        Row(Modifier.padding(top = 14.dp, bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            Text("바구니", color = TextPrimary, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f))
            p?.let { Pill("${it.basket.size}종목") }
        }
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(CARD).padding(3.dp)) {
            listOf("items" to "종목", "rules" to "규칙", "history" to "성과").forEach { (k, l) ->
                val on = k == seg
                Box(Modifier.weight(1f).clip(RoundedCornerShape(9.dp)).background(if (on) CARD2 else Color.Transparent)
                    .clickable { seg = k }.padding(vertical = 8.dp), contentAlignment = Alignment.Center) {
                    Text(l, color = if (on) TextPrimary else TextSecondary, fontSize = 13.sp, fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
                }
            }
        }
        // 당겨서 새로고침 — 계좌를 토스에서 바로 다시 받는다
        PullToRefreshBox(isRefreshing = vm.loading, onRefresh = { vm.load(force = true) }, modifier = Modifier.fillMaxSize()) {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(top = 12.dp, bottom = 20.dp)) {
                vm.error?.let { Text("⚠️ $it", color = Loss, fontSize = 12.sp, modifier = Modifier.padding(bottom = 8.dp)) }
                if (p == null) {
                    Text("PC 에서 받는 중…", color = TextSecondary, fontSize = 13.sp)
                    return@Column
                }
                when (seg) {
                    "rules" -> RulesPanel(p, vm)
                    "history" -> HistoryPanel(p)
                    else -> ItemsPanel(p, vm, priceDays) { priceDays = it }
                }
            }
        }
    }
}

private val PRICE_RANGES = listOf(21 to "1개월", 63 to "3개월", 252 to "1년")

@Composable
private fun ItemsPanel(p: Server.Rebalance, vm: RebalanceViewModel, priceDays: Int, onPriceDays: (Int) -> Unit) {
    // 누른 타일 → 종목 상세 시트
    var sheet by rememberSaveable { mutableStateOf<String?>(null) }
    sheet?.let { TickerSheet(it, p, vm) { sheet = null } }
    val rows = p.rows.associateBy { it.ticker }
    val tot = p.totalKrw.takeIf { it > 0 } ?: 1.0
    // 막대 읽는 법 — 두 줄 범례
    Row(Modifier.padding(bottom = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("막대  ", color = TextMuted, fontSize = 11.sp)
        Box(Modifier.size(14.dp, 6.dp).clip(RoundedCornerShape(3.dp)).background(GROUP_COLOR["us"]!!))
        Text(" 지금  ", color = TextSecondary, fontSize = 11.sp)
        Box(Modifier.size(2.dp, 12.dp).background(Color.White))
        Text(" 목표", color = TextSecondary, fontSize = 11.sp)
    }
    Row(Modifier.padding(bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("        ", color = TextMuted, fontSize = 11.sp)
        Box(Modifier.size(14.dp, 6.dp).background(Loss))
        Text(" 많음(팔기)  ", color = TextSecondary, fontSize = 11.sp)
        Box(Modifier.size(14.dp, 6.dp).background(Profit.copy(alpha = 0.35f)))
        Text(" 모자람(사기)", color = TextSecondary, fontSize = 11.sp)
    }
    Text("타일을 누르면 종목 상세 — 평단선 차트·매매 기록·주문", color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(bottom = 8.dp))
    // 가격 추이 기간 — 타일 맨 아래 줄에 쓴다(내 수익률과 헷갈리지 않게 라벨을 붙여 작게)
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(bottom = 6.dp)) {
        Text("가격 추이", color = TextMuted, fontSize = 12.sp, modifier = Modifier.weight(1f))
        PRICE_RANGES.forEach { (d, l) ->
            val on = d == priceDays
            Box(Modifier.padding(start = 4.dp).clip(RoundedCornerShape(99.dp)).background(if (on) CARD2 else Color.Transparent)
                .clickable { onPriceDays(d) }.padding(horizontal = 10.dp, vertical = 5.dp)) {
                Text(l, color = if (on) TextPrimary else TextMuted, fontSize = 12.sp, fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
            }
        }
    }
    val byGroup = p.basket.groupBy { p.groups[it] ?: "etc" }
    GROUP_ORDER.filter { byGroup[it] != null }.forEach { g ->
        val members = byGroup[g]!!
        val now = members.sumOf { (rows[it]?.valueKrw ?: 0.0) } / tot * 100
        val tg = members.sumOf { (rows[it]?.targetKrw ?: 0.0) } / tot * 100
        Row(Modifier.padding(top = 6.dp, bottom = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(4.dp, 16.dp).clip(RoundedCornerShape(2.dp)).background(GROUP_COLOR[g]!!))
            Spacer(Modifier.width(8.dp))
            Text(GROUP_LABEL[g]!!, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
            Text("%.1f%% → %.1f%%".format(now, tg), color = TextMuted, fontSize = 12.sp)
        }
        members.chunked(2).forEach { pair ->
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(bottom = 8.dp)) {
                pair.forEach { t -> Tile(t, p, rows[t], tot, priceDays, Modifier.weight(1f)) { sheet = t } }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
    }
    // 바구니 밖 · 현금
    if (p.outside.isNotEmpty()) {
        Row(Modifier.padding(top = 10.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(4.dp, 16.dp).clip(RoundedCornerShape(2.dp)).background(TextMuted))
            Spacer(Modifier.width(8.dp))
            Text("바구니 밖 ${p.outside.size}", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
            Text(manwon(p.outsideKrw), color = TextMuted, fontSize = 12.sp)
        }
        p.outside.forEach { (t, v) ->
            val r = (p.outsidePnl[t] ?: 0.0) * 100
            Row(Modifier.fillMaxWidth().height(46.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(Tickers.displayName(t), color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold, fontFamily = Mono, modifier = Modifier.weight(1f))
                Column(horizontalAlignment = Alignment.End) {
                    Text(manwon(v), color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    Text(pct(r), color = if (r >= 0) Profit else Loss, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                }
            }
            HDivider()
        }
        Text(if (p.sellOutside) "규칙: 정리 — 홈의 맞추기에 '전량 매도'로" else "규칙: 그대로 두기 — 리밸런싱 계산에서 뺌",
            color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
    }
    Row(Modifier.fillMaxWidth().padding(top = 10.dp).height(46.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(CASH_COLOR).border(1.dp, TextMuted, RoundedCornerShape(50)))
        Spacer(Modifier.width(8.dp))
        Text("현금", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
        Text(manwon(p.cashKrw), color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
    }
}

/**
 * 종목 타일 — **내 수익률이 주인공**. 위: 이름·배수·사기/팔기 / 크게: 평가손익률 / 작게: 손익 금액·보유 금액 /
 * 막대: 지금·목표 비중 / 맨 아래 한 줄: 라벨 붙인 가격 추이(기간은 위에서 고른다).
 * 예전엔 가격 1년 등락률을 크게 써서 내 수익률로 읽혔다(사용자 지적, 2026-10-22).
 */
@Composable
private fun Tile(t: String, p: Server.Rebalance, r: Server.RebalRow?, tot: Double, priceDays: Int, modifier: Modifier, onOpen: () -> Unit) {
    val gc = GROUP_COLOR[p.groups[t] ?: "etc"]!!
    val m = p.metrics[t]
    // 누르면 종목 상세 시트(설명·⚠ 의 뜻도 거기에)
    Column(modifier.clip(RoundedCornerShape(16.dp)).background(CARD).clickable { onOpen() }.padding(12.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(8.dp).clip(RoundedCornerShape(50)).background(gc))
            Spacer(Modifier.width(6.dp))
            Text(Tickers.displayName(t), color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.ExtraBold, fontFamily = Mono, maxLines = 1)
            levOf(p.desc[t])?.let { Text(" ${it}x", color = TextMuted, fontSize = 11.sp) }
            if (p.notes[t] != null) Text(" ⚠", color = Gold, fontSize = 11.sp)
            Spacer(Modifier.weight(1f))
            when (r?.action) {
                "buy" -> Text("▲ 사기", color = Profit, fontSize = 11.sp, fontWeight = FontWeight.ExtraBold)
                "sell" -> Text("▼ 팔기", color = Loss, fontSize = 11.sp, fontWeight = FontWeight.ExtraBold)
                else -> Text("✓", color = TextMuted, fontSize = 11.sp)
            }
        }
        // 내 수익률 — 크게
        if (r != null && r.held) {
            val pr = r.pnlRate * 100
            val gain = r.valueKrw * r.pnlRate / (1 + r.pnlRate)
            Text(pct(pr), color = if (pr >= 0) Profit else Loss, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold,
                modifier = Modifier.padding(top = 4.dp))
            Text("${signed(gain)} · 보유 ${manwon(r.valueKrw)}", color = TextSecondary, fontSize = 11.sp, maxLines = 1)
        } else {
            Text("미보유", color = TextMuted, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.padding(top = 4.dp))
            Text(r?.let { "새로 담기 ${manwon(it.targetKrw)}" } ?: "", color = TextSecondary, fontSize = 11.sp, maxLines = 1)
        }
        // 지금 / 목표 비중 — 막대 + 뜻을 글로
        if (r != null) {
            val now = r.valueKrw / tot * 100
            val tg = r.targetKrw / tot * 100
            val gap = now - tg
            Spacer(Modifier.height(8.dp))
            TargetBar(now, tg, gc)
            val (txt, col) = when (r.action) {
                "sell" -> "목표보다 %.1f%%p 많음".format(gap) to Loss
                "buy" -> "목표보다 %.1f%%p 모자람".format(-gap) to Profit
                else -> "목표 근처" to TextMuted
            }
            Text(txt, color = col, fontSize = 11.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 3.dp))
            Text("지금 %.1f%% · 목표 %.1f%%".format(now, tg), color = TextMuted, fontSize = 10.sp)
        }
        // 가격 추이 — 라벨을 붙여 맨 아래에 작게
        if (m != null && m.spark.size > 2) {
            val sl = m.spark.copyOfRange((m.spark.size - 1 - priceDays).coerceAtLeast(0), m.spark.size)
            val chg = (sl.last() / sl.first() - 1) * 100
            HDivider(Modifier.padding(top = 8.dp, bottom = 6.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("가격 ${PRICE_RANGES.first { it.first == priceDays }.second}", color = TextMuted, fontSize = 10.sp)
                Spacer(Modifier.width(6.dp))
                Sparkline(sl, 46.dp, 16.dp)
                Spacer(Modifier.weight(1f))
                Text(pct(chg, 0), color = TextSecondary, fontSize = 11.sp)
            }
        }
    }
}

private val SHEET_RANGES = listOf(5 to "1주", 21 to "1M", 63 to "3M", 252 to "1Y", 0 to "전체")

/** 1주 가격 표시 — 원화는 정수, 달러는 소수 2자리. */
private fun money(v: Double, krw: Boolean) = if (krw) "%,d원".format(v.roundToLong()) else "$%,.2f".format(v)

/**
 * 종목 상세 시트(목업 research/mockup_v7 의 1-B) — 내 보유 먼저.
 * 내 수익률·목표 막대 → 지금 가격·평단 대비 → 평단선·매매 점 차트 → 매매 기록 → 아래 고정 주문 버튼.
 * 일봉·매매·평단은 서버의 종목 분석(/api/analysis)을 그대로 쓴다(받아 둔 게 있으면 먼저 보여 준다).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun TickerSheet(t: String, p: Server.Rebalance, vm: RebalanceViewModel, onClose: () -> Unit) {
    val ctx = LocalContext.current
    val r = p.rows.firstOrNull { it.ticker == t }
    val tot = p.totalKrw.takeIf { it > 0 } ?: 1.0
    val gc = GROUP_COLOR[p.groups[t] ?: "etc"]!!
    var an by remember(t) { mutableStateOf(AnalysisCache.peek(t)?.a) }
    var err by remember(t) { mutableStateOf<String?>(null) }
    var days by rememberSaveable { mutableIntStateOf(63) }
    LaunchedEffect(t) {
        if (an == null) withContext(Dispatchers.IO) { runCatching { AnalysisCache.get(t) }.getOrNull() }?.let { an = it.a }
        withContext(Dispatchers.IO) { runCatching { AnalysisCache.fetch(t) } }
            .onSuccess { an = it }.onFailure { if (an == null) err = it.message ?: "받지 못했어요" }
    }
    ModalBottomSheet(onDismissRequest = onClose, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        containerColor = BgApp) {
        Column(Modifier.fillMaxWidth().fillMaxHeight(0.92f)) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)) {
                // 머리 — 이름·배수·설명·닫기
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(10.dp).clip(RoundedCornerShape(50)).background(gc))
                    Spacer(Modifier.width(8.dp))
                    Text(Tickers.displayName(t), color = TextPrimary, fontSize = 24.sp, fontWeight = FontWeight.ExtraBold, fontFamily = Mono)
                    Spacer(Modifier.width(8.dp))
                    levOf(p.desc[t])?.let { Pill("${it}x") }
                    Spacer(Modifier.weight(1f))
                    Text("✕", color = TextSecondary, fontSize = 20.sp, modifier = Modifier.clickable { onClose() }.padding(6.dp))
                }
                p.desc[t]?.let { Text(it, color = TextSecondary, fontSize = 12.sp, lineHeight = 17.sp) }
                p.notes[t]?.let { Text("⚠ $it", color = Gold, fontSize = 12.sp, lineHeight = 17.sp) }

                // 내 수익률 — 주인공
                val pr = (r?.pnlRate ?: 0.0) * 100
                val held = r != null && r.held
                val qty = an?.qty ?: 0.0
                val tone = if (!held) TextMuted else if (pr >= 0) Profit else Loss
                Card(brush = Brush.linearGradient(listOf(tone.copy(alpha = 0.22f), CARD))) {
                    Text("내 수익률", color = TextSecondary, fontSize = 12.sp)
                    Row(verticalAlignment = Alignment.Bottom) {
                        Text(if (held) pct(pr) else "미보유", color = tone, fontSize = 32.sp, fontWeight = FontWeight.ExtraBold,
                            modifier = Modifier.weight(1f))
                        Column(horizontalAlignment = Alignment.End) {
                            if (r != null && held) {
                                Text(signed(r.valueKrw * r.pnlRate / (1 + r.pnlRate)), color = tone, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                                Text("보유 ${manwon(r.valueKrw)}" + (if (qty > 0) " · ${qtyLabel(qty)}" else ""), color = TextSecondary, fontSize = 12.sp)
                            } else if (r != null) {
                                Text("목표 ${manwon(r.targetKrw)}", color = TextSecondary, fontSize = 12.sp)
                            }
                        }
                    }
                    if (r != null) {
                        val now = r.valueKrw / tot * 100
                        val tg = r.targetKrw / tot * 100
                        Spacer(Modifier.height(10.dp))
                        TargetBar(now, tg, gc)
                        Row(Modifier.padding(top = 4.dp)) {
                            val (txt, col) = when (r.action) {
                                "sell" -> "목표보다 %.1f%%p 많음".format(now - tg) to Loss
                                "buy" -> "목표보다 %.1f%%p 모자람".format(tg - now) to Profit
                                else -> "목표 근처" to TextMuted
                            }
                            Text(txt, color = col, fontSize = 12.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                            Text("지금 %.1f%% · 목표 %.1f%%".format(now, tg), color = TextMuted, fontSize = 11.sp)
                        }
                    }
                }

                val a = an
                if (a == null) {
                    Text(err?.let { "⚠️ $it" } ?: "차트 받는 중…", color = if (err != null) Loss else TextSecondary, fontSize = 13.sp)
                } else {
                    val cs = a.candles
                    // 지금 가격 · 하루 등락 · 평단 대비
                    if (cs.size >= 2) {
                        val last = cs.last().close
                        val dc = (last / cs[cs.size - 2].close - 1) * 100
                        Row(verticalAlignment = Alignment.Bottom, modifier = Modifier.padding(top = 4.dp)) {
                            Text(money(last, a.krw), color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold, fontFamily = Mono)
                            Text("  ${pct(dc)}", color = if (dc >= 0) Profit else Loss, fontSize = 13.sp, fontWeight = FontWeight.Bold,
                                modifier = Modifier.padding(bottom = 2.dp))
                            Spacer(Modifier.weight(1f))
                            if (a.qty > 0 && a.avgPrice > 0) {
                                val vs = (last / a.avgPrice - 1) * 100
                                Text("평단 ${money(a.avgPrice, a.krw)} 대비 ", color = TextMuted, fontSize = 12.sp, modifier = Modifier.padding(bottom = 2.dp))
                                Text(pct(vs, 0), color = if (vs >= 0) Profit else Loss, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                                    modifier = Modifier.padding(bottom = 2.dp))
                            }
                        }
                    }
                    // 차트 — 일봉 시각은 UTC 자정이라 날짜는 epoch 일로 바로 바꾼다
                    val n = if (days == 0) cs.size else minOf(cs.size, days + 1)
                    val sl = cs.takeLast(n)
                    val dates = sl.map { java.time.LocalDate.ofEpochDay(Math.floorDiv(it.t, 86_400L)).toString() }
                    val marks = a.trades.mapNotNull { tr ->
                        if (dates.isEmpty() || tr.date < dates.first()) return@mapNotNull null
                        val i = dates.indexOfFirst { it >= tr.date }.let { if (it < 0) dates.size - 1 else it }
                        i to (tr.type == "buy")
                    }
                    val hasAvg = a.qty > 0 && a.avgPrice > 0
                    Card(pad = 12.dp) {
                        PriceChart(sl.map { it.close }.toDoubleArray(), a.avgPrice.takeIf { hasAvg },
                            if (hasAvg) "avg ${money(a.avgPrice, a.krw)}" else null, marks, 180.dp)
                        Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            SHEET_RANGES.forEach { (d, l) ->
                                val on = d == days
                                Box(Modifier.clip(RoundedCornerShape(99.dp)).background(if (on) CARD2 else Color.Transparent)
                                    .clickable { days = d }.padding(horizontal = 12.dp, vertical = 6.dp)) {
                                    Text(l, color = if (on) TextPrimary else TextMuted, fontSize = 12.sp, fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
                                }
                            }
                        }
                    }
                    // 매매 기록
                    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
                        Text("매매 기록", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f))
                        Text("${a.trades.size}건", color = TextMuted, fontSize = 12.sp)
                    }
                    if (a.trades.isEmpty()) Text("아직 사고판 기록이 없어요", color = TextMuted, fontSize = 12.sp)
                    Column {
                        a.trades.sortedByDescending { it.date }.take(12).forEach { tr ->
                            val buy = tr.type == "buy"
                            Row(Modifier.fillMaxWidth().height(40.dp), verticalAlignment = Alignment.CenterVertically) {
                                Text(tr.date.drop(2).replace('-', '.'), color = TextSecondary, fontSize = 13.sp, fontFamily = Mono, modifier = Modifier.width(82.dp))
                                Text(if (buy) "매수" else "매도", color = if (buy) Profit else Loss, fontSize = 13.sp, fontWeight = FontWeight.Bold,
                                    modifier = Modifier.width(44.dp))
                                Text(qtyLabel(tr.qty), color = TextPrimary, fontSize = 13.sp, modifier = Modifier.weight(1f))
                                Text(money(tr.price, a.krw), color = TextPrimary, fontSize = 13.sp, fontFamily = Mono)
                            }
                            HDivider()
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
            }
            // 아래 고정 — 목표까지 주문
            val act = r?.action
            if (r != null && (act == "buy" || act == "sell")) {
                val sell = act == "sell"
                val c = if (sell) Loss else Profit
                Row(Modifier.fillMaxWidth().background(c.copy(alpha = 0.14f)).padding(horizontal = 16.dp, vertical = 12.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("목표까지 ${if (r.outside) "전량" else qtyLabel(r.qty)} ${if (sell) "매도" else "매수"}",
                            color = c, fontSize = 15.sp, fontWeight = FontWeight.ExtraBold)
                        Text("약 ${manwon(r.amountKrw)}", color = TextSecondary, fontSize = 12.sp)
                    }
                    Box(Modifier.clip(RoundedCornerShape(12.dp)).background(c).clickable { vm.check(t); TossLink.openOrder(ctx, t) }
                        .padding(horizontal = 18.dp, vertical = 12.dp)) {
                        Text("토스에서 주문", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    }
                }
            } else if (r != null) {
                Text("✓ 목표 근처 — 이번엔 할 일 없음", color = TextMuted, fontSize = 13.sp,
                    modifier = Modifier.fillMaxWidth().padding(16.dp))
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RulesPanel(p: Server.Rebalance, vm: RebalanceViewModel) {
    var period by remember(p) { mutableStateOf(p.period) }
    var fractional by remember(p) { mutableStateOf(p.fractional) }
    var sellOutside by remember(p) { mutableStateOf(p.sellOutside) }
    var cash by remember(p) { mutableFloatStateOf(p.cashPct.toFloat()) }
    val basket = remember(p) { mutableStateListOf<String>().also { it.addAll(p.basket) } }

    SectionLabel("언제 맞출까", top = 4.dp)
    val best = p.periodTable?.maxByOrNull { it.cagr }?.key ?: "quarterly"
    UnderlineSegments(
        listOf("monthly", "quarterly", "yearly").map { it to (PERIOD_LABEL[it]!! + if (it == best) " (추천)" else "") },
        selected = period, onSelect = { period = it },
    )
    p.periodTable?.let { t ->
        Text("과거(${p.tableFrom.take(4)}~) 세후: " + t.joinToString(" · ") { "${PERIOD_LABEL[it.key]} %+.1f%%".format(it.cagr * 100) },
            color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 6.dp))
    }

    SectionLabel("현금으로 둘 비중")
    Row(verticalAlignment = Alignment.Bottom) {
        Text("${cash.toInt()}%", color = Gold, fontSize = 24.sp, fontWeight = FontWeight.Bold, fontFamily = Mono,
            modifier = Modifier.weight(1f))
        p.cashTable?.firstOrNull { it.key.toIntOrNull() == cash.toInt() }?.let {
            Text("과거 최악 −%.0f%% · 세후 연 %+.1f%%".format(it.mdd * 100, it.cagr * 100), color = TextSecondary, fontSize = 12.sp)
        }
    }
    Slider(
        value = cash, onValueChange = { cash = it }, valueRange = 0f..50f, steps = 4,     // 0·10·…·50
        colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Gold,
            inactiveTrackColor = SurfaceInput, activeTickColor = Gold, inactiveTickColor = TextMuted),
    )
    Text("떨어지면 현금 비율이 커지고, 맞출 때 그 현금으로 빠진 종목을 산다 · 이자 연 2.5% 가정 · 자세히는 '과거 성과'",
        color = TextMuted, fontSize = 11.sp, lineHeight = 15.sp)

    SectionLabel("미국 종목 주문 단위")
    UnderlineSegments(listOf("whole" to "정수 주", "frac" to "소수점(0.01주)"),
        selected = if (fractional) "frac" else "whole", onSelect = { fractional = it == "frac" })

    SectionLabel("바구니 밖 보유")
    UnderlineSegments(listOf("keep" to "그대로 두기", "sell" to "정리(전량 매도)"),
        selected = if (sellOutside) "sell" else "keep", onSelect = { sellOutside = it == "sell" })
    Text("정리 = 계좌 전체를 바구니 + 현금으로 맞춘다. 판 이익은 양도세에 들어간다(손실 종목과 섞어 팔면 줄어든다).",
        color = TextMuted, fontSize = 11.sp, lineHeight = 15.sp, modifier = Modifier.padding(top = 6.dp))

    SectionLabel("바구니 (동일 비중으로 담을 종목)")
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
        (p.defaultBasket + p.watchlist + basket).distinct().forEach { t ->
            val on = t in basket
            Box(
                Modifier.clip(RoundedCornerShape(8.dp)).background(if (on) Gold.copy(alpha = 0.18f) else SurfaceInput)
                    .border(BorderStroke(1.dp, if (on) Gold else Color.Transparent), RoundedCornerShape(8.dp))
                    .clickable { if (on) basket.remove(t) else basket.add(t) }
                    .padding(horizontal = 10.dp, vertical = 6.dp),
            ) { Text(Tickers.displayName(t), color = if (on) Gold else TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.Bold) }
        }
    }
    Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        if (p.defaultBasket.isNotEmpty()) {
            GhostButton("기본 ${p.defaultBasket.size}개") { basket.clear(); basket.addAll(p.defaultBasket) }
        }
        GhostButton("관심종목 전체") { basket.clear(); basket.addAll(p.watchlist) }
    }
    Text("기본은 단일 종목 ETF 를 뺀 여러 종목 레버리지 ETF 의 상관 묶음 대표. 배수는 기초자산이 감당하는 만큼 — " +
        "S&P 만 3배, 나머지는 2배가 있으면 2배.", color = TextMuted, fontSize = 11.sp, lineHeight = 15.sp,
        modifier = Modifier.padding(top = 8.dp))

    Spacer(Modifier.height(16.dp))
    val changed = period != p.period || fractional != p.fractional || sellOutside != p.sellOutside ||
        cash.toInt() != p.cashPct || basket.toList() != p.basket
    PrimaryButton(if (changed) "저장" else "바뀐 것 없음", enabled = changed && basket.isNotEmpty()) {
        vm.save(basket.toList(), cash.toInt(), period, fractional, sellOutside)
    }
}

@Composable
private fun HistoryPanel(p: Server.Rebalance) {
    val table = p.cashTable
    if (table == null) {
        Text("과거 성과를 PC 가 계산하는 중… (10초쯤)", color = TextSecondary, fontSize = 13.sp)
        return
    }
    SectionLabel("현금 비중별 — 이 바구니 ${p.basket.size}종목, ${p.tableFrom.take(4)}년~, ${PERIOD_LABEL[p.period]}, 세후", top = 4.dp)
    CashChart(table, p.cashPct)
    Row(Modifier.fillMaxWidth().padding(top = 6.dp)) {
        listOf("현금", "세후 연수익", "최대 낙폭", "1억 최악일 때").forEachIndexed { i, h ->
            Text(h, color = TextMuted, fontSize = 11.sp, modifier = Modifier.weight(if (i == 0) 0.7f else 1f))
        }
    }
    table.forEach { r ->
        val sel = r.key.toIntOrNull() == p.cashPct
        val c = if (sel) Gold else TextPrimary
        val fw = if (sel) FontWeight.Bold else FontWeight.Normal
        HDivider()
        Row(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
            Text("${r.key}%", color = c, fontWeight = fw, fontSize = 12.sp, modifier = Modifier.weight(0.7f))
            Text("%+.1f%%".format(r.cagr * 100), color = c, fontWeight = fw, fontSize = 12.sp, modifier = Modifier.weight(1f))
            Text("−%.0f%%".format(r.mdd * 100), color = c, fontWeight = fw, fontSize = 12.sp, modifier = Modifier.weight(1f))
            Text(manwon(r.worst), color = c, fontWeight = fw, fontSize = 12.sp, modifier = Modifier.weight(1f))
        }
    }
    p.periodTable?.let { t ->
        SectionLabel("맞추는 주기별 (현금 ${p.cashPct}%)")
        t.forEach { r ->
            Row(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Text(PERIOD_LABEL[r.key] ?: r.key, color = TextPrimary, fontSize = 12.sp, modifier = Modifier.weight(0.7f))
                Text("%+.1f%%".format(r.cagr * 100), color = TextPrimary, fontSize = 12.sp, modifier = Modifier.weight(1f))
                Text("−%.0f%%".format(r.mdd * 100), color = TextPrimary, fontSize = 12.sp, modifier = Modifier.weight(1f))
                Spacer(Modifier.weight(1f))
            }
        }
    }
    Text("지금 고른 종목으로 돌린 백테스트라 절대 수익은 과장돼 있다 — 현금·주기에 따라 무엇을 얼마나 잃는지의 비교로만 볼 것.",
        color = TextMuted, fontSize = 11.sp, lineHeight = 15.sp, modifier = Modifier.padding(top = 10.dp))
}

/** 현금 비중별 세후 연수익(빨강)·최대 낙폭(파랑). 라벨은 영문(그래프 규칙). */
@Composable
private fun CashChart(rows: List<Server.RebalStat>, selected: Int) {
    Box(Modifier.fillMaxWidth().height(170.dp).padding(top = 8.dp)) {
        Canvas(Modifier.fillMaxSize().padding(start = 30.dp, end = 8.dp, top = 6.dp, bottom = 22.dp)) {
            val xs = rows.map { it.key.toFloatOrNull() ?: 0f }
            val xMax = (xs.maxOrNull() ?: 50f).coerceAtLeast(1f)
            val yMax = 0.8f
            fun pt(x: Float, y: Double) = Offset(size.width * x / xMax, size.height * (1f - (y.toFloat() / yMax).coerceIn(0f, 1f)))
            drawLine(BorderColor, Offset(0f, size.height), Offset(size.width, size.height))
            drawLine(BorderColor, Offset(0f, 0f), Offset(0f, size.height))
            for ((series, col) in listOf(rows.map { it.mdd } to Loss, rows.map { it.cagr } to Profit)) {
                for (i in 1 until rows.size) drawLine(col, pt(xs[i - 1], series[i - 1]), pt(xs[i], series[i]), strokeWidth = 5f)
            }
            val sx = size.width * selected / xMax
            drawLine(Gold, Offset(sx, 0f), Offset(sx, size.height), strokeWidth = 2f,
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 8f)))
        }
        Text("80%", color = TextMuted, fontSize = 9.sp, fontFamily = Mono)
        Text("0%", color = TextMuted, fontSize = 9.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.BottomStart).offset(y = (-18).dp))
        Text("Max drawdown", color = Loss, fontSize = 10.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.TopEnd).offset(y = 18.dp))
        Text("CAGR after tax", color = Profit, fontSize = 10.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.CenterEnd).offset(y = 30.dp))
        Text("Cash weight 0 → 50%", color = TextMuted, fontSize = 10.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.BottomCenter))
    }
}

// ══════════════════════ ③ 더보기 ══════════════════════

/** 나머지 — 손익·세금, 계좌 상세(예전 포트폴리오), 종목 분석(예전 비교·분석), 설정. */
@Composable
fun MoreScreen(onOpen: (String) -> Unit) {
    Column(Modifier.fillMaxSize().background(BgApp).padding(horizontal = 16.dp)) {
        Text("더보기", color = TextPrimary, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.padding(top = 14.dp, bottom = 10.dp))
        listOf(
            Triple("pnl", "손익·세금", "평단 대비 지금 · 해외 양도세 공제 · 원금"),
            Triple("account", "계좌 상세", "보유 종목 전체 · 오늘 손익 · 자산 비중"),
            Triple("compare", "종목 분석", "관심종목 표 · Z·M 차트 · 장기 제안"),
            Triple("settings", "설정", "집 PC 연결 · 관심종목 · 원금 기록 · 앱 업데이트"),
        ).forEach { (id, title, sub) ->
            Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(16.dp)).background(CARD).clickable { onOpen(id) }.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(title, color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    Text(sub, color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 2.dp))
                }
                Text("›", color = TextMuted, fontSize = 18.sp)
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

/** 손익·세금 — 해외 양도세 공제 게이지, 평단 대비 지금, 원금. */
@Composable
fun PnlScreen(vm: RebalanceViewModel = viewModel(), onBack: () -> Unit) {
    LaunchedEffect(Unit) { if (vm.plan == null) vm.load() }
    androidx.activity.compose.BackHandler { onBack() }
    val snapRev = Snapshots.rev
    val sr = remember(snapRev) { Snapshots.series(false) }
    Column(Modifier.fillMaxSize().background(BgApp).verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("‹", color = TextSecondary, fontSize = 24.sp, modifier = Modifier.clickable { onBack() }.padding(end = 12.dp))
            Text("손익·세금", color = TextPrimary, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold)
        }
        val p = vm.plan ?: run { Text("PC 에서 받는 중…", color = TextSecondary, fontSize = 13.sp); return@Column }

        Card {
            Row { Text("해외 양도세", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f))
                Text("올해 · 공제 250만", color = TextMuted, fontSize = 11.sp) }
            Row(Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                SemiGauge(p.taxYtd / 2_500_000.0, 140.dp, if (p.taxYtd > 2_500_000) Profit else OK)
                Spacer(Modifier.width(12.dp))
                Column {
                    Text(manwon(p.taxYtd), color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold)
                    Text("실현 이익(추정)", color = TextMuted, fontSize = 11.sp)
                    val left = 2_500_000 - p.taxYtd
                    Text(if (left > 0) "${manwon(left)}까지 세금 없음" else "초과분 22% ≈ ${manwon(-left * 0.22)}",
                        color = if (left > 0) OK else Profit, fontSize = 12.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 4.dp))
                }
            }
        }

        // 평단 대비 지금 — 바구니 안·밖 보유 전체
        val held = p.rows.filter { it.held && !it.outside }.map { Triple(it.ticker, it.valueKrw, it.pnlRate * 100) } +
            p.outside.map { (t, v) -> Triple(t, v, (p.outsidePnl[t] ?: 0.0) * 100) }
        Card {
            Row { Text("평단 대비 지금", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f))
                Text("│ 평단  ● 지금", color = TextMuted, fontSize = 11.sp) }
            Spacer(Modifier.height(6.dp))
            held.sortedByDescending { it.second }.forEach { (t, v, r) ->
                Row(Modifier.fillMaxWidth().height(38.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(Tickers.displayName(t), color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.Bold, fontFamily = Mono,
                        maxLines = 1, modifier = Modifier.width(64.dp))
                    BulletBar(r, Modifier.weight(1f))
                    Text(pct(r, 0), color = if (r >= 0) Profit else Loss, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                        modifier = Modifier.width(52.dp).padding(start = 6.dp))
                }
            }
        }

        Card {
            val principal = sr.principal.lastOrNull() ?: 0.0
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("원금", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.ExtraBold, modifier = Modifier.weight(1f))
                Text(if (principal > 0) manwon(principal) else "기록 없음", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Bold)
            }
            Text("입금 기록은 더보기 › 설정에서", color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
        }
    }
}

/** 평단(가운데 흰 선) 대비 지금(점) — ±50% 에서 자른다. */
@Composable
private fun BulletBar(r: Double, modifier: Modifier) {
    val c = if (r >= 0) Profit else Loss
    Canvas(modifier.height(16.dp)) {
        val lim = 50.0
        val x0 = size.width / 2
        val x = x0 + (r.coerceIn(-lim, lim) / lim).toFloat() * (size.width / 2 - 6f)
        val cy = size.height / 2
        drawLine(CARD2, Offset(0f, cy), Offset(size.width, cy), strokeWidth = 8f, cap = androidx.compose.ui.graphics.StrokeCap.Round)
        drawLine(c.copy(alpha = 0.55f), Offset(x0, cy), Offset(x, cy), strokeWidth = 10f)
        drawLine(TextPrimary, Offset(x0, 0f), Offset(x0, size.height), strokeWidth = 3f)
        drawCircle(c, 12f, Offset(x, cy))
    }
}
