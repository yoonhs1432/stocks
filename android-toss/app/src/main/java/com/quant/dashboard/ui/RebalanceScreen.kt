package com.quant.dashboard.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Server
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
 * 리밸런싱 중심 화면 — 홈(맞추기)과 바구니. 계산은 PC 가 한다(web/rebalance.py).
 * 이 앱은 주문을 내지 않고 토스 주문 화면으로 건네준다.
 * 색은 한국식: 매수(목표보다 적음)=빨강, 매도(목표보다 많음)=파랑.
 * 목업: research/mockup_app_v2.png — 홈과 '맞추기'를 한 화면으로 합쳤고(겹쳐서),
 * 종목별 지금/목표 비중은 바구니 카드에만 둔다.
 */

private val PERIOD_LABEL = mapOf("monthly" to "매월", "quarterly" to "분기", "yearly" to "매년")
private val PERIOD_SUB = mapOf("monthly" to "매월", "quarterly" to "분기마다", "yearly" to "매년")
private val GREEN = Color(0xFF2EA078)

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

private fun mmdd(iso: String): String = iso.takeIf { it.length >= 10 }?.let {
    "${it.substring(5, 7).toInt()}/${it.substring(8, 10).toInt()}"
} ?: iso

private fun qtyLabel(q: Double): String =
    if (q == kotlin.math.floor(q)) "%,d주".format(q.toLong()) else "%.2f주".format(q)

/** 1주 가격 — 미국은 달러와 원화 둘 다. */
private fun priceLabel(r: Server.RebalRow): String = when {
    r.price.isNaN() || r.price <= 0 -> ""
    r.krw -> "1주 %,d원".format(r.price.roundToLong())
    else -> "1주 $%.2f(%s)".format(r.price, manwon(r.priceKrw))
}

/** 매도 먼저(현금을 먼저 만든다, 큰 것부터) → 매수(큰 것부터). */
private fun ordersOf(p: Server.Rebalance): List<Server.RebalRow> =
    p.rows.filter { it.action == "sell" }.sortedByDescending { it.amountKrw } +
        p.rows.filter { it.action == "buy" }.sortedByDescending { it.amountKrw }

private fun levOf(desc: String?): String? = desc?.let { Regex("(\\d(?:\\.\\d+)?)배").find(it)?.groupValues?.get(1) }

// ══════════════════════ ① 홈 = 맞추기 ══════════════════════

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(vm: RebalanceViewModel = viewModel()) {
    LaunchedEffect(AppState.dataVersion) { vm.load() }
    val ctx = LocalContext.current
    Column(Modifier.fillMaxSize().background(BgApp)) {
        ScreenHeader("홈") {
            val p = vm.plan
            if (p != null) Text("동일 비중 ${p.basket.size} · ${PERIOD_SUB[p.period]} · 현금 ${p.cashPct}%",
                color = TextMuted, fontSize = 11.sp)
        }
        PullToRefreshBox(isRefreshing = vm.loading, onRefresh = { vm.load() }, modifier = Modifier.fillMaxSize()) {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                .padding(horizontal = ScreenPad).padding(top = 12.dp, bottom = 20.dp)) {
                vm.error?.let { Text("⚠️ $it", color = Loss, fontSize = 12.sp, modifier = Modifier.padding(bottom = 8.dp)) }
                val p = vm.plan ?: run {
                    Text(if (vm.error == null) "PC 에서 계산하는 중…" else "PC 에 닿으면 다시 계산합니다",
                        color = TextSecondary, fontSize = 13.sp)
                    return@Column
                }
                val orders = ordersOf(p)
                val doneN = orders.count { it.ticker in vm.checked }

                StatusCard(p, orders.size, doneN)

                SectionLabel("계좌")
                Text(manwon(p.accountTotalKrw), color = TextPrimary, fontSize = 26.sp,
                    fontWeight = FontWeight.ExtraBold, fontFamily = Mono)
                Text("바구니 ${manwon(p.basketKrw)} · 바구니 밖 ${manwon(p.outsideKrw)} · 현금 ${manwon(p.cashKrw)}",
                    color = TextSecondary, fontSize = 12.sp)

                SectionLabel("비중 — 지금 / 목표")
                WeightBars(p)

                SectionLabel(if (orders.isEmpty()) "할 일" else "맞추기 — 매도 먼저, 그다음 매수 · $doneN / ${orders.size}")
                if (orders.isEmpty()) {
                    Text("모든 종목이 목표에서 20만 원 안쪽입니다 — 다음 맞추기 ${mmdd(p.nextEnd)}",
                        color = TextSecondary, fontSize = 13.sp)
                } else {
                    Progress(doneN.toFloat() / orders.size)
                    orders.forEach { r ->
                        CheckRow(r, if (r.outside) "바구니 밖 — 전량 정리" else p.desc[r.ticker], p.fractional,
                            checked = r.ticker in vm.checked, onCheck = { vm.toggle(r.ticker) },
                            onToss = { vm.check(r.ticker); TossLink.openOrder(ctx, r.ticker) })
                    }
                    Text("체크는 이번 ${PERIOD_LABEL[p.period]}까지 기억합니다 · 차이가 20만 원 아래면 건너뜀 · " +
                        (if (p.fractional) "미국은 0.01주" else "정수 주(현금을 넘으면 1주씩 줄임)"),
                        color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 8.dp))
                }
                if (!p.sellOutside && p.outside.isNotEmpty()) {
                    Text("바구니 밖 ${p.outside.size}개(${manwon(p.outsideKrw)})는 그대로 둡니다 — 바구니 › 규칙에서 '정리'로 바꿀 수 있음",
                        color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
                }

                if (orders.isNotEmpty()) {
                    Spacer(Modifier.height(10.dp))
                    InfoBox {
                        Text("매도 ${p.sellCount}건 ${manwon(p.sellKrw)} · 매수 ${p.buyCount}건 ${manwon(p.buyKrw)} · 비용 약 ${manwon(p.costKrw)}",
                            color = TextPrimary, fontSize = 12.sp)
                        if (p.rows.any { it.action == "sell" && !it.krw }) {
                            Text("양도세(추정) — 올해 이익 ${manwon(p.taxYtd)} → ${manwon(p.taxAfter)}, " +
                                if (p.taxThis > 0) "늘어나는 세금 약 ${manwon(p.taxThis)} (내년 5월)" else "공제 250만 안쪽이라 추가 세금 없음",
                                color = if (p.taxThis > 0) Profit else TextSecondary, fontSize = 12.sp)
                        }
                    }
                    val next = orders.firstOrNull { it.ticker !in vm.checked }
                    Spacer(Modifier.height(12.dp))
                    PrimaryButton(
                        if (next != null) "다음: ${Tickers.displayName(next.ticker)} ${if (next.action == "sell") "매도" else "매수"} ${qtyLabel(next.qty)} — 토스에서"
                        else "모두 체크했습니다", enabled = next != null,
                    ) { next?.let { vm.check(it.ticker); TossLink.openOrder(ctx, it.ticker) } }
                }
                if (p.due) {
                    Spacer(Modifier.height(8.dp))
                    GhostButton("다 맞췄어요 — 다음 ${PERIOD_LABEL[p.period]}까지 '맞출 때' 끔", modifier = Modifier.fillMaxWidth()) { vm.done() }
                }
            }
        }
    }
}

@Composable
private fun StatusCard(p: Server.Rebalance, total: Int, done: Int) {
    val due = p.due && total > 0
    val c = if (due) Profit else TextSecondary
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
            .background(if (due) Color(0xFF22161A) else SurfaceInput)
            .border(1.dp, if (due) Profit else Color.Transparent, RoundedCornerShape(12.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.clip(RoundedCornerShape(6.dp)).border(1.dp, c, RoundedCornerShape(6.dp))
                .padding(horizontal = 7.dp, vertical = 2.dp)) {
                Text(if (p.due) "맞출 때" else "D-${p.daysLeft}", color = c, fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.width(8.dp))
            Text(
                when {
                    due -> "${PERIOD_LABEL[p.period]} 끝(${mmdd(p.periodEnd)}) — ${total - done}개 남음"
                    p.due -> "${PERIOD_LABEL[p.period]} 끝 — 모두 목표 근처"
                    else -> "다음 맞추기 ${mmdd(p.nextEnd)}"
                },
                color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold,
            )
        }
        if (!p.due && p.alerts.isNotEmpty()) {
            Text("목표의 2배/절반을 넘은 종목: " + p.alerts.joinToString(", ") { Tickers.displayName(it) },
                color = Profit, fontSize = 12.sp)
        } else if (!p.due) {
            Text("그 사이 목표의 2배/절반을 넘으면 알립니다" + (p.lastDone?.let { " · 마지막으로 맞춘 날 ${mmdd(it)}" } ?: ""),
                color = TextSecondary, fontSize = 12.sp)
        }
    }
}

/** 비중 막대 두 줄 — 지금 / 목표. 회색=바구니, 금색=바구니 밖, 빈칸=현금. */
@Composable
private fun WeightBars(p: Server.Rebalance) {
    val tot = p.accountTotalKrw.takeIf { it > 0 } ?: return
    val consider = p.basketKrw + p.cashKrw + if (p.sellOutside) p.outsideKrw else 0.0
    val now = Triple(p.basketKrw / tot, p.outsideKrw / tot, p.cashKrw / tot)
    val tgt = Triple(consider * (1 - p.cashPct / 100.0) / tot,
        if (p.sellOutside) 0.0 else p.outsideKrw / tot, consider * p.cashPct / 100.0 / tot)
    @Composable
    fun bar(label: String, w: Triple<Double, Double, Double>) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 3.dp)) {
            Text(label, color = TextMuted, fontSize = 11.sp, modifier = Modifier.width(30.dp))
            Row(Modifier.weight(1f).height(14.dp).clip(RoundedCornerShape(4.dp)).background(SurfaceInput)) {
                if (w.first > 0.001) Box(Modifier.weight(w.first.toFloat()).fillMaxHeight().background(Color(0xFF4A5060)))
                if (w.second > 0.001) Box(Modifier.weight(w.second.toFloat()).fillMaxHeight().background(Color(0xFF6B5A30)))
                val rest = (1 - w.first - w.second).coerceAtLeast(0.001)
                Box(Modifier.weight(rest.toFloat()).fillMaxHeight())
            }
        }
    }
    bar("지금", now)
    bar("목표", tgt)
    Text("■ 바구니 %.0f%% → %.0f%% · ■ 바구니 밖 %.0f%% → %.0f%% · □ 현금 %.0f%% → %.0f%%".format(
        now.first * 100, tgt.first * 100, now.second * 100, tgt.second * 100, now.third * 100, tgt.third * 100),
        color = TextMuted, fontSize = 11.sp)
}

@Composable
private fun Progress(f: Float) {
    BoxWithConstraints(Modifier.fillMaxWidth().padding(bottom = 4.dp).height(6.dp).clip(RoundedCornerShape(3.dp)).background(SurfaceInput)) {
        Box(Modifier.width(maxWidth * f.coerceIn(0f, 1f)).fillMaxHeight().background(GREEN))
    }
}

@Composable
private fun CheckRow(r: Server.RebalRow, desc: String?, fractional: Boolean, checked: Boolean,
                     onCheck: () -> Unit, onToss: () -> Unit) {
    val sell = r.action == "sell"
    val c = if (sell) Loss else Profit
    val dim = if (checked) TextMuted else TextPrimary
    Column {
        Row(Modifier.fillMaxWidth().padding(top = 10.dp, bottom = 2.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(22.dp).clip(RoundedCornerShape(6.dp))
                    .background(if (checked) GREEN else Color.Transparent)
                    .border(1.5.dp, if (checked) GREEN else TextMuted, RoundedCornerShape(6.dp))
                    .clickable { onCheck() },
                contentAlignment = Alignment.Center,
            ) { if (checked) Text("✓", color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.Bold) }
            Spacer(Modifier.width(10.dp))
            Text(Tickers.displayName(r.ticker), color = dim, fontSize = 14.sp, fontWeight = FontWeight.Bold,
                fontFamily = Mono, maxLines = 1, modifier = Modifier.width(68.dp))
            Text(if (r.outside) "${manwon(r.valueKrw)} → 0" else "${manwon(r.valueKrw)} → ${manwon(r.targetKrw)}",
                color = TextSecondary, fontSize = 11.sp, modifier = Modifier.weight(1f))
            Column(horizontalAlignment = Alignment.End) {
                Text("${if (sell) "매도" else "매수"} ${if (r.outside) "전량" else qtyLabel(r.qty)}",
                    color = if (checked) TextMuted else c, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                Text(manwon(r.amountKrw), color = TextSecondary, fontSize = 11.sp)
            }
            Spacer(Modifier.width(8.dp))
            GhostButton("토스", color = if (checked) TextMuted else TextPrimary) { onToss() }
        }
        val gap = when {
            r.outside -> "${qtyLabel(r.qty)} 전부"
            abs(r.afterGapKrw) < 10_000 -> "목표와 거의 같음"
            else -> "맞추면 목표보다 ${if (r.afterGapKrw > 0) "+" else ""}${manwon(r.afterGapKrw)}"
        }
        val alt = if (!r.outside && !r.krw && !fractional && r.qtyFrac > 0 && r.qtyFrac != r.qty) " · 소수점이면 ${qtyLabel(r.qtyFrac)}" else ""
        Text(listOf(priceLabel(r), gap + alt).filter { it.isNotBlank() }.joinToString(" · "),
            color = TextSecondary, fontSize = 11.sp, modifier = Modifier.padding(start = 32.dp))
        if (!desc.isNullOrBlank()) {
            Text(desc, color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(start = 32.dp))
        }
        Spacer(Modifier.height(8.dp))
        HDivider()
    }
}

@Composable
private fun InfoBox(content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(SurfaceInput).padding(horizontal = 12.dp, vertical = 11.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) { content() }
}

// ══════════════════════ ② 바구니 ══════════════════════

@Composable
fun BasketScreen(vm: RebalanceViewModel = viewModel()) {
    LaunchedEffect(AppState.dataVersion) { if (vm.plan == null) vm.load() }
    // 과거 성과·종목 숫자는 PC 가 백그라운드로 계산한다 — 아직이면 잠깐씩 다시 묻는다
    LaunchedEffect(vm.plan?.cashTable == null) {
        while (vm.plan != null && vm.plan?.cashTable == null) { delay(4_000); vm.load() }
    }
    var seg by rememberSaveable { mutableStateOf("items") }
    Column(Modifier.fillMaxSize().background(BgApp)) {
        val p = vm.plan
        ScreenHeader("바구니") {
            UnderlineSegments(
                listOf("items" to "종목${p?.let { " ${it.basket.size}" } ?: ""}", "rules" to "규칙", "history" to "과거 성과"),
                selected = seg, onSelect = { seg = it },
            )
        }
        Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())
            .padding(horizontal = ScreenPad).padding(top = 8.dp, bottom = 20.dp)) {
            vm.error?.let { Text("⚠️ $it", color = Loss, fontSize = 12.sp, modifier = Modifier.padding(bottom = 8.dp)) }
            if (p == null) {
                Text("PC 에서 받는 중…", color = TextSecondary, fontSize = 13.sp)
                return@Column
            }
            when (seg) {
                "rules" -> RulesPanel(p, vm)
                "history" -> HistoryPanel(p)
                else -> ItemsPanel(p)
            }
        }
    }
}

@Composable
private fun ItemsPanel(p: Server.Rebalance) {
    Text("동일 비중 · ${PERIOD_SUB[p.period]} · 현금 ${p.cashPct}% · ${if (p.fractional) "소수점" else "정수 주"}",
        color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(bottom = 4.dp))
    val rows = p.rows.associateBy { it.ticker }
    val tot = p.totalKrw.takeIf { it > 0 } ?: 1.0
    p.basket.forEach { t ->
        val r = rows[t]
        Column(Modifier.fillMaxWidth().padding(vertical = 9.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(Tickers.displayName(t), color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.ExtraBold, fontFamily = Mono)
                levOf(p.desc[t])?.let {
                    Spacer(Modifier.width(6.dp))
                    Box(Modifier.clip(RoundedCornerShape(4.dp)).background(Color(0xFF2A2A33)).padding(horizontal = 5.dp, vertical = 1.dp)) {
                        Text("${it}배", color = TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                    }
                }
                Spacer(Modifier.weight(1f))
                if (r != null) {
                    val now = r.valueKrw / tot * 100; val tg = r.targetKrw / tot * 100
                    val c = when { r.action == "none" -> TextSecondary; r.devPct > 0 -> Loss; else -> Profit }
                    Text("지금 %.1f%% / 목표 %.1f%%".format(now, tg), color = c, fontSize = 11.sp, fontFamily = Mono)
                }
            }
            p.desc[t]?.let { Text(it, color = TextSecondary, fontSize = 12.sp, modifier = Modifier.padding(top = 2.dp)) }
            p.metrics[t]?.let { m ->
                Text("최근 1년 변동성 %.0f%% · 수익 %+.0f%%".format(m.vol * 100, m.ret1y * 100),
                    color = TextMuted, fontSize = 11.sp, fontFamily = Mono, modifier = Modifier.padding(top = 2.dp))
            }
            p.notes[t]?.let { Text("⚠️ $it", color = Gold, fontSize = 11.sp, modifier = Modifier.padding(top = 2.dp)) }
            if (r != null && r.held) DevBar(r)
        }
        HDivider()
    }
    if (p.outside.isNotEmpty()) {
        SectionLabel("바구니 밖 보유 ${p.outside.size} · ${manwon(p.outsideKrw)}")
        p.outside.forEach { (t, v) ->
            Row(Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
                Text(Tickers.displayName(t), color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.Bold,
                    fontFamily = Mono, modifier = Modifier.weight(1f))
                Text(manwon(v), color = TextSecondary, fontSize = 13.sp, fontFamily = Mono)
            }
        }
        Text(if (p.sellOutside) "규칙: 정리 — 홈의 맞추기 목록에 '전량 매도'로 들어갑니다"
            else "규칙: 그대로 두기 — 리밸런싱 계산에서 뺍니다", color = TextMuted, fontSize = 11.sp)
    }
}

/** 목표 대비 이탈 막대 — 가운데 0, 오른쪽(파랑)=많음, 왼쪽(빨강)=적음. ±100% 에서 자른다. */
@Composable
private fun DevBar(r: Server.RebalRow) {
    val c = when { r.action == "none" -> TextSecondary; r.devPct > 0 -> Loss; else -> Profit }
    Row(Modifier.fillMaxWidth().height(20.dp).padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        BoxWithConstraints(Modifier.weight(1f).height(8.dp)) {
            val half = maxWidth / 2
            val frac = (abs(r.devPct) / 100).coerceIn(0.0, 1.0).toFloat()
            Box(Modifier.offset(x = half).width(1.dp).fillMaxHeight().background(BorderColor))
            if (r.devPct >= 0) {
                Box(Modifier.offset(x = half).width(half * frac).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(c))
            } else {
                Box(Modifier.offset(x = half - half * frac).width(half * frac).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(c))
            }
        }
        Text("목표 대비 %+.0f%%".format(r.devPct), color = c, fontSize = 10.sp, fontFamily = Mono,
            modifier = Modifier.padding(start = 8.dp))
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

// ══════════════════════ ④ 더보기 ══════════════════════

/** 나머지 — 예전 비교·분석(종목 분석)과 설정. 리밸런싱이 주력이 되면서 한 단계 안쪽으로. */
@Composable
fun MoreScreen(onOpen: (String) -> Unit) {
    Column(Modifier.fillMaxSize().background(BgApp)) {
        ScreenHeader("더보기")
        Column(Modifier.padding(horizontal = ScreenPad)) {
            listOf(
                Triple("compare", "종목 분석", "관심종목 표 · Z·M 차트 · 장기 제안 카드 (예전 비교·분석 탭)"),
                Triple("settings", "설정", "집 PC 연결 · 관심종목 · 원금 기록 · 분석 기간 · 앱 업데이트"),
            ).forEach { (id, title, sub) ->
                Row(Modifier.fillMaxWidth().clickable { onOpen(id) }.padding(vertical = 14.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(title, color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                        Text(sub, color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 2.dp))
                    }
                    Text("›", color = TextMuted, fontSize = 18.sp)
                }
                HDivider()
            }
            Text("검증해 보니 Z·M 으로 사고팔 시점을 맞히는 근거가 약했고, 바구니 + 현금 + 분기 맞추기가 가장 일관됐다. " +
                "그래서 매일 보는 화면을 그 일에 맞췄다.", color = TextMuted, fontSize = 11.sp, lineHeight = 16.sp,
                modifier = Modifier.padding(top = 16.dp))
        }
    }
}
