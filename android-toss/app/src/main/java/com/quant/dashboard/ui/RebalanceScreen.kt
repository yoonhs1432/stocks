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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
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
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.abs
import kotlin.math.roundToLong

/*
 * 리밸런싱 도우미 — 바구니를 동일 비중으로 맞추려면 무엇을 얼마나 사고팔지.
 * 계산은 PC 가 한다(web/rebalance.py). 이 앱은 주문을 내지 않고 토스 주문 화면으로 건네준다.
 * 색은 한국식: 매수(목표보다 적음)=빨강, 매도(목표보다 많음)=파랑.
 * 목업: research/mockup_rebalance.png
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

private fun mmdd(iso: String): String = iso.takeIf { it.length >= 10 }?.let {
    "${it.substring(5, 7).toInt()}/${it.substring(8, 10).toInt()}"
} ?: iso

private fun qtyLabel(q: Double): String =
    if (q == kotlin.math.floor(q)) "%,d주".format(q.toLong()) else "%.2f주".format(q)

/** 1주 가격 — 미국은 달러와 원화 둘 다. */
private fun priceLabel(r: Server.RebalRow): String = when {
    r.price.isNaN() -> ""
    r.krw -> "1주 %,d원".format(r.price.roundToLong())
    else -> "1주 $%.2f(%s)".format(r.price, manwon(r.priceKrw))
}

/** 포트폴리오 탭의 진입 줄 — 상태만 보이고 누르면 도우미가 열린다. */
@Composable
fun RebalanceEntry() {
    var plan by remember { mutableStateOf<Server.Rebalance?>(null) }
    var err by remember { mutableStateOf<String?>(null) }
    var open by remember { mutableStateOf(false) }
    LaunchedEffect(AppState.dataVersion) {
        val r = withContext(Dispatchers.IO) { runCatching { Server.rebalance() } }
        r.onSuccess { plan = it; err = null }.onFailure { err = it.message }
    }
    HDivider()
    Row(
        Modifier.fillMaxWidth().clickable { open = true }.padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Column(Modifier.weight(1f)) {
            Text("리밸런싱", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Bold)
            val p = plan
            Text(
                when {
                    p != null -> "동일 비중 ${p.basket.size}종목 · ${PERIOD_SUB[p.period]} · 현금 ${p.cashPct}%"
                    err != null -> "PC 에서 받지 못했습니다"
                    else -> "불러오는 중…"
                },
                color = TextSecondary, fontSize = 11.sp,
            )
        }
        plan?.let { StatusPill(it) }
        Text("›", color = TextMuted, fontSize = 16.sp)
    }
    if (open) {
        RebalanceDialog(plan, onClose = { open = false }, onPlan = { plan = it })
    }
}

@Composable
private fun StatusPill(p: Server.Rebalance) {
    val (txt, c) = if (p.due) "맞출 때" to Profit else "D-${p.daysLeft}" to TextSecondary
    Box(
        Modifier.clip(RoundedCornerShape(6.dp)).border(1.dp, c, RoundedCornerShape(6.dp))
            .padding(horizontal = 7.dp, vertical = 2.dp),
    ) { Text(txt, color = c, fontSize = 11.sp, fontWeight = FontWeight.Bold, maxLines = 1) }
}

/** ① 도우미 본문 — 전체 화면 대화상자. */
@Composable
private fun RebalanceDialog(initial: Server.Rebalance?, onClose: () -> Unit, onPlan: (Server.Rebalance) -> Unit) {
    var plan by remember { mutableStateOf(initial) }
    var err by remember { mutableStateOf<String?>(null) }
    var settings by remember { mutableStateOf(false) }
    var expanded by remember { mutableStateOf(false) }
    val opened = remember { mutableStateListOf<String>() }      // 토스로 넘긴 종목 — 다음 순서로
    val scope = rememberCoroutineScope()
    val ctx = LocalContext.current

    fun apply(r: Result<Server.Rebalance>) {
        r.onSuccess { plan = it; onPlan(it); err = null }.onFailure { err = it.message }
    }
    LaunchedEffect(Unit) {
        apply(withContext(Dispatchers.IO) { runCatching { Server.rebalance() } })
    }

    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize(), color = BgApp) {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = ScreenPad, vertical = 16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("리밸런싱", color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold)
                        plan?.let {
                            Text("동일 비중 · ${PERIOD_SUB[it.period]} · 현금 ${it.cashPct}%",
                                color = TextSecondary, fontSize = 12.sp)
                        }
                    }
                    GhostButton("설정", enabled = plan != null) { settings = true }
                    Spacer(Modifier.width(8.dp))
                    Text("✕", color = TextSecondary, fontSize = 22.sp, modifier = Modifier.clickable { onClose() })
                }
                err?.let { Text("⚠️ $it", color = Loss, fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp)) }
                val p = plan ?: run {
                    Text("PC 에서 계산하는 중…", color = TextSecondary, fontSize = 13.sp, modifier = Modifier.padding(top = 24.dp))
                    return@Column
                }

                // 상태
                Row(Modifier.padding(top = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                    StatusPill(p)
                    Spacer(Modifier.width(8.dp))
                    Text(
                        if (p.due) "${PERIOD_LABEL[p.period]} 말(${mmdd(p.periodEnd)}) 지남 — 맞춘 뒤 아래 '다 맞췄어요'"
                        else "다음 ${mmdd(p.nextEnd)}" + (p.lastDone?.let { " · 마지막으로 맞춘 날 ${mmdd(it)}" } ?: ""),
                        color = TextSecondary, fontSize = 12.sp,
                    )
                }

                // 투자 · 현금
                SectionLabel("투자 · 현금")
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(manwon(p.totalKrw), color = TextPrimary, fontSize = 24.sp, fontWeight = FontWeight.Bold,
                        fontFamily = Mono, modifier = Modifier.weight(1f))
                    Text("현금 %.0f%% · 목표 %d%%".format(p.cashPctNow, p.cashPct), color = TextSecondary, fontSize = 12.sp)
                }
                CashBar(investedPct = 100 - p.cashPctNow, targetInvestPct = 100.0 - p.cashPct)
                Text("회색 = 투자 %.0f%% · 금색 선 = 목표 투자 %d%%".format(100 - p.cashPctNow, 100 - p.cashPct),
                    color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 6.dp))

                // 목표 대비
                SectionLabel("목표 대비 (종목당 목표 ${manwon(p.perTargetKrw)})")
                val big = p.rows.filter { abs(it.devPct) >= 5 }
                val small = p.rows.filter { abs(it.devPct) < 5 }
                (if (expanded) p.rows else big).forEach { DevBar(it) }
                Row(Modifier.fillMaxWidth().padding(top = 2.dp)) {
                    Spacer(Modifier.width(66.dp))
                    Row(Modifier.weight(1f), horizontalArrangement = Arrangement.SpaceBetween) {
                        listOf("−100%", "0", "+100%").forEach { Text(it, color = TextMuted, fontSize = 10.sp) }
                    }
                    Spacer(Modifier.width(52.dp))
                }
                if (small.isNotEmpty()) {
                    Text(if (expanded) "접기 ‹" else "나머지 ${small.size}개는 ±5% 안 · 펼치기 ›",
                        color = TextMuted, fontSize = 11.sp,
                        modifier = Modifier.padding(top = 6.dp).clickable { expanded = !expanded })
                }
                Text("파랑 = 목표보다 많음(매도) · 빨강 = 적음(매수)", color = TextMuted, fontSize = 11.sp,
                    modifier = Modifier.padding(top = 4.dp))

                // 주문 목록 — 매도 먼저(현금을 먼저 만든다)
                val orders = p.rows.filter { it.action == "sell" }.sortedByDescending { it.amountKrw } +
                    p.rows.filter { it.action == "buy" }.sortedByDescending { it.amountKrw }
                SectionLabel("이렇게 맞추면 — 매도 먼저")
                if (orders.isEmpty()) {
                    Text("맞출 것이 없습니다 — 모든 종목이 목표에서 20만 원 안쪽", color = TextSecondary, fontSize = 13.sp)
                }
                orders.forEach { r ->
                    OrderRow(r, p.desc[r.ticker], p.fractional, done = r.ticker in opened) { opened.add(r.ticker); TossLink.openOrder(ctx, r.ticker) }
                }
                Text("차이가 20만 원 아래인 종목은 건너뜀 · " +
                    (if (p.fractional) "미국은 소수점(0.01주)" else "정수 주로 목표에 가장 가깝게(현금을 넘으면 1주씩 줄임)") +
                    " · 국내는 1주 단위",
                    color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 8.dp))
                if (p.outside.isNotEmpty()) {
                    Text("바구니 밖 보유 ${p.outside.size}개(${manwon(p.outside.sumOf { it.second })})는 계산에서 뺐습니다 — " +
                        p.outside.joinToString(", ") { Tickers.displayName(it.first) },
                        color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
                }

                // 합계
                SectionLabel("합계")
                InfoBox {
                    Text("매도 ${p.sellCount}건 ${manwon(p.sellKrw)} · 매수 ${p.buyCount}건 ${manwon(p.buyKrw)} · 예상 비용 약 ${manwon(p.costKrw)}",
                        color = TextPrimary, fontSize = 12.sp)
                    Text("다 맞추면 현금 %.0f%% → %d%%".format(p.cashPctNow, p.cashPct), color = TextSecondary, fontSize = 12.sp)
                }
                if (p.rows.any { it.action == "sell" && !it.krw }) {
                    Spacer(Modifier.height(8.dp))
                    InfoBox {
                        Text("해외주식 양도세 (추정)", color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                        Text("올해 실현 이익 ${manwon(p.taxYtd)} → 이번 매도 반영 시 ${manwon(p.taxAfter)}",
                            color = TextSecondary, fontSize = 12.sp)
                        Text(
                            if (p.taxThis > 0) "공제 250만 초과분 × 22% ≈ ${manwon(p.taxThis)} (내년 5월)"
                            else "공제 250만 안쪽이라 이번 매도로 늘어나는 세금 없음",
                            color = if (p.taxThis > 0) Profit else TextSecondary, fontSize = 12.sp,
                        )
                    }
                }

                // 다음 주문을 순서대로 토스로
                val next = orders.firstOrNull { it.ticker !in opened }
                Spacer(Modifier.height(16.dp))
                PrimaryButton(
                    if (next != null) "토스에서 주문 (${orders.indexOf(next) + 1}/${orders.size}) · ${Tickers.displayName(next.ticker)}"
                    else "주문 화면을 모두 열었습니다",
                    enabled = next != null,
                ) {
                    next?.let { opened.add(it.ticker); TossLink.openOrder(ctx, it.ticker) }
                }
                Spacer(Modifier.height(8.dp))
                GhostButton("다 맞췄어요 — 다음 ${PERIOD_LABEL[p.period]}까지 알림 끔", modifier = Modifier.fillMaxWidth()) {
                    scope.launch { apply(withContext(Dispatchers.IO) { runCatching { Server.rebalanceDone() } }) }
                }
            }
        }
    }
    if (settings) {
        plan?.let { p ->
            RebalanceSettings(p, onDismiss = { settings = false }) { basket, cash, period, frac ->
                scope.launch {
                    apply(withContext(Dispatchers.IO) { runCatching { Server.saveRebalance(basket, cash, period, frac) } })
                    settings = false
                }
            }
        }
    }
}

@Composable
private fun CashBar(investedPct: Double, targetInvestPct: Double) {
    BoxWithConstraints(Modifier.fillMaxWidth().padding(top = 8.dp).height(16.dp)) {
        val w = maxWidth
        Box(Modifier.fillMaxWidth().height(10.dp).offset(y = 3.dp).clip(RoundedCornerShape(5.dp)).background(SurfaceInput)) {
            Box(Modifier.fillMaxHeight().width(w * (investedPct / 100).coerceIn(0.0, 1.0).toFloat()).background(Color(0xFF3A3F4B)))
        }
        Box(Modifier.offset(x = w * (targetInvestPct / 100).coerceIn(0.0, 1.0).toFloat()).width(2.dp).height(16.dp).background(Gold))
    }
}

/** 목표 대비 이탈 막대 — 가운데 0, 오른쪽(파랑)=많음, 왼쪽(빨강)=적음. ±100% 에서 자른다. */
@Composable
private fun DevBar(r: Server.RebalRow) {
    val c = when {
        r.action == "none" -> TextSecondary
        r.devPct > 0 -> Loss
        else -> Profit
    }
    Row(Modifier.fillMaxWidth().height(24.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(Tickers.displayName(r.ticker), color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold,
            fontFamily = Mono, maxLines = 1, modifier = Modifier.width(66.dp))
        BoxWithConstraints(Modifier.weight(1f).height(12.dp)) {
            val half = maxWidth / 2
            val frac = (abs(r.devPct) / 100).coerceIn(0.0, 1.0).toFloat()
            Box(Modifier.offset(x = half).width(1.dp).fillMaxHeight().background(BorderColor))
            if (r.devPct >= 0) {
                Box(Modifier.offset(x = half).width(half * frac).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(c))
            } else {
                Box(Modifier.offset(x = half - half * frac).width(half * frac).fillMaxHeight().clip(RoundedCornerShape(3.dp)).background(c))
            }
        }
        Text(if (!r.held) "새로" else "%+.0f%%".format(r.devPct), color = c, fontSize = 11.sp,
            fontFamily = Mono, maxLines = 1, modifier = Modifier.width(52.dp).padding(start = 6.dp))
    }
}

@Composable
private fun OrderRow(r: Server.RebalRow, desc: String?, fractional: Boolean, done: Boolean, onToss: () -> Unit) {
    val sell = r.action == "sell"
    val c = if (sell) Loss else Profit
    Column {
        Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(Tickers.displayName(r.ticker), color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold,
                fontFamily = Mono, maxLines = 1, modifier = Modifier.width(78.dp))
            Text("${manwon(r.valueKrw)} → ${manwon(r.targetKrw)}", color = TextSecondary, fontSize = 11.sp,
                modifier = Modifier.weight(1f))
            Column(horizontalAlignment = Alignment.End) {
                Text("${if (sell) "매도" else "매수"} ${qtyLabel(r.qty)}", color = c, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                Text(manwon(r.amountKrw), color = TextSecondary, fontSize = 11.sp)
            }
            Spacer(Modifier.width(8.dp))
            GhostButton(if (done) "✓ 토스" else "토스", color = if (done) TextMuted else TextPrimary) { onToss() }
        }
        // 1주 가격 · 맞춘 뒤 목표와의 차이 · (정수 주일 때) 소수점이면 몇 주
        val gap = if (abs(r.afterGapKrw) < 10_000) "목표와 거의 같음"
            else "맞추면 목표보다 ${if (r.afterGapKrw > 0) "+" else ""}${manwon(r.afterGapKrw)}"
        val alt = if (!r.krw && !fractional && r.qtyFrac > 0 && r.qtyFrac != r.qty) " · 소수점이면 ${qtyLabel(r.qtyFrac)}" else ""
        Text("${priceLabel(r)} · $gap$alt", color = TextSecondary, fontSize = 11.sp,
            modifier = Modifier.padding(bottom = if (desc.isNullOrBlank()) 8.dp else 2.dp))
        if (!desc.isNullOrBlank()) {
            Text(desc, color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(bottom = 8.dp))
        }
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

/** ② 설정 — 주기·현금 비중·바구니. 현금 비중을 고를 때 그 바구니의 과거 결과를 같이 보여 준다. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RebalanceSettings(p: Server.Rebalance, onDismiss: () -> Unit,
                              onSave: (List<String>, Int, String, Boolean) -> Unit) {
    var period by remember { mutableStateOf(p.period) }
    var fractional by remember { mutableStateOf(p.fractional) }
    var cash by remember { mutableFloatStateOf(p.cashPct.toFloat()) }
    val basket = remember { mutableStateListOf<String>().also { it.addAll(p.basket) } }
    var plan by remember { mutableStateOf(p) }

    // 표가 아직 계산 중이면(바구니를 바꾼 직후 등) 잠깐씩 다시 묻는다
    LaunchedEffect(plan.cashTable == null) {
        while (plan.cashTable == null) {
            delay(4_000)
            withContext(Dispatchers.IO) { runCatching { Server.rebalance() } }.onSuccess { plan = it }
        }
    }

    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize(), color = BgApp) {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = ScreenPad, vertical = 16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("리밸런싱 설정", color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold,
                        modifier = Modifier.weight(1f))
                    Text("✕", color = TextSecondary, fontSize = 22.sp, modifier = Modifier.clickable { onDismiss() })
                }

                SectionLabel("언제 맞출까")
                val best = plan.periodTable?.maxByOrNull { it.cagr }?.key ?: "quarterly"
                UnderlineSegments(
                    listOf("monthly", "quarterly", "yearly").map { it to (PERIOD_LABEL[it]!! + if (it == best) " (추천)" else "") },
                    selected = period, onSelect = { period = it },
                )
                plan.periodTable?.let { t ->
                    Text("과거(${plan.tableFrom.take(4)}~) 세후: " + t.joinToString(" · ") {
                        "${PERIOD_LABEL[it.key]} %+.1f%%".format(it.cagr * 100)
                    }, color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(top = 6.dp))
                }

                SectionLabel("미국 종목 주문 단위")
                UnderlineSegments(listOf("whole" to "정수 주", "frac" to "소수점(0.01주)"),
                    selected = if (fractional) "frac" else "whole", onSelect = { fractional = it == "frac" })
                Text("정수 주 = 목표에 가장 가까운 주식 수(비싼 종목은 목표와 조금 차이). 소수점 = 목표 금액에 거의 딱 맞음 — " +
                    "토스에서 그 종목의 소수점 주문이 될 때만.", color = TextMuted, fontSize = 11.sp, lineHeight = 15.sp,
                    modifier = Modifier.padding(top = 6.dp))

                SectionLabel("현금으로 둘 비중")
                Row(verticalAlignment = Alignment.Bottom) {
                    Text("${cash.toInt()}%", color = Gold, fontSize = 24.sp, fontWeight = FontWeight.Bold,
                        fontFamily = Mono, modifier = Modifier.weight(1f))
                    Text("이자 연 2.5% 가정", color = TextSecondary, fontSize = 12.sp)
                }
                Slider(
                    value = cash, onValueChange = { cash = it },     // steps=4 → 0·10·…·50 에 붙는다
                    valueRange = 0f..50f, steps = 4,
                    colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Gold,
                        inactiveTrackColor = SurfaceInput, activeTickColor = Gold, inactiveTickColor = TextMuted),
                )
                val table = plan.cashTable
                if (table == null) {
                    Text("과거 성과를 PC 가 계산하는 중… (10초쯤)", color = TextSecondary, fontSize = 12.sp)
                } else {
                    CashChart(table, cash.toInt())
                    Row(Modifier.fillMaxWidth().padding(top = 6.dp)) {
                        listOf("현금", "세후 연수익", "최대 낙폭", "1억 최악일 때").forEachIndexed { i, h ->
                            Text(h, color = TextMuted, fontSize = 11.sp, modifier = Modifier.weight(if (i == 0) 0.7f else 1f))
                        }
                    }
                    table.forEach { r ->
                        val sel = r.key.toIntOrNull() == cash.toInt()
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
                    Text("이 바구니 ${plan.basket.size}종목 · ${plan.tableFrom.take(4)}년~ 백테스트(${PERIOD_LABEL[plan.period]}, 세후). " +
                        "지금 고른 종목이라 절대 수익은 과장돼 있다 — 현금 비중에 따라 무엇을 얼마나 잃는지의 비교로만 볼 것.",
                        color = TextMuted, fontSize = 10.sp, lineHeight = 14.sp, modifier = Modifier.padding(top = 8.dp))
                }

                SectionLabel("바구니 (동일 비중으로 담을 종목)")
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    (plan.watchlist + basket.filter { it !in plan.watchlist }).forEach { t ->
                        val on = t in basket
                        Box(
                            Modifier.clip(RoundedCornerShape(8.dp)).background(if (on) Gold.copy(alpha = 0.18f) else SurfaceInput)
                                .border(BorderStroke(1.dp, if (on) Gold else Color.Transparent), RoundedCornerShape(8.dp))
                                .clickable { if (on) basket.remove(t) else basket.add(t) }
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                        ) { Text(Tickers.displayName(t), color = if (on) Gold else TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.Bold) }
                    }
                }
                // 고른 종목이 무엇인지 — 설명은 서버가 준다
                Column(Modifier.padding(top = 10.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    basket.forEach { t ->
                        Row {
                            Text(Tickers.displayName(t), color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                                fontFamily = Mono, modifier = Modifier.width(64.dp))
                            Text(plan.desc[t] ?: "", color = TextSecondary, fontSize = 12.sp, lineHeight = 16.sp)
                        }
                    }
                }
                Text("기본은 단일 종목 ETF 를 뺀 여러 종목 레버리지 ETF 의 상관 묶음 대표. 배수는 기초자산이 감당하는 만큼 — " +
                    "S&P 만 3배, 나머지는 2배가 있으면 2배. 급락장에선 레버리지 주식끼리 같이 빠져, 낙폭을 줄이는 건 현금이다.",
                    color = TextMuted, fontSize = 11.sp, lineHeight = 15.sp, modifier = Modifier.padding(top = 8.dp))
                Row(Modifier.padding(top = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (plan.defaultBasket.isNotEmpty()) {
                        GhostButton("기본 ${plan.defaultBasket.size}개") { basket.clear(); basket.addAll(plan.defaultBasket) }
                    }
                    GhostButton("관심종목 전체") { basket.clear(); basket.addAll(plan.watchlist) }
                }

                SectionLabel("알림")
                InfoBox {
                    Text("${PERIOD_LABEL[period]} 말 다음 날 아침에 알림 · 그 사이 보유 종목이 목표의 2배/절반을 넘으면 한 번 더",
                        color = TextSecondary, fontSize = 12.sp)
                }
                Spacer(Modifier.height(16.dp))
                PrimaryButton("저장", enabled = basket.isNotEmpty()) { onSave(basket.toList(), cash.toInt(), period, fractional) }
            }
        }
    }
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
        Text("80%", color = TextMuted, fontSize = 9.sp, fontFamily = Mono, modifier = Modifier.offset(y = 0.dp))
        Text("0%", color = TextMuted, fontSize = 9.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.BottomStart).offset(y = (-18).dp))
        Text("Max drawdown", color = Loss, fontSize = 10.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.TopEnd).offset(y = 18.dp))
        Text("CAGR after tax", color = Profit, fontSize = 10.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.CenterEnd).offset(y = 30.dp))
        Text("Cash weight 0 → 50%", color = TextMuted, fontSize = 10.sp, fontFamily = Mono, modifier = Modifier.align(Alignment.BottomCenter))
    }
}
