package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.quant.Portfolio
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgCard
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Neutral
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private fun pc(v: Double) = if (v > 0) Profit else if (v < 0) Loss else Neutral
private fun money(v: Double) = (if (v >= 0) "+$" else "-$") + "%,.0f".format(kotlin.math.abs(v))

@Composable
fun PortfolioScreen(vm: PortfolioViewModel = viewModel()) {
    val s = vm.state
    LaunchedEffect(Unit) { if (s.result == null && !s.empty && !s.loading) vm.load() }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("💼 포트폴리오", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Bold)
            Button(onClick = { vm.load() }) { Text("🔄") }
        }

        when {
            s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                CircularProgressIndicator()
            }
            s.empty -> Text(
                "매매 기록이 없습니다.\n분석 탭에서 종목을 보고 ‘매매 기록 추가’로 입력하세요.",
                color = TextSecondary, fontSize = 14.sp,
            )
            s.result != null -> ResultBody(s.result)
        }
    }
}

@Composable
private fun ResultBody(r: Portfolio.Result) {
    val curVal = r.seed + r.totalPnl
    val retPct = if (r.seed > 0) r.totalPnl / r.seed * 100 else 0.0

    // 손익 종합 카드
    Column(
        Modifier.fillMaxWidth().background(BgCard).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("평가자산 $${"%,.0f".format(curVal)}", color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.Bold)
        Text("${money(r.totalPnl)}  (${if (retPct >= 0) "+" else ""}${"%.2f".format(retPct)}%)",
            color = pc(r.totalPnl), fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
        Text("고점대비 ${"%.1f".format(r.currentDd)}%  ·  MDD ${"%.1f".format(r.mdd)}%" +
            (r.mddDate?.let { "  (${SimpleDateFormat("yy/MM/dd", Locale.US).format(Date(it * 1000L))})" } ?: ""),
            color = TextSecondary, fontSize = 12.sp)
    }

    if (r.equity.size >= 2) {
        Text("자산 추이 (누적손익 $)", color = TextSecondary, fontSize = 12.sp)
        EquityChart(r.equity.map { it.second }.toDoubleArray())
    }

    if (r.holdings.isNotEmpty()) {
        Text("보유 종목", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        r.holdings.forEach { h ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("${h.name} ${h.qty}주", color = TextPrimary, fontSize = 13.sp)
                Text("$${"%,.0f".format(h.eval)}  ${money(h.pnl)} (${if (h.retPct >= 0) "+" else ""}${"%.1f".format(h.retPct)}%)",
                    color = pc(h.pnl), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }

    if (r.realized.isNotEmpty()) {
        Text("실현손익", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        r.realized.forEach { rz ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(rz.name, color = TextPrimary, fontSize = 13.sp)
                Text(money(rz.realized), color = pc(rz.realized), fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            }
        }
    }

    // ── 사이클 통계 (완료된 사이클) ──
    val statsList = Store.loadTrades().mapNotNull { (tk, list) ->
        Portfolio.cycleStats(list)?.let { Tickers.displayName(tk) to it }
    }
    if (statsList.isNotEmpty()) {
        Text("사이클 통계 (완료 사이클)", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        statsList.forEach { (name, st) ->
            val pf = st.profitFactor?.let { "%.2f".format(it) } ?: "∞"
            Column {
                Text(name, color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                Text(
                    "${st.count}회 · 승률 ${st.winRate.toInt()}% · PF $pf · 평균 ${if (st.avgRet >= 0) "+" else ""}${"%.1f".format(st.avgRet)}% · ${st.avgHoldDays.toInt()}일",
                    color = TextSecondary, fontSize = 12.sp,
                )
            }
        }
    }
}
