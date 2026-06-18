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
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.Trade
import com.quant.dashboard.quant.Quant
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import com.quant.dashboard.ui.theme.pctColor
import com.quant.dashboard.ui.theme.signalColor
import java.time.LocalDate

val SIGNAL_LABEL = mapOf(
    "strong_buy" to "강한 매수", "buy" to "매수", "hold" to "중립",
    "sell" to "매도", "strong_sell" to "강한 매도",
)

private val PERIODS = listOf("1개월" to 1, "2개월" to 2, "4개월" to 4, "1년" to 12)

@Composable
fun AnalysisScreen(vm: AnalysisViewModel = viewModel()) {
    val s = vm.state
    var menuOpen by remember { mutableStateOf(false) }
    var periodMonths by remember { mutableStateOf(2) }

    LaunchedEffect(Unit) { if (s.result == null && !s.loading) vm.load() }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { menuOpen = true }) { Text("${Tickers.displayName(s.ticker)} ▾") }
            DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                Store.loadTickers().forEach { tk ->
                    DropdownMenuItem(text = { Text(Tickers.displayName(tk)) },
                        onClick = { menuOpen = false; vm.select(tk) })
                }
            }
            Button(onClick = { vm.refresh() }) { Text("🔄") }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            PERIODS.forEach { (label, m) ->
                FilterChip(selected = periodMonths == m, onClick = { periodMonths = m },
                    label = { Text(label, fontSize = 12.sp) })
            }
        }

        when {
            s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                CircularProgressIndicator()
            }
            s.error != null -> Text("⚠️ ${s.error}", color = signalColor("strong_sell"))
            s.result != null -> ResultView(s.result, periodMonths)
        }

        TradeSection(s.ticker)
    }
}

@Composable
private fun ResultView(r: Quant.Result, periodMonths: Int) {
    Text("$${"%,.2f".format(r.lastPrice)}", color = TextPrimary, fontSize = 22.sp, fontWeight = FontWeight.Bold)
    Row(horizontalArrangement = Arrangement.spacedBy(14.dp)) {
        Metric("β·SPY", "%.2f×".format(r.beta))
        Metric("σ", "±%.0f%%".format(r.sigmaPct))
        Metric("Z", "%.0f".format(r.lastZpct), pctColor(r.lastZpct))
        Metric("M", "%.0f".format(r.lastMpct), pctColor(r.lastMpct))
    }
    Text("신호: ${SIGNAL_LABEL[r.signal] ?: r.signal}",
        color = signalColor(r.signal), fontWeight = FontWeight.Bold, fontSize = 15.sp)

    val n = r.dates.size
    val cutoff = r.dates[n - 1] - periodMonths.toLong() * 30 * 86400
    var start = 0
    while (start < n - 2 && r.dates[start] < cutoff) start++
    val base = r.price[0]
    fun seg(a: DoubleArray) = a.copyOfRange(start, n)
    fun segDollar(a: DoubleArray) = DoubleArray(n - start) { a[start + it] * base }
    val dates = r.dates.copyOfRange(start, n)

    Text("가격 · 회귀선 · ±1.5σ", color = TextSecondary, fontSize = 12.sp)
    PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower))

    Text("Z(흰) · M(주황) — 20/40/60/80", color = TextSecondary, fontSize = 12.sp)
    ZmChart(seg(r.zPct), seg(r.mPct))

    Text("RSI — 30/70", color = TextSecondary, fontSize = 12.sp)
    RsiChart(seg(r.rsi))

    DateAxis(dates)
}

@Composable
fun Metric(label: String, value: String, valueColor: Color = TextPrimary) {
    Column {
        Text(label, color = TextSecondary, fontSize = 11.sp)
        Text(value, color = valueColor, fontWeight = FontWeight.Bold, fontSize = 15.sp)
    }
}

@Composable
private fun TradeSection(ticker: String) {
    var refresh by remember { mutableStateOf(0) }
    val trades = remember(ticker, refresh) { Store.loadTrades()[ticker].orEmpty() }
    var type by remember { mutableStateOf("buy") }
    var qty by remember { mutableStateOf("") }
    var price by remember { mutableStateOf("") }
    var date by remember { mutableStateOf(LocalDate.now().toString()) }
    var memo by remember { mutableStateOf("") }
    var err by remember { mutableStateOf<String?>(null) }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("📝 매매 기록 — ${Tickers.displayName(ticker)}",
            color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold)

        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            FilterChip(selected = type == "buy", onClick = { type = "buy" },
                label = { Text("매수") })
            FilterChip(selected = type == "sell", onClick = { type = "sell" },
                label = { Text("매도") })
        }
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            OutlinedTextField(date, { date = it }, label = { Text("날짜") },
                singleLine = true, modifier = Modifier.weight(1.4f))
            OutlinedTextField(qty, { qty = it }, label = { Text("수량") },
                singleLine = true, modifier = Modifier.weight(1f))
            OutlinedTextField(price, { price = it }, label = { Text("단가$") },
                singleLine = true, modifier = Modifier.weight(1f))
        }
        OutlinedTextField(memo, { memo = it }, label = { Text("메모 (선택)") },
            singleLine = true, modifier = Modifier.fillMaxWidth())
        Button(onClick = {
            val q = qty.toIntOrNull()
            val p = price.toDoubleOrNull()
            if (q == null || q <= 0 || p == null || p <= 0) {
                err = "수량·단가를 올바르게 입력하세요"
            } else {
                Store.addTrade(ticker, Trade(date.trim(), type, q, p, memo.ifBlank { null }))
                qty = ""; price = ""; memo = ""; err = null; refresh++
            }
        }) { Text("저장") }
        err?.let { Text(it, color = Loss, fontSize = 12.sp) }

        trades.forEachIndexed { i, t ->
            Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
                Text("${t.date}  ${if (t.type == "buy") "▲" else "▼"} ${t.qty}주 @$${"%.2f".format(t.price)}",
                    color = TextSecondary, fontSize = 12.sp)
                TextButton(onClick = { Store.deleteTrade(ticker, i); refresh++ }) {
                    Text("삭제", color = Loss, fontSize = 12.sp)
                }
            }
        }
    }
}
