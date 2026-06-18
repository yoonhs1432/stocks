package com.quant.dashboard.ui

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
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.quant.Quant
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import com.quant.dashboard.ui.theme.pctColor
import com.quant.dashboard.ui.theme.signalColor
import androidx.compose.foundation.background

private val SIGNAL_LABEL = mapOf(
    "strong_buy" to "강한 매수", "buy" to "매수", "hold" to "중립",
    "sell" to "매도", "strong_sell" to "강한 매도",
)

@Composable
fun AnalysisScreen(vm: AnalysisViewModel = viewModel()) {
    val s = vm.state
    var menuOpen by remember { mutableStateOf(false) }

    // 최초 1회 로드
    androidx.compose.runtime.LaunchedEffect(Unit) {
        if (s.result == null && !s.loading) vm.load()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BgApp)
            .verticalScroll(rememberScrollState())
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text("📊 퀀트 대시보드", color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.Bold)

        // 종목 선택 + 새로고침
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { menuOpen = true }) {
                Text("${Tickers.displayName(s.ticker)} ▾")
            }
            DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }) {
                Tickers.DEFAULT.forEach { tk ->
                    DropdownMenuItem(
                        text = { Text(Tickers.displayName(tk)) },
                        onClick = { menuOpen = false; vm.select(tk) },
                    )
                }
            }
            Button(onClick = { vm.refresh() }) { Text("🔄") }
        }

        when {
            s.loading -> Row(
                modifier = Modifier.fillMaxWidth().padding(24.dp),
                horizontalArrangement = Arrangement.Center,
            ) { CircularProgressIndicator() }

            s.error != null -> Text("⚠️ ${s.error}", color = signalColor("strong_sell"))

            s.result != null -> ResultView(s.result)
        }
    }
}

@Composable
private fun ResultView(r: Quant.Result) {
    // 요약 행
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text("$${"%,.2f".format(r.lastPrice)}", color = TextPrimary,
            fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Metric("β·SPY", "%.2f×".format(r.beta))
            Metric("σ", "±%.0f%%".format(r.sigmaPct))
            Metric("Z", "%.0f".format(r.lastZpct), pctColor(r.lastZpct))
            Metric("M", "%.0f".format(r.lastMpct), pctColor(r.lastMpct))
        }
        Text(
            "신호: ${SIGNAL_LABEL[r.signal] ?: r.signal}",
            color = signalColor(r.signal), fontWeight = FontWeight.Bold, fontSize = 15.sp,
        )
    }

    Text("가격 · 회귀선 · ±1.5σ", color = TextSecondary, fontSize = 12.sp)
    PriceChart(r.price, r.predicted, r.bandUpper, r.bandLower, r.tickerNorm)

    Text("Z(흰) · M(주황)  — 20/40/60/80", color = TextSecondary, fontSize = 12.sp)
    ZmChart(r.zPct, r.mPct)

    Text("RSI — 30/70", color = TextSecondary, fontSize = 12.sp)
    RsiChart(r.rsi)
}

@Composable
private fun Metric(label: String, value: String, valueColor: androidx.compose.ui.graphics.Color = TextPrimary) {
    Column {
        Text(label, color = TextSecondary, fontSize = 11.sp)
        Text(value, color = valueColor, fontWeight = FontWeight.Bold, fontSize = 15.sp)
    }
}
