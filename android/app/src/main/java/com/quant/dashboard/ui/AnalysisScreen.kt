package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.ui.draw.clip
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
import com.quant.dashboard.ui.theme.Neutral
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
    var periodMonths by remember { mutableStateOf(2) }

    LaunchedEffect(Unit) {
        if (s.result == null && !s.loading) vm.load()
        if (vm.overview.isEmpty()) vm.loadOverview()
    }

    val ov = vm.overview.associateBy { it.ticker }
    val tickers = if (vm.overview.isNotEmpty())
        Store.loadTickers().sortedBy { ov[it]?.mPct ?: 50.0 } else Store.loadTickers()

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(8.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            // ── 좌측: 종목 버튼 세로 리스트 ──
            Column(
                Modifier.weight(0.30f),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                tickers.forEach { tk ->
                    val row = ov[tk]
                    val bg = if (row != null) pctColor(row.mPct) else Neutral
                    val dark = row != null && (row.mPct < 20 || row.mPct >= 80)
                    val dayStr = row?.let { (if (it.day >= 0) "+" else "") + "%.1f%%".format(it.day) } ?: ""
                    val selected = tk == s.ticker
                    Box(
                        Modifier.fillMaxWidth()
                            .clip(RoundedCornerShape(5.dp))
                            .background(bg)
                            .then(if (selected) Modifier.border(1.5.dp, Color.White, RoundedCornerShape(5.dp)) else Modifier)
                            .clickable { vm.select(tk) }
                            .padding(horizontal = 6.dp, vertical = 5.dp),
                    ) {
                        Text(
                            (if (row?.holding == true) "★" else "") + Tickers.displayName(tk) + "  " + dayStr,
                            color = if (dark) Color.White else Color.Black,
                            fontSize = 11.sp, maxLines = 1,
                            fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium,
                        )
                    }
                }
                Box(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(5.dp))
                        .background(Color(0xFF21262D)).clickable { vm.refresh() }
                        .padding(vertical = 5.dp),
                    contentAlignment = Alignment.Center,
                ) { Text("🔄", fontSize = 12.sp) }
            }

            // ── 우측: 콘텐츠 ──
            Column(
                Modifier.weight(0.70f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    PERIODS.forEach { (label, m) ->
                        FilterChip(selected = periodMonths == m, onClick = { periodMonths = m },
                            label = { Text(label, fontSize = 11.sp) })
                    }
                }
                when {
                    s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                        CircularProgressIndicator()
                    }
                    s.error != null -> Text("⚠️ ${s.error}", color = signalColor("strong_sell"))
                    s.result != null -> ResultView(s.result, periodMonths, s.ticker, s.ohlc)
                }
                TradeSection(s.ticker)
            }
        }
    }
}

@Composable
private fun ResultView(r: Quant.Result, periodMonths: Int, ticker: String, ohlc: List<com.quant.dashboard.data.Candle>) {
    // 종목 헤더: 종목명 + σ·β·Z·M
    Text(Tickers.displayName(ticker), color = pctColor(r.lastMpct), fontSize = 18.sp, fontWeight = FontWeight.Bold)
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Metric("σ", "±%.0f%%".format(r.sigmaPct))
        Metric("β·SPY", "%.1f×".format(r.beta))
        Metric("Z", "%.0f".format(r.lastZpct), pctColor(r.lastZpct))
        Metric("M", "%.0f".format(r.lastMpct), pctColor(r.lastMpct))
    }

    // 정보 카드: 현재가 / 평균단가 / 보유수량
    val pos = remember(ticker) { com.quant.dashboard.quant.Portfolio.position(Store.loadTrades()[ticker].orEmpty()) }
    Row(
        Modifier.fillMaxWidth()
            .border(1.5.dp, if (pos != null) Color(0xFF3FB950) else Color(0xFF6B7280), RoundedCornerShape(8.dp))
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Metric("현재가", Tickers.priceLabel(ticker, r.lastPrice))
        if (pos != null) {
            Metric("평균단가", Tickers.priceLabel(ticker, pos.avg))
            Metric("보유", "${pos.qty}주")
        }
    }
    Text("신호: ${SIGNAL_LABEL[r.signal] ?: r.signal}",
        color = signalColor(r.signal), fontWeight = FontWeight.Bold, fontSize = 14.sp)

    val n = r.dates.size
    val cutoff = r.dates[n - 1] - periodMonths.toLong() * 30 * 86400
    var start = 0
    while (start < n - 2 && r.dates[start] < cutoff) start++
    val base = r.price[0]
    fun seg(a: DoubleArray) = a.copyOfRange(start, n)
    fun segDollar(a: DoubleArray) = DoubleArray(n - start) { a[start + it] * base }
    val dates = r.dates.copyOfRange(start, n)

    // 매매 마커 — 윈도우 인덱스 매핑
    val trades = remember(ticker, n) { Store.loadTrades()[ticker].orEmpty() }
    val priceMarks = ArrayList<Mark>()
    val zmMarks = ArrayList<Mark>()
    val scatterIdx = ArrayList<Pair<Int, Boolean>>()
    for (t in trades) {
        val sec = try { LocalDate.parse(t.date).toEpochDay() * 86400 } catch (e: Exception) { continue }
        val buy = t.type == "buy"
        // 전체 인덱스 (산점도용)
        var full = -1
        for (i in r.dates.indices) { if (r.dates[i] <= sec) full = i else break }
        if (full >= 0) scatterIdx.add(full to buy)
        // 윈도우 인덱스 (라인 차트용)
        if (full >= start) {
            val wi = full - start
            priceMarks.add(Mark(wi, t.price, buy))
            val mv = seg(r.mPct)[wi]
            if (!mv.isNaN()) zmMarks.add(Mark(wi, mv, buy))
        }
    }

    // ① 회귀 산점도 (로그-로그, Turbo + 밴드 + ★) — 모두 정규화(Norm) 값
    Text("회귀 산점도 (SPY 대비)", color = TextSecondary, fontSize = 11.sp)
    RegressionScatter(seg(r.spyNorm), seg(r.tickerNorm), seg(r.predicted),
        seg(r.bandUpper), seg(r.bandLower), r.beta)

    // ② Z·M 시간 궤적 산점도
    Text("Z·M 궤적 (시간 파랑→빨강, ● 현재)", color = TextSecondary, fontSize = 11.sp)
    ZmScatter(r.zPct, r.mPct, scatterIdx)

    // ③ 가격 캔들
    val candleByDay = remember(ohlc) { ohlc.associateBy { it.t / 86400L } }
    val wN = n - start
    val opens = DoubleArray(wN); val highs = DoubleArray(wN)
    val lows = DoubleArray(wN); val closes = DoubleArray(wN)
    for (i in 0 until wN) {
        val cd = candleByDay[dates[i] / 86400L]
        if (cd != null) { opens[i] = cd.open; highs[i] = cd.high; lows[i] = cd.low; closes[i] = cd.close }
        else { opens[i] = Double.NaN; highs[i] = Double.NaN; lows[i] = Double.NaN; closes[i] = Double.NaN }
    }
    val priceLbl = Tickers.priceLabel(ticker, r.lastPrice)
    if (closes.any { !it.isNaN() }) {
        CandleChart(opens, highs, lows, closes,
            segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
            priceMarks, Tickers.currencySymbol(ticker), priceLbl)
    } else {
        PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
            priceMarks, Tickers.currencySymbol(ticker))
    }

    // ④ Z+M 라인
    ZmChart(seg(r.zPct), seg(r.mPct), zmMarks,
        topLabel = "Z ${"%.0f".format(r.lastZpct)} · M ${"%.0f".format(r.lastMpct)}")

    // ⑤ MACD
    val macdW = seg(r.macd); val sigW = seg(r.macdSignal)
    val macdLast = macdW.lastOrNull { !it.isNaN() } ?: 0.0
    val sigLast = sigW.lastOrNull { !it.isNaN() } ?: 0.0
    MacdChart(macdW, sigW, topLabel = "${"%.2f".format(sigLast)} (${"%+.2f".format(macdLast - sigLast)})")

    // ⑥ RSI
    RsiChart(seg(r.rsi), topLabel = "RSI ${"%.1f".format(r.rsi.lastOrNull { !it.isNaN() } ?: 50.0)}")

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
    var expanded by remember { mutableStateOf(false) }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            "📝 매매 기록 — ${Tickers.displayName(ticker)}  ${if (expanded) "▲" else "▼"} (${trades.size})",
            color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold,
            modifier = Modifier.fillMaxWidth().clickable { expanded = !expanded },
        )

        if (expanded) {
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
}
