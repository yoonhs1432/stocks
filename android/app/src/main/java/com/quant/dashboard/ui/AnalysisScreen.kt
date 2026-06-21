package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
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
import com.quant.dashboard.data.Candle
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.Trade
import com.quant.dashboard.quant.Portfolio
import com.quant.dashboard.quant.Quant
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgCard
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.HoldingBg
import com.quant.dashboard.ui.theme.HoldingBorder
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Neutral
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.SegmentOn
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.Teal
import com.quant.dashboard.ui.theme.TextMuted
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import com.quant.dashboard.ui.theme.mHeat
import com.quant.dashboard.ui.theme.pctColor
import com.quant.dashboard.ui.theme.signalColor
import java.time.LocalDate

@Composable
fun AnalysisScreen(vm: AnalysisViewModel = viewModel()) {
    val s = vm.state
    var filter by remember { mutableStateOf("전체") }   // 전체 / ETF / 개별

    LaunchedEffect(AppState.dataVersion) { vm.sync(AppState.dataVersion) }

    val ov = vm.overview.associateBy { it.ticker }
    val allTickers = if (vm.overview.isNotEmpty())
        Store.loadTickers().sortedBy { ov[it]?.mPct ?: 50.0 } else Store.loadTickers()
    val tickers = allTickers.filter {
        when (filter) {
            "ETF" -> !Store.isIndividual(it)
            "개별" -> Store.isIndividual(it)
            else -> true
        }
    }

    var diOpen by remember { mutableStateOf(false) }
    var diText by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        // ── 세그먼트 필터(전체/ETF/개별) + 직접입력 + refresh (한 줄) ──
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Row(Modifier.weight(1f).clip(RoundedCornerShape(9.dp)).background(SurfaceInput).padding(2.dp),
                horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                listOf("전체", "ETF", "개별").forEach { f ->
                    val on = filter == f
                    Box(
                        Modifier.weight(1f).clip(RoundedCornerShape(7.dp))
                            .background(if (on) SegmentOn else Color.Transparent)
                            .clickable { filter = f }.padding(vertical = 6.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(f, color = if (on) TextPrimary else TextMuted, fontSize = 12.sp,
                            fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
                    }
                }
            }
            Box(
                Modifier.clip(RoundedCornerShape(8.dp))
                    .background(if (diOpen) SegmentOn else SurfaceInput)
                    .clickable { diOpen = !diOpen }.padding(horizontal = 9.dp, vertical = 7.dp),
            ) { Text("⌨", fontSize = 13.sp) }
            Box(
                Modifier.clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                    .clickable { vm.refresh() }.padding(horizontal = 9.dp, vertical = 7.dp),
            ) { Text("🔄", fontSize = 13.sp) }
        }
        // 직접입력 펼침
        if (diOpen) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(diText, { diText = it },
                    placeholder = { Text("NVDA · 005930", fontSize = 11.sp) },
                    singleLine = true, modifier = Modifier.weight(1f))
                Button(onClick = {
                    val t = diText.trim().uppercase()
                    if (t.isNotEmpty()) { vm.select(t); diOpen = false; diText = "" }
                }) { Text("분석") }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            // ── 좌측: 종목 알약 버튼 (폭 ⅔로 축소) ──
            Column(Modifier.weight(0.22f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                tickers.forEach { tk ->
                    TickerPill(tk, ov[tk], selected = tk == s.ticker) { vm.select(tk) }
                }
            }
            // ── 우측: 분석 콘텐츠 ──
            Column(Modifier.weight(0.78f), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                when {
                    s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                        CircularProgressIndicator()
                    }
                    s.error != null -> Text("⚠️ ${s.error}", color = signalColor("strong_sell"))
                    s.result != null -> ResultView(s.result, s.ticker, s.ohlc, ov[s.ticker]?.day)
                }
            }
        }

        // ── 하단 풀폭 아코디언 (지표설명 · 매매기록) ──
        if (s.result != null) {
            Accordion("📝 매매 기록 입력") { TradeInputSection(s.ticker) }
            Accordion("🗑️ 매매 기록 삭제 / 메모 편집") { TradeListSection(s.ticker) }
        }
    }
}

/** 좌측 종목 칩 — M 히트맵 배경, 상태 점(보유=금채움/이력=금링) + 티커 … 일간%. */
@Composable
private fun TickerPill(tk: String, row: com.quant.dashboard.data.OverviewRepo.Row?, selected: Boolean, onClick: () -> Unit) {
    val bg = if (row != null) mHeat(row.mPct) else BgCard
    val day = row?.day
    val dayStr = day?.let { (if (it >= 0) "+" else "") + "%.1f".format(it) } ?: ""
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(7.dp)).background(bg)
            .then(if (selected) Modifier.border(1.5.dp, Profit, RoundedCornerShape(7.dp)) else Modifier)
            .clickable { onClick() }.padding(horizontal = 6.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        // 상태 점: 보유=금색 채움 / 이력=금색 링 / 관심=없음
        StatusDot(when { row?.holding == true -> 2; row?.hasHistory == true -> 1; else -> 0 })
        Text(Tickers.displayName(tk), color = TextPrimary, fontSize = 11.sp,
            maxLines = 1, fontWeight = FontWeight.SemiBold, fontFamily = Mono,
            modifier = Modifier.weight(1f))
        if (dayStr.isNotEmpty()) {
            Text(dayStr, color = if ((day ?: 0.0) >= 0) Profit else Loss,
                fontSize = 10.sp, maxLines = 1, fontWeight = FontWeight.Bold, fontFamily = Mono)
        }
    }
}

/** 상태 점 (0=없음 / 1=이력 링 / 2=보유 채움), 금색. */
@Composable
private fun StatusDot(state: Int) {
    if (state == 0) { Spacer(Modifier.size(7.dp)); return }
    Box(
        Modifier.size(7.dp).clip(RoundedCornerShape(50))
            .then(if (state == 2) Modifier.background(com.quant.dashboard.ui.theme.Gold)
            else Modifier.border(1.4.dp, com.quant.dashboard.ui.theme.Gold, RoundedCornerShape(50))),
    )
}

/** 하단 아코디언 카드 (Streamlit expander 미러). */
@Composable
private fun Accordion(title: String, content: @Composable () -> Unit) {
    var open by remember { mutableStateOf(false) }
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(BgCard)
            .border(1.dp, BorderColor, RoundedCornerShape(12.dp)).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("${if (open) "⌄" else "›"}  $title", color = TextPrimary, fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.fillMaxWidth().clickable { open = !open })
        if (open) content()
    }
}

@Composable
private fun ResultView(r: Quant.Result, ticker: String, ohlc: List<Candle>, dayPct: Double?) {
    // ── 종목명 + σ·β·Z·M 인라인 (한 줄, 넘치면 가로 스크롤) ──
    Row(
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
    ) {
        Text(Tickers.displayName(ticker), color = Profit,
            fontSize = 22.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
        dayPct?.let {
            Text("${if (it >= 0) "+" else ""}${"%.1f%%".format(it)}",
                color = if (it > 0) Profit else if (it < 0) Loss else Neutral,
                fontSize = 13.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
        }
        Text("σ±%.0f%% · β %.1f".format(r.sigmaPct, r.beta), color = TextSecondary, fontSize = 11.sp, fontFamily = Mono)
    }

    // ── 정보 카드 (현재가/평균단가/보유) ──
    val trades = remember(ticker) { Store.loadTrades()[ticker].orEmpty() }
    val pos = remember(ticker) { Portfolio.position(trades) }
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp))
            .then(if (pos != null) Modifier.background(HoldingBg) else Modifier)
            .border(1.5.dp, if (pos != null) HoldingBorder else BorderColor, RoundedCornerShape(10.dp))
            .padding(10.dp),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Column {
            Text("현재가", color = TextMuted, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
            Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(Tickers.priceLabel(ticker, r.lastPrice), color = TextPrimary, fontSize = 17.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
                // 보유 중이면 매수 평균단가 대비 손익률, 아니면 일간 등락률
                val chgPct = if (pos != null && pos.avg > 0) (r.lastPrice / pos.avg - 1.0) * 100.0 else dayPct
                chgPct?.let {
                    Text("(${if (it >= 0) "+" else ""}${"%.1f%%".format(it)})",
                        color = if (it > 0) Profit else if (it < 0) Loss else Neutral, fontSize = 11.sp, fontFamily = Mono)
                }
            }
        }
        if (pos != null) {
            Column {
                Text("평균단가", color = TextMuted, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                Text(Tickers.priceLabel(ticker, pos.avg), color = TextPrimary, fontSize = 17.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
            }
            Column {
                Text("보유수량", color = TextMuted, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
                Text("${pos.qty}주", color = TextPrimary, fontSize = 17.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
            }
        }
    }

    // ── 차트 윈도우 + 매매 마커/화살표 ──
    val periodMonths = Store.chartMonths()
    val n = r.dates.size
    val cutoff = r.dates[n - 1] - periodMonths.toLong() * 30 * 86400
    var start = 0
    while (start < n - 2 && r.dates[start] < cutoff) start++
    val base = r.price[0]
    fun seg(a: DoubleArray) = a.copyOfRange(start, n)
    fun segDollar(a: DoubleArray) = DoubleArray(n - start) { a[start + it] * base }
    val dates = r.dates.copyOfRange(start, n)

    val priceMarks = ArrayList<Mark>()
    val zmMarks = ArrayList<Mark>()
    val scatterIdx = ArrayList<Pair<Int, Boolean>>()
    for (t in trades) {
        val sec = try { LocalDate.parse(t.date).toEpochDay() * 86400 } catch (e: Exception) { continue }
        val buy = t.type == "buy"
        var full = -1
        for (i in r.dates.indices) { if (r.dates[i] <= sec) full = i else break }
        if (full >= 0) scatterIdx.add(full to buy)
        if (full >= start) {
            val wi = full - start
            priceMarks.add(Mark(wi, t.price, buy))
            val mv = seg(r.mPct)[wi]
            if (!mv.isNaN()) zmMarks.add(Mark(wi, mv, buy))
        }
    }
    val arrows = ArrayList<CycleArrow>()
    run {
        val sorted = trades.filter { it.qty > 0 && it.price > 0 }.sortedBy { it.date }
        var holdQty = 0; var buyQty = 0; var buyCost = 0.0; var sellQty = 0; var sellProceeds = 0.0
        var firstBuyFull = -1; var lastSellFull = -1
        for (t in sorted) {
            val sec = try { LocalDate.parse(t.date).toEpochDay() * 86400 } catch (e: Exception) { continue }
            var full = -1
            for (i in r.dates.indices) { if (r.dates[i] <= sec) full = i else break }
            if (t.type == "buy") {
                if (holdQty == 0) { buyQty = 0; buyCost = 0.0; sellQty = 0; sellProceeds = 0.0; firstBuyFull = full }
                holdQty += t.qty; buyQty += t.qty; buyCost += t.qty * t.price
            } else if (t.type == "sell" && holdQty > 0) {
                sellQty += t.qty; sellProceeds += t.qty * t.price; lastSellFull = full
                holdQty = maxOf(holdQty - t.qty, 0)
                if (holdQty == 0 && buyQty > 0 && sellQty > 0) {
                    val avgBuy = buyCost / buyQty; val avgSell = sellProceeds / sellQty
                    val x1 = firstBuyFull - start; val x2 = lastSellFull - start
                    if (x1 >= 0 && x2 >= 0) arrows.add(CycleArrow(x1, avgBuy, x2, avgSell, avgSell >= avgBuy))
                }
            }
        }
    }
    // ① 회귀 산점도 — 전체 분석기간(조회기간 미적용)
    ChartCard("회귀 산점도", "SPY 대비", "β ${"%.2f".format(r.beta)}", Profit) {
        RegressionScatter(r.spyNorm, r.tickerNorm, r.predicted,
            r.bandUpper, r.bandLower, r.beta, markIdx = scatterIdx)
    }

    // ② Z·M 궤적 — 전체 분석기간(조회기간 미적용)
    ChartCard("Z·M 궤적", "시간색 · ● 현재") {
        ZmScatter(r.zPct, r.mPct, scatterIdx)
    }

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
    ChartCard("가격 · 일봉", value = Tickers.priceLabel(ticker, r.lastPrice)) {
        if (closes.any { !it.isNaN() }) {
            CandleChart(opens, highs, lows, closes,
                segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                markers = priceMarks, arrows = arrows,
                currency = Tickers.currencySymbol(ticker), topLabel = "",
                dates = dates, dailyChgPct = dayPct ?: Double.NaN)
        } else {
            PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                markers = priceMarks, arrows = arrows, currency = Tickers.currencySymbol(ticker))
        }
    }

    // ④ Z·M 오실레이터
    ChartCard("Z ${"%.0f".format(r.lastZpct)} · M ${"%.0f".format(r.lastMpct)}", "Z 빨강 · M 회색") {
        ZmChart(seg(r.zPct), seg(r.mPct), zmMarks)
    }

    // ⑤ MACD
    val macdW = seg(r.macd); val sigW = seg(r.macdSignal)
    val macdLast = macdW.lastOrNull { !it.isNaN() } ?: 0.0
    val sigLast = sigW.lastOrNull { !it.isNaN() } ?: 0.0
    ChartCard("MACD", "● MACD · ● Signal",
        "${"%.2f".format(macdLast)} (${"%+.2f".format(macdLast - sigLast)})") {
        MacdChart(macdW, sigW)
    }

    // ⑥ RSI
    val rsiLast = r.rsi.lastOrNull { !it.isNaN() } ?: 50.0
    ChartCard("RSI", value = "%.1f".format(rsiLast), valueColor = Teal) {
        RsiChart(seg(r.rsi))
        DateAxis(dates)
    }
}

/** 차트 카드 — 헤더(제목/부제 + 우측 값) + 플롯. */
@Composable
private fun ChartCard(title: String, sub: String = "", value: String = "",
                      valueColor: Color = TextSecondary, content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BgCard)
            .border(1.dp, BorderColor, RoundedCornerShape(14.dp)).padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(title, color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
            if (sub.isNotEmpty()) Text("  $sub", color = TextMuted, fontSize = 11.sp)
            Spacer(Modifier.weight(1f))
            if (value.isNotEmpty()) Text(value, color = valueColor, fontSize = 12.sp,
                fontWeight = FontWeight.Bold, fontFamily = Mono)
        }
        content()
    }
}

@Composable
private fun TradeInputSection(ticker: String) {
    var refresh by remember { mutableStateOf(0) }
    var type by remember { mutableStateOf("buy") }
    var qty by remember { mutableStateOf("") }
    var price by remember { mutableStateOf("") }
    var date by remember { mutableStateOf(LocalDate.now().toString()) }
    var memo by remember { mutableStateOf("") }
    var err by remember { mutableStateOf<String?>(null) }
    refresh.let {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("종목: ${Tickers.displayName(ticker)}", color = TextSecondary, fontSize = 12.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                FilterChip(selected = type == "buy", onClick = { type = "buy" }, label = { Text("매수") })
                FilterChip(selected = type == "sell", onClick = { type = "sell" }, label = { Text("매도") })
            }
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedTextField(date, { date = it }, label = { Text("날짜") }, singleLine = true, modifier = Modifier.weight(1.4f))
                OutlinedTextField(qty, { qty = it }, label = { Text("수량") }, singleLine = true, modifier = Modifier.weight(1f))
                OutlinedTextField(price, { price = it }, label = { Text("단가$") }, singleLine = true, modifier = Modifier.weight(1f))
            }
            OutlinedTextField(memo, { memo = it }, label = { Text("메모 (선택)") }, singleLine = true, modifier = Modifier.fillMaxWidth())
            Button(onClick = {
                val q = qty.toIntOrNull(); val p = price.toDoubleOrNull()
                if (q == null || q <= 0 || p == null || p <= 0) err = "수량·단가를 올바르게 입력하세요"
                else { Store.addTrade(ticker, Trade(date.trim(), type, q, p, memo.ifBlank { null })); qty = ""; price = ""; memo = ""; err = null; refresh++ }
            }) { Text("저장") }
            err?.let { Text(it, color = Loss, fontSize = 12.sp) }
        }
    }
}

@Composable
private fun TradeListSection(ticker: String) {
    var refresh by remember { mutableStateOf(0) }
    val trades = remember(ticker, refresh) { Store.loadTrades()[ticker].orEmpty() }
    var editIdx by remember { mutableStateOf(-1) }
    var editMemo by remember { mutableStateOf("") }
    if (trades.isEmpty()) { Text("기록 없음", color = TextSecondary, fontSize = 12.sp); return }
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        trades.forEachIndexed { i, t ->
            Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween, Alignment.CenterVertically) {
                Text("${t.date}  ${if (t.type == "buy") "▲" else "▼"} ${t.qty}주 @$${"%.2f".format(t.price)}" + (t.memo?.let { "  · $it" } ?: ""),
                    color = TextSecondary, fontSize = 12.sp, modifier = Modifier.weight(1f))
                TextButton(onClick = { editIdx = if (editIdx == i) -1 else i; editMemo = t.memo ?: "" }) { Text("메모", fontSize = 12.sp) }
                TextButton(onClick = { Store.deleteTrade(ticker, i); refresh++ }) { Text("삭제", color = Loss, fontSize = 12.sp) }
            }
            if (editIdx == i) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                    OutlinedTextField(editMemo, { editMemo = it }, label = { Text("메모") }, singleLine = true, modifier = Modifier.weight(1f))
                    Button(onClick = { Store.updateTradeMemo(ticker, i, editMemo); editIdx = -1; refresh++ }) { Text("저장") }
                }
            }
        }
    }
}

