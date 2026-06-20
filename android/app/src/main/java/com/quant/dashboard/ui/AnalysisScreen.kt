package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
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
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Neutral
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
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
        // ── 분류 라디오(전체/ETF/개별) + 직접입력 + refresh (한 줄) ──
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                listOf("전체", "ETF", "개별").forEach { f ->
                    Text(
                        "${if (filter == f) "●" else "○"} $f",
                        color = if (filter == f) TextPrimary else TextSecondary,
                        fontSize = 13.sp, fontWeight = if (filter == f) FontWeight.Bold else FontWeight.Normal,
                        modifier = Modifier.clickable { filter = f },
                    )
                }
            }
            Text(if (diOpen) "⌨ 닫기" else "⌨ 직접입력", color = TextSecondary, fontSize = 12.sp,
                modifier = Modifier.clickable { diOpen = !diOpen }.padding(end = 10.dp))
            Box(
                Modifier.clip(RoundedCornerShape(6.dp)).background(Color(0xFF21262D))
                    .clickable { vm.refresh() }.padding(horizontal = 8.dp, vertical = 4.dp),
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
            Accordion("ℹ️ 지표 설명 (σ · β·SPY · Z · M)") { IndicatorHelpBody() }
            Accordion("📝 매매 기록 입력") { TradeInputSection(s.ticker) }
            Accordion("🗑️ 매매 기록 삭제 / 메모 편집") { TradeListSection(s.ticker) }
        }
    }
}

/** 좌측 종목 알약 버튼 — 모멘텀색 배경, ★보유/☆이력, 중앙정렬 굵은 라벨 + 일간%. */
@Composable
private fun TickerPill(tk: String, row: com.quant.dashboard.data.OverviewRepo.Row?, selected: Boolean, onClick: () -> Unit) {
    val bg = if (row != null) pctColor(row.mPct) else Neutral
    val dark = row != null && (row.mPct < 20 || row.mPct >= 80)
    val dayStr = row?.let { (if (it.day >= 0) "+" else "") + "%.1f%%".format(it.day) } ?: ""
    val mark = when { row?.holding == true -> "★ "; row?.hasHistory == true -> "☆ "; else -> "" }
    Box(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(7.dp)).background(bg)
            .then(if (selected) Modifier.border(1.5.dp, Color.White, RoundedCornerShape(7.dp)) else Modifier)
            .clickable { onClick() }.padding(horizontal = 5.dp, vertical = 4.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            "$mark${Tickers.displayName(tk)} $dayStr",
            color = if (dark) Color.White else Color.Black,
            fontSize = 10.sp, maxLines = 1, fontWeight = FontWeight.Bold,
        )
    }
}

/** 하단 아코디언 카드 (Streamlit expander 미러). */
@Composable
private fun Accordion(title: String, content: @Composable () -> Unit) {
    var open by remember { mutableStateOf(false) }
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp))
            .border(1.dp, Color(0xFF30363D), RoundedCornerShape(10.dp)).padding(12.dp),
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
        Text(Tickers.displayName(ticker), color = pctColor(r.lastMpct),
            fontSize = 18.sp, fontWeight = FontWeight.Bold)
        Text("σ±%.0f%%".format(r.sigmaPct), color = TextSecondary, fontSize = 11.sp)
        Text("β·SPY %.1f×".format(r.beta), color = TextSecondary, fontSize = 11.sp)
        Text("Z %.0f".format(r.lastZpct), color = pctColor(r.lastZpct), fontSize = 11.sp, fontWeight = FontWeight.Bold)
        Text("M %.0f".format(r.lastMpct), color = pctColor(r.lastMpct), fontSize = 11.sp, fontWeight = FontWeight.Bold)
    }

    // ── 정보 카드 (현재가/평균단가/보유) ──
    val trades = remember(ticker) { Store.loadTrades()[ticker].orEmpty() }
    val pos = remember(ticker) { Portfolio.position(trades) }
    Row(
        Modifier.fillMaxWidth()
            .border(1.5.dp, if (pos != null) Color(0xFF3FB950) else Color(0xFF6B7280), RoundedCornerShape(8.dp))
            .padding(8.dp),
        horizontalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Column {
            Text("현재가", color = TextSecondary, fontSize = 11.sp)
            Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(Tickers.priceLabel(ticker, r.lastPrice), color = TextPrimary, fontSize = 16.sp, fontWeight = FontWeight.Bold)
                // 보유 중이면 매수 평균단가 대비 손익률, 아니면 일간 등락률
                val chgPct = if (pos != null && pos.avg > 0) (r.lastPrice / pos.avg - 1.0) * 100.0 else dayPct
                chgPct?.let {
                    Text("(${if (it >= 0) "+" else ""}${"%.1f%%".format(it)})",
                        color = if (it > 0) Profit else if (it < 0) Loss else Neutral, fontSize = 11.sp)
                }
            }
        }
        if (pos != null) {
            Column {
                Text("평균단가", color = TextSecondary, fontSize = 11.sp)
                Text(Tickers.priceLabel(ticker, pos.avg), color = TextPrimary, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            }
            Column {
                Text("보유수량", color = TextSecondary, fontSize = 11.sp)
                Text("${pos.qty}주", color = TextPrimary, fontSize = 16.sp, fontWeight = FontWeight.Bold)
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
    Text("회귀 산점도 (SPY 대비) · 분석기간", color = TextSecondary, fontSize = 11.sp)
    RegressionScatter(r.spyNorm, r.tickerNorm, r.predicted,
        r.bandUpper, r.bandLower, r.beta, markIdx = scatterIdx)

    // ② Z·M 궤적 — 전체 분석기간(조회기간 미적용)
    Text("Z·M 궤적 (Turbo 시간색, ★ 현재) · 분석기간", color = TextSecondary, fontSize = 11.sp)
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
private fun IndicatorHelpBody() {
    Text(
        "σ = SPY 회귀 잔차의 변동성(±%). 클수록 SPY 추세에서 벗어나는 폭이 큼.\n" +
            "β·SPY = SPY 대비 민감도(배). 1보다 크면 SPY보다 더 출렁임.\n" +
            "Z = 회귀선 대비 현재 가격 위치(0~100). 낮을수록 저평가(매수), 높을수록 고평가(매도).\n" +
            "M = 모멘텀(0~100). RSI·MACD·변곡을 변동성으로 정규화한 종합 추세. 낮을수록 매수 우위.",
        color = TextSecondary, fontSize = 12.sp,
    )
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

