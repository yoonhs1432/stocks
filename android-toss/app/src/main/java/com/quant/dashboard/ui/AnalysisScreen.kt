package com.quant.dashboard.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
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
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Candle
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.MarketHours
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
import com.quant.dashboard.ui.theme.ProfitBtn
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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalysisScreen(vm: AnalysisViewModel = viewModel(), onBack: () -> Unit = {}) {
    val s = vm.state

    // 설정/기준일 변경과 다른 탭에서 넘어온 종목을 한 효과에서 처리 —
    // 따로 두면 두 로드가 경쟁해 이전 종목 결과가 늦게 도착해 화면에 남는 경우가 있었음
    LaunchedEffect(AppState.dataVersion, AppState.pendingTicker) {
        val pending = AppState.pendingTicker
        if (pending != null) AppState.pendingTicker = null
        vm.sync(AppState.dataVersion, pending)
    }
    // 자동 새로고침 — 화면 켜진 분석 탭 + 장중에만, 60초 (조용히)
    LaunchedEffect(Unit) {
        while (true) {
            kotlinx.coroutines.delay(60_000)
            if (MarketHours.anyOpen()) vm.autoRefresh()
        }
    }

    val ov = vm.overview.associateBy { it.ticker }

    var diOpen by remember { mutableStateOf(false) }
    var diText by remember { mutableStateOf("") }

    // 기기 뒤로가기 → 비교 탭으로
    BackHandler { onBack() }

    Column(modifier = Modifier.fillMaxSize().background(BgApp)) {
        // ── 상단 고정: 뒤로가기(비교로) + 직접입력 토글 ──
        Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Box(
                Modifier.clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                    .clickable { onBack() }.padding(horizontal = 12.dp, vertical = 7.dp),
            ) { Text("← 비교", color = TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.Bold) }
            Spacer(Modifier.weight(1f))
            Box(
                Modifier.clip(RoundedCornerShape(8.dp))
                    .background(if (diOpen) SegmentOn else SurfaceInput)
                    .clickable { diOpen = !diOpen }.padding(horizontal = 12.dp, vertical = 7.dp),
            ) { Text("⌨ 직접", color = if (diOpen) TextPrimary else TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.Bold) }
        }
        // ── 상단 고정: 직접입력 (열렸을 때만) ──
        if (diOpen) {
            Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
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

        // ── 본문: 당겨서 새로고침 + 그래프 2열 그리드 + 매매 아코디언 ──
        Box(Modifier.weight(1f).fillMaxWidth()) {
            when {
                // 당겨서 새로고침 → 전체 탭 새로고침 (dataVersion bump로 모든 탭이 재로드)
                s.result != null -> PullToRefreshBox(
                    isRefreshing = s.loading, onRefresh = { AppState.bump() },
                    modifier = Modifier.fillMaxSize(),
                ) {
                    Column(
                        Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                            .padding(horizontal = 8.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        ResultView(s.result, s.ticker, s.ohlc, ov[s.ticker]?.day)
                        Spacer(Modifier.height(4.dp))
                    }
                }
                s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                    CircularProgressIndicator()
                }
                s.error != null -> Text("⚠️ ${s.error}", color = signalColor("strong_sell"),
                    modifier = Modifier.padding(16.dp))
            }
        }

    }
}

/** 하단 고정 종목 선택 바 — 보유 토글 + 직접입력 + 가로 스크롤 종목 칩. */
@Composable
private fun TickerBar(
    tickers: List<String>,
    ov: Map<String, com.quant.dashboard.data.OverviewRepo.Row>,
    selected: String,
    holdingsOnly: Boolean, onToggleHold: () -> Unit,
    diOpen: Boolean, onToggleDi: () -> Unit,
    onSelect: (String) -> Unit,
) {
    Column {
        Box(Modifier.fillMaxWidth().height(1.dp).background(com.quant.dashboard.ui.theme.DividerColor))
        Row(
            Modifier.fillMaxWidth().background(com.quant.dashboard.ui.theme.BgElevated)
                .horizontalScroll(rememberScrollState()).padding(horizontal = 8.dp, vertical = 7.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // 보유 토글 (가장 처음)
            BarToggle("보유", holdingsOnly, Color(0xFF2EA078), onToggleHold)
            // 직접입력 토글
            BarToggle("＋직접", diOpen, SegmentOn, onToggleDi)
            tickers.forEach { tk -> TickerChipH(tk, ov[tk], tk == selected) { onSelect(tk) } }
        }
    }
}

/** 바 토글 칩 (보유/직접). */
@Composable
private fun BarToggle(label: String, on: Boolean, onBg: Color, onClick: () -> Unit) {
    Box(
        Modifier.clip(RoundedCornerShape(8.dp)).background(if (on) onBg else SurfaceInput)
            .clickable { onClick() }.padding(horizontal = 11.dp, vertical = 6.dp),
    ) {
        Text(label, color = if (on) Color.White else TextSecondary, fontSize = 12.sp,
            fontWeight = FontWeight.Bold, maxLines = 1)
    }
}

/** 가로 종목 칩 — 전체 M 색 배경, 상태 점 + 티커 + 일간%, 선택=초록 테두리. */
@Composable
private fun TickerChipH(tk: String, row: com.quant.dashboard.data.OverviewRepo.Row?, selected: Boolean, onClick: () -> Unit) {
    val bg = if (row != null) pctColor(row.mPct) else BgCard
    val dark = row != null && (row.mPct < 20 || row.mPct >= 80)
    val txt = if (dark) Color.White else Color.Black
    val day = row?.day
    val dayStr = day?.let { (if (it >= 0) "+" else "") + "%.1f".format(it) } ?: ""
    Row(
        Modifier.clip(RoundedCornerShape(8.dp)).background(bg)
            .then(if (selected) Modifier.border(2.dp, Color(0xFF2EA078), RoundedCornerShape(8.dp)) else Modifier)
            .clickable { onClick() }.padding(horizontal = 9.dp, vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        StatusDot(when { row?.holding == true -> 2; row?.hasHistory == true -> 1; else -> 0 })
        Text(Tickers.displayName(tk), color = txt, fontSize = 12.sp, maxLines = 1,
            fontWeight = FontWeight.Bold, fontFamily = Mono)
        if (dayStr.isNotEmpty()) {
            Text(dayStr, color = txt, fontSize = 10.sp, maxLines = 1, fontWeight = FontWeight.Bold, fontFamily = Mono)
        }
    }
}

/** 헤더 우측 소형 지표 (라벨 + 값 + 선택적 등락). */
@Composable
private fun Mini(label: String, value: String, extra: String? = null, extraColor: Color = TextMuted) {
    Row(verticalAlignment = Alignment.Bottom, horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(label, color = TextMuted, fontSize = 9.sp)
        Text(value, color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
        if (extra != null) Text(extra, color = extraColor, fontSize = 10.sp, fontFamily = Mono)
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
    // ── 종목명 + σ·β + (우측) 현재가/평단/수량 — 한 줄, 넘치면 가로 스크롤 ──
    val trades = remember(ticker) { Store.visibleTrades()[ticker].orEmpty() }
    val pos = remember(ticker) { Portfolio.position(trades) }
    Row(
        verticalAlignment = Alignment.Bottom,
        horizontalArrangement = Arrangement.spacedBy(7.dp),
        modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
    ) {
        Text(Tickers.displayName(ticker), color = Profit,
            fontSize = 20.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
        dayPct?.let {
            Text("${if (it >= 0) "+" else ""}${"%.1f%%".format(it)}",
                color = if (it > 0) Profit else if (it < 0) Loss else Neutral,
                fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
        }
        Text("σ±%.0f%% · β %.1f".format(r.sigmaPct, r.beta), color = TextSecondary, fontSize = 10.sp, fontFamily = Mono)
        // 우측: 현재가/평단/수량 (작게)
        val livePx = LivePrices.price(ticker) ?: r.lastPrice
        val chgPct = if (pos != null && pos.avg > 0) (livePx / pos.avg - 1.0) * 100.0 else null
        Mini("현재가", Tickers.priceLabel(ticker, livePx),
            extra = chgPct?.let { "${if (it >= 0) "+" else ""}${"%.1f%%".format(it)}" },
            extraColor = chgPct?.let { if (it >= 0) Profit else Loss } ?: TextMuted)
        if (pos != null) {
            Mini("평단", Tickers.priceLabel(ticker, pos.avg))
            Mini("보유", "${pos.qty}주")
        }
    }

    // ── 차트 기간 (시계열 4개에만 적용 — 산점도 2개는 전 구간 기준) ──
    // 상태는 여기서 선언해야 아래 윈도우 계산에 쓸 수 있고, 버튼은 차트 아래에 그린다.
    var periodMonths by remember { mutableStateOf(Store.chartMonths()) }

    // ── 차트 윈도우 + 매매 마커/화살표 ──
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
    // 캔들 데이터 준비
    val candleByDay = remember(ohlc) { ohlc.associateBy { it.t / 86400L } }
    val wN = n - start
    val opens = DoubleArray(wN); val highs = DoubleArray(wN)
    val lows = DoubleArray(wN); val closes = DoubleArray(wN)
    for (i in 0 until wN) {
        val cd = candleByDay[dates[i] / 86400L]
        if (cd != null) { opens[i] = cd.open; highs[i] = cd.high; lows[i] = cd.low; closes[i] = cd.close }
        else { opens[i] = Double.NaN; highs[i] = Double.NaN; lows[i] = Double.NaN; closes[i] = Double.NaN }
    }
    val macdW = seg(r.macd); val sigW = seg(r.macdSignal)
    val macdLast = macdW.lastOrNull { !it.isNaN() } ?: 0.0
    val sigLast = sigW.lastOrNull { !it.isNaN() } ?: 0.0
    val rsiLast = r.rsi.lastOrNull { !it.isNaN() } ?: 50.0
    val gh = 106.dp   // 그리드 차트 높이

    // 확대 다이얼로그로 넘길 선택 차트 인덱스(-1=닫힘). 그리드/확대에서 동일 렌더 재사용.
    var zoom by remember(ticker) { mutableStateOf(-1) }
    // 확대 다이얼로그의 축 확대/이동 상태 — 차트를 바꾸면 원본 배율로 리셋
    var view by remember(ticker, zoom) { mutableStateOf(ChartView()) }
    // 인덱스→확대 차트 렌더 (h=차트 높이, m=제스처 modifier). 0회귀 1Z·M궤적 2일봉 3Z·M 4MACD 5RSI
    val renderChart: @Composable (Int, androidx.compose.ui.unit.Dp, Modifier) -> Unit = { idx, h, m ->
        when (idx) {
            0 -> RegressionScatter(r.spyNorm, r.tickerNorm, r.predicted, r.bandUpper, r.bandLower, r.beta,
                markIdx = scatterIdx, height = h, view = view, zoomed = true, modifier = m)
            1 -> ZmScatter(r.zPct, r.mPct, scatterIdx, height = h, view = view, zoomed = true, modifier = m)
            2 -> {
                if (closes.any { !it.isNaN() })
                    CandleChart(opens, highs, lows, closes, segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                        markers = priceMarks, currency = Tickers.currencySymbol(ticker), dates = dates,
                        dailyChgPct = dayPct ?: Double.NaN, height = h, view = view, zoomed = true, modifier = m)
                else PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                    markers = priceMarks, currency = Tickers.currencySymbol(ticker),
                    height = h, view = view, zoomed = true, modifier = m)
                DateAxis(dates, view, zoomed = true)
            }
            3 -> { ZmChart(seg(r.zPct), seg(r.mPct), zmMarks, height = h, view = view, zoomed = true, modifier = m); DateAxis(dates, view, zoomed = true) }
            4 -> { MacdChart(macdW, sigW, height = h, view = view, zoomed = true, modifier = m); DateAxis(dates, view, zoomed = true) }
            else -> { RsiChart(seg(r.rsi), height = h, view = view, zoomed = true, modifier = m); DateAxis(dates, view, zoomed = true) }
        }
    }

    // ── 1행: ① 회귀 산점도 · ② Z·M 궤적 ──
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ChartCard(Modifier.weight(1f), "회귀 산점도", value = "β ${"%.2f".format(r.beta)}", valueColor = Profit, onClick = { zoom = 0 }) {
            RegressionScatter(r.spyNorm, r.tickerNorm, r.predicted,
                r.bandUpper, r.bandLower, r.beta, markIdx = scatterIdx, height = gh)
        }
        ChartCard(Modifier.weight(1f), "Z·M 궤적", sub = "현재 ★", onClick = { zoom = 1 }) {
            ZmScatter(r.zPct, r.mPct, scatterIdx, height = gh)
        }
    }

    // ── 2행: ③ 가격 캔들 · ④ Z·M 오실레이터 ──
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ChartCard(Modifier.weight(1f), "가격·일봉", value = Tickers.priceLabel(ticker, r.lastPrice), onClick = { zoom = 2 }) {
            if (closes.any { !it.isNaN() }) {
                CandleChart(opens, highs, lows, closes,
                    segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                    markers = priceMarks,
                    currency = Tickers.currencySymbol(ticker), topLabel = "",
                    dates = dates, dailyChgPct = dayPct ?: Double.NaN, height = gh)
            } else {
                PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                    markers = priceMarks, currency = Tickers.currencySymbol(ticker), height = gh)
            }
        }
        ChartCard(Modifier.weight(1f), "Z·M", value = "Z${"%.0f".format(r.lastZpct)}·M${"%.0f".format(r.lastMpct)}", onClick = { zoom = 3 }) {
            ZmChart(seg(r.zPct), seg(r.mPct), zmMarks, height = gh)
        }
    }

    // ── 3행: ⑤ MACD · ⑥ RSI ──
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        ChartCard(Modifier.weight(1f), "MACD",
            value = "${"%.2f".format(macdLast)}(${"%+.2f".format(macdLast - sigLast)})", onClick = { zoom = 4 }) {
            MacdChart(macdW, sigW, height = gh)
            DateAxis(dates)
        }
        ChartCard(Modifier.weight(1f), "RSI", value = "%.1f".format(rsiLast), valueColor = Teal, onClick = { zoom = 5 }) {
            RsiChart(seg(r.rsi), height = gh)
            DateAxis(dates)
        }
    }

    // 시계열 차트 4개(가격·Z·M·MACD·RSI)의 표시 기간
    Row(Modifier.fillMaxWidth().padding(top = 2.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("차트 기간", color = TextMuted, fontSize = 11.sp,
            modifier = Modifier.align(Alignment.CenterVertically))
        listOf(1, 6, 12, 24).forEach { m ->
            val on = periodMonths == m
            Box(
                Modifier.clip(RoundedCornerShape(7.dp))
                    .background(if (on) Teal else SurfaceInput)
                    .clickable { periodMonths = m; Store.setChartMonths(m) }
                    .padding(horizontal = 11.dp, vertical = 4.dp),
            ) {
                Text(if (m >= 12) "${m / 12}년" else "${m}개월",
                    color = if (on) Color(0xFF0C0E11) else TextSecondary,
                    fontSize = 11.sp, fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
            }
        }
    }

    // ── 차트 확대 다이얼로그 (그리드에서 탭한 차트를 전체화면 크게) ──
    if (zoom >= 0) {
        val titles = listOf("회귀 산점도", "Z·M 궤적", "가격·일봉", "Z·M 오실레이터", "MACD", "RSI")
        Dialog(onDismissRequest = { zoom = -1 },
            properties = DialogProperties(usePlatformDefaultWidth = false)) {
            Surface(Modifier.fillMaxWidth(0.98f).fillMaxHeight(0.9f), color = BgApp,
                shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, BorderColor)) {
                Column(Modifier.fillMaxSize().padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(titles[zoom], color = TextPrimary, fontSize = 16.sp,
                            fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                        // 확대·이동 상태일 때만 원본 배율 복귀 버튼
                        if (!view.isIdentity) {
                            val zoomLabel = if (zoom <= 1) "${"%.1f".format(view.sx)}×${"%.1f".format(view.sy)}"
                            else "${"%.1f".format(view.sx)}×"
                            Box(
                                Modifier.clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                                    .clickable { view = ChartView() }
                                    .padding(horizontal = 10.dp, vertical = 5.dp),
                            ) {
                                Text("⟲ $zoomLabel",
                                    color = TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
                            }
                        }
                        Text("✕", color = TextSecondary, fontSize = 22.sp,
                            modifier = Modifier.clickable { zoom = -1 })
                    }
                    Text("${Tickers.displayName(ticker)}  ${Tickers.priceLabel(ticker, r.lastPrice)}",
                        color = Profit, fontSize = 14.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
                    // 시계열 차트(②~⑤)는 x축만 확대 — y는 보이는 구간에 자동으로 맞춰짐
                    val xOnly = zoom >= 2
                    Box(Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                        Column(Modifier.fillMaxWidth()) {
                            renderChart(zoom, if (zoom <= 1) 420.dp else 340.dp,
                                Modifier.chartGestures(view, { view = it }, onTap = { zoom = -1 }, xOnly = xOnly))
                        }
                    }
                    Text(
                        if (xOnly) "두 손가락 가로로 벌리면 기간 확대 (Y축은 자동) · 끌어서 이동 · 탭하면 닫기"
                        else "두 손가락 가로/세로로 벌리면 X·Y축 확대 · 확대 후 끌어서 이동 · 탭하면 닫기",
                        color = TextMuted, fontSize = 10.sp)
                }
            }
        }
    }

}

/** 차트 카드 — 헤더(제목/부제 + 우측 값) + 플롯. 그리드 셀에서는 modifier=weight 전달.
 *  onClick 지정 시 카드 전체 탭 → 확대 다이얼로그. (탭 힌트로 우측에 ⤢ 표시) */
@Composable
private fun ChartCard(modifier: Modifier = Modifier, title: String, sub: String = "", value: String = "",
                      valueColor: Color = TextSecondary, onClick: (() -> Unit)? = null,
                      content: @Composable () -> Unit) {
    Column(
        modifier.clip(RoundedCornerShape(12.dp)).background(BgCard)
            .border(1.dp, BorderColor, RoundedCornerShape(12.dp))
            .then(if (onClick != null) Modifier.clickable { onClick() } else Modifier)
            .padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(title, color = TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Bold,
                fontFamily = Mono, maxLines = 1)
            if (sub.isNotEmpty()) Text(" $sub", color = TextMuted, fontSize = 9.sp, maxLines = 1)
            Spacer(Modifier.weight(1f))
            if (value.isNotEmpty()) Text(value, color = valueColor, fontSize = 11.sp,
                fontWeight = FontWeight.Bold, fontFamily = Mono, maxLines = 1)
            if (onClick != null) Text(" ⤢", color = TextMuted, fontSize = 11.sp)
        }
        content()
    }
}
