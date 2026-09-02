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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
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
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Candle
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.MarketHours
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.quant.Portfolio
import com.quant.dashboard.quant.Quant
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgCard
import com.quant.dashboard.ui.theme.BorderColor
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
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.ui.unit.Dp

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
    // 차트 묶음 전환 (산점도 2개 / 시계열 4개) — 비교 탭 시장 전환과 같은 자리·같은 모양
    var group by remember { mutableStateOf(Store.chartGroup()) }

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
        // 차트 높이를 화면에 맞춰 계산하려면 가용 높이를 알아야 한다.
        // 스크롤 안쪽에서 재면 무한대가 나오므로 **스크롤 바깥**에서 잰다.
        BoxWithConstraints(Modifier.weight(1f).fillMaxWidth()) {
            val avail = maxHeight
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
                        ResultView(s.result, s.ticker, s.ohlc, ov[s.ticker]?.day, group, avail,
                            vm.chartView(s.ticker)) { vm.setChartView(s.ticker, it) }
                    }
                }
                s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                    CircularProgressIndicator()
                }
                s.error != null -> Text("⚠️ ${s.error}", color = signalColor("strong_sell"),
                    modifier = Modifier.padding(16.dp))
            }
        }

        // ── 하단 고정: 차트 묶음 전환 ──
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            listOf("scatter" to "산점도", "series" to "시계열").forEach { (id, label) ->
                val on = group == id
                Box(
                    Modifier.clip(RoundedCornerShape(8.dp))
                        .background(if (on) Teal else SurfaceInput)
                        .clickable { group = id; Store.setChartGroup(id) }
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                ) {
                    Text(label, color = if (on) Color(0xFF0C0E11) else TextSecondary,
                        fontSize = 13.sp, fontWeight = FontWeight.Bold)
                }
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
private fun ResultView(r: Quant.Result, ticker: String, ohlc: List<Candle>, dayPct: Double?,
                       group: String, avail: Dp,
                       savedView: ChartView?, onView: (ChartView) -> Unit) {
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

    // ── 차트 데이터 — 분석 기간 전체를 넘긴다 ──
    // 예전에는 여기서 기간만큼 잘라 보여주고 기간 버튼으로 바꿨는데,
    // 이제 보이는 구간은 ChartView(핀치=범위, 드래그=이동)가 정하므로 자르지 않는다.
    val n = r.dates.size
    val start = 0
    val base = r.price[0]
    fun seg(a: DoubleArray) = a
    fun segDollar(a: DoubleArray) = DoubleArray(n) { a[it] * base }
    val dates = r.dates

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

    if (group == "scatter") {
        // ── 산점도 2개 — 전체 폭으로 위아래 배치 (반폭일 땐 점이 뭉개졌다) ──
        // 화면에 딱 맞는 높이 — 헤더줄(34) + 카드 2개의 제목·여백(≈76) + 카드 사이(8)
        val sc = ((avail - 118.dp) / 2).coerceAtLeast(140.dp)
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            ChartCard(Modifier.fillMaxWidth(), "회귀 산점도", value = "β ${"%.2f".format(r.beta)}",
                valueColor = Profit, onClick = { zoom = 0 }) {
                // zoomed=true → 우측 y축·하단 x축 눈금 값 표시
                RegressionScatter(r.spyNorm, r.tickerNorm, r.predicted,
                    r.bandUpper, r.bandLower, r.beta, markIdx = scatterIdx,
                    height = sc, zoomed = true)
            }
            ChartCard(Modifier.fillMaxWidth(), "Z·M 궤적", sub = "현재 ★", onClick = { zoom = 1 }) {
                ZmScatter(r.zPct, r.mPct, scatterIdx, height = sc, zoomed = true)
            }
        }
    } else {
        // ── 시계열 4개 — x축을 통일해 4행으로 ──
        // ChartView 를 하나만 두고 넷이 공유한다. 아무 차트에서나 핀치·드래그하면 넷이 같이 움직이고,
        // 날짜축은 공통이므로 맨 아래 한 번만 그린다.
        // 카드 테두리를 빼고 얇은 제목줄만 둬서 네 개가 한 화면에 들어가게 했다.
        // 처음 보여줄 구간 = 설정의 "차트 표시기간". 전체를 그려 놓고 배율로 잘라 보여준다.
        val totalMonths = ((dates.last() - dates.first()).toDouble() / 2_629_746.0).coerceAtLeast(1.0)
        // 배율은 ViewModel 이 종목별로 들고 있다 — 탭을 옮겼다 와도 보던 구간이 유지된다.
        // 처음 보는 종목만 "차트 표시기간"으로 초기 배율을 만든다.
        var sView by remember(ticker, n) {
            mutableStateOf(
                savedView ?: run {
                    val s0 = (totalMonths / Store.chartMonths()).toFloat()
                        .coerceIn(1f, ChartView.MAX_ZOOM)
                    // nx = 1 - sx 이면 오른쪽 끝(최신)에 붙는다
                    ChartView(sx = s0, nx = 1f - s0)
                }
            )
        }
        fun pushView(v: ChartView) { sView = v; onView(v) }
        // 남는 높이 = 헤더줄(34) + 제목줄 4개(≈72) + 날짜축(20) + 안내(20)
        // MACD·RSI 는 보조지표라 가격·Z·M 보다 낮게 (가중치 1 : 1 : 0.7 : 0.7)
        val body = (avail - 146.dp).coerceAtLeast(360.dp)
        val shMain = (body / 3.4f).coerceAtLeast(100.dp)
        val shSub = (shMain * 0.7f).coerceAtLeast(70.dp)
        val gest = Modifier.chartGestures(sView, { pushView(it) }, xOnly = true)

        // 간격 없는 Column — 바깥 Column 의 spacedBy(8dp) 가 항목마다 붙으면
        // 제목줄·차트 사이가 벌어져 네 개가 한 화면에 안 들어간다
        Column(Modifier.fillMaxWidth()) {
        SeriesHeader("가격·일봉", Tickers.priceLabel(ticker, r.lastPrice), TextPrimary) { zoom = 2 }
        if (closes.any { !it.isNaN() }) {
            CandleChart(opens, highs, lows, closes,
                segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                markers = priceMarks, currency = Tickers.currencySymbol(ticker), topLabel = "",
                dates = dates, dailyChgPct = dayPct ?: Double.NaN,
                height = shMain, view = sView, zoomed = true, modifier = gest)
        } else {
            PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper),
                segDollar(r.bandLower), markers = priceMarks,
                currency = Tickers.currencySymbol(ticker),
                height = shMain, view = sView, zoomed = true, modifier = gest)
        }

        SeriesHeader("Z·M", "Z${"%.0f".format(r.lastZpct)}·M${"%.0f".format(r.lastMpct)}", TextPrimary) { zoom = 3 }
        ZmChart(seg(r.zPct), seg(r.mPct), zmMarks, height = shMain, view = sView, zoomed = true, modifier = gest)

        SeriesHeader("MACD", "${"%.2f".format(macdLast)}(${"%+.2f".format(macdLast - sigLast)})", TextPrimary) { zoom = 4 }
        MacdChart(macdW, sigW, height = shSub, view = sView, zoomed = true, modifier = gest)

        SeriesHeader("RSI", "%.1f".format(rsiLast), Teal) { zoom = 5 }
        RsiChart(seg(r.rsi), height = shSub, view = sView, zoomed = true, modifier = gest)

        DateAxis(dates, sView, zoomed = true)
        Text(
            if (sView.isIdentity) "두 손가락으로 기간 확대·축소 · 끌어서 이동"
            else "보이는 기간 ${"%.1f".format(totalMonths / sView.sx)}개월 — 탭하면 전체",
            color = TextMuted, fontSize = 11.sp,
            modifier = Modifier.fillMaxWidth().clickable { pushView(ChartView()) },
        )
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

/** 시계열 스택용 얇은 제목줄 — 카드 테두리 없이 제목·값·확대(⤢)만. 세로 공간을 아낀다. */
@Composable
private fun SeriesHeader(title: String, value: String, valueColor: Color, onZoom: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable { onZoom() }.padding(top = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(title, color = TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Bold,
            fontFamily = Mono, maxLines = 1)
        Spacer(Modifier.weight(1f))
        Text(value, color = valueColor, fontSize = 11.sp, fontWeight = FontWeight.Bold,
            fontFamily = Mono, maxLines = 1)
        Text(" ⤢", color = TextMuted, fontSize = 11.sp)
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
