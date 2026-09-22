package com.quant.dashboard.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.pulltorefresh.PullToRefreshBox
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
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
import com.quant.dashboard.quant.Quant
import com.quant.dashboard.ui.theme.Accent
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.Ghost
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Neutral
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.Teal
import com.quant.dashboard.ui.theme.TextMuted
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
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
    // 차트 묶음 전환 (산점도 2개 / 시계열 4개) — 비교 탭 시장 전환과 같은 자리·같은 모양
    var group by remember { mutableStateOf(Store.chartGroup()) }

    // 기기 뒤로가기 → 비교 탭으로
    BackHandler { onBack() }

    Column(modifier = Modifier.fillMaxSize().background(BgApp)) {
        // ── 상단 헤더: 제목 + 뒤로(비교) + 직접입력 토글 ──
        ScreenHeader("분석") {
            GhostButton("← 비교") { onBack() }
            Spacer(Modifier.width(6.dp))
            GhostButton("직접", color = if (diOpen) Accent else TextPrimary) { diOpen = !diOpen }
        }
        // ── 상단 고정: 직접입력 (열렸을 때만) ──
        if (diOpen) {
            Row(Modifier.fillMaxWidth().padding(horizontal = ScreenPad, vertical = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically) {
                OutlinedTextField(diText, { diText = it },
                    placeholder = { Text("NVDA · 005930", fontSize = 11.sp) },
                    singleLine = true, modifier = Modifier.weight(1f))
                GhostButton("분석", color = Accent) {
                    val t = diText.trim().uppercase()
                    if (t.isNotEmpty()) { vm.select(t); diOpen = false; diText = "" }
                }
            }
        }

        // ── 본문: 당겨서 새로고침 + 그래프 2열 그리드 + 매매 아코디언 ──
        // 차트 높이를 화면에 맞춰 계산하려면 가용 높이를 알아야 한다.
        // 스크롤 안쪽에서 재면 무한대가 나오므로 **스크롤 바깥**에서 잰다.
        BoxWithConstraints(Modifier.weight(1f).fillMaxWidth()) {
            // 본문에 실제로 남는 높이 — 아래 Column 의 세로 패딩(8+8)을 뺀 값이어야
            // ResultView 가 잰 여백과 더했을 때 딱 맞는다
            val avail = maxHeight - BODY_PAD * 2
            when {
                // 당겨서 새로고침 → 전체 탭 새로고침 (dataVersion bump로 모든 탭이 재로드)
                s.result != null -> PullToRefreshBox(
                    // bump() 만 하면 일봉 캐시 때문에 값이 그대로다 — 캐시를 건너뛰고 다시 받는다
                    isRefreshing = s.loading, onRefresh = { vm.refresh() },
                    modifier = Modifier.fillMaxSize(),
                ) {
                    Column(
                        Modifier.fillMaxSize().verticalScroll(rememberScrollState())
                            .padding(horizontal = 8.dp, vertical = BODY_PAD),
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        ResultView(s.result, s.ticker, s.ohlc, ov[s.ticker]?.day, group, avail,
                            s.minutes, s.trades, s.avgPrice, s.qty) { vm.loadMinutes() }
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
        BottomActionBar {
            UnderlineSegments(
                listOf("scatter" to "산점도", "series" to "시계열", "sub" to "보조지표"),
                selected = group,
                onSelect = { group = it; Store.setChartGroup(it) },
            )
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
            BarToggle("＋직접", diOpen, Ghost, onToggleDi)
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
    val bg = if (row != null) pctColor(row.mPct) else SurfaceInput
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
        Modifier.fillMaxWidth().padding(vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        HDivider()
        Text("${if (open) "⌄" else "›"}  $title", color = TextPrimary, fontSize = 14.sp,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.fillMaxWidth().clickable { open = !open })
        if (open) content()
    }
}

/** 수량 표시 — 소수점 보유(미국 소수점 매매)는 필요한 자리까지만. */
private fun heldQtyLabel(q: Double): String =
    if (q == Math.floor(q)) "%,.0f주".format(q)
    else "%,.4f".format(q).trimEnd('0').trimEnd('.') + "주"

/** 분석 본문 Column 의 세로 패딩. 가용 높이 계산에서 정확히 이만큼 빠져야 한다. */
private val BODY_PAD = 8.dp

/** 컴포지션 중에 쓰고 레이아웃 뒤에 읽는 값 — 스냅샷 상태가 아니라 재구성을 유발하지 않는다. */
private class ChartsHeight(var total: Dp = 0.dp)

@Composable
private fun ResultView(r: Quant.Result, ticker: String, ohlc: List<Candle>, dayPct: Double?,
                       group: String, avail: Dp, minutes: List<Candle> = emptyList(),
                       trades: List<Trade> = emptyList(),
                       avgPrice: Double = Double.NaN, heldQty: Double = Double.NaN,
                       onNeedMinutes: () -> Unit = {}) {
    // ── 차트가 화면에 딱 맞게 — 여백을 **추정하지 않고 잰다** ──
    //
    // 종목 헤더·제목줄·날짜축·안내문의 높이를 dp 로 추정해 빼고 있었는데, 글꼴 크기나 줄바꿈으로
    // 조금만 어긋나도 세로 스크롤이 생겼다. 실제로 그려진 높이에서 차트 몫을 빼면 여백이 정확히
    // 나오고, 그 값은 차트 높이와 무관하므로 한 프레임 만에 수렴한다(진동하지 않는다).
    val density = LocalDensity.current
    // Dp.Unspecified 는 NaN 이라 == 비교가 항상 false 다 — 널로 둔다
    var chrome by remember(group) { mutableStateOf<Dp?>(null) }
    // 이번 컴포지션에서 차트에 준 높이 합. 스냅샷 상태로 두면 컴포지션 중 쓰기가 되므로 평범한 객체.
    val charts = remember { ChartsHeight() }
    fun chromeDp(fallback: Dp): Dp = chrome ?: fallback

    Column(
        Modifier.fillMaxWidth().onSizeChanged { sz ->
            val measured = with(density) { sz.height.toDp() } - charts.total
            val cur = chrome
            if (cur == null || kotlin.math.abs((measured - cur).value) > 0.5f) chrome = measured
        },
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
    // ── 종목명 + σ·β + (우측) 현재가/평단/수량 — 한 줄, 넘치면 가로 스크롤 ──
    // 매매기록·평단·수량은 **서버가 분석과 같이 준다**(증권사 값 우선). 화면이 따로 뒤지지 않는다.
    val held = avgPrice.isFinite() && avgPrice > 0
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
        val chgPct = if (held) (livePx / avgPrice - 1.0) * 100.0 else null
        Mini("현재가", Tickers.priceLabel(ticker, livePx),
            extra = chgPct?.let { "${if (it >= 0) "+" else ""}${"%.1f%%".format(it)}" },
            extraColor = chgPct?.let { if (it >= 0) Profit else Loss } ?: TextMuted)
        if (held) Mini("평단", Tickers.priceLabel(ticker, avgPrice))
        if (heldQty.isFinite() && heldQty > 0) Mini("보유", heldQtyLabel(heldQty))
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
    // ── 오늘 봉을 현재가로 키운다 ──
    // 장중에는 오늘 봉이 계속 자란다. 서버에서 일봉을 다시 받는 건 몇 분에 한 번이라
    // 그 사이 캔들이 멈춰 보였다. 현재가로 종가·고가·저가만 직접 늘린다 — 요청이 늘지 않고,
    // LivePrices 가 Compose 상태라 틱이 올 때마다 저절로 다시 그려진다.
    val tickPx = LivePrices.price(ticker)
    if (tickPx != null && wN > 0 && !closes[wN - 1].isNaN()) {
        closes[wN - 1] = tickPx
        if (tickPx > highs[wN - 1]) highs[wN - 1] = tickPx
        if (tickPx < lows[wN - 1]) lows[wN - 1] = tickPx
    }
    val macdW = seg(r.macd); val sigW = seg(r.macdSignal)
    val macdLast = macdW.lastOrNull { !it.isNaN() } ?: 0.0
    val sigLast = sigW.lastOrNull { !it.isNaN() } ?: 0.0
    val rsiLast = r.rsi.lastOrNull { !it.isNaN() } ?: 50.0
    val gh = 106.dp   // 그리드 차트 높이

    // ── 1분봉 ──
    // 토스가 받아 주는 주기는 1d·1m 뿐이다. 1분 모드에서는 Z·M 을 숨기고 가격 차트를 크게 쓴다
    // (Z·M·회귀·σ·β 는 SPY 대비 일봉 회귀라 분봉에 얹으면 숫자만 나오고 의미가 없다).
    var bar by remember { mutableStateOf(Store.barMode()) }
    LaunchedEffect(ticker, bar) { if (bar == "1m") onNeedMinutes() }
    val mN = minutes.size
    val mCloses = remember(minutes) { DoubleArray(mN) { minutes[it].close } }
    val mOpens = remember(minutes) { DoubleArray(mN) { minutes[it].open } }
    val mHighs = remember(minutes) { DoubleArray(mN) { minutes[it].high } }
    val mLows = remember(minutes) { DoubleArray(mN) { minutes[it].low } }
    val mDates = remember(minutes) { LongArray(mN) { minutes[it].t } }
    // MACD·RSI 는 종가만 있으면 되는 값이라 봉 주기대로 다시 계산한다
    val mMacd = remember(minutes) { Quant.macdOf(mCloses) }
    val mRsi = remember(minutes) { Quant.rsiOf(mCloses) }
    // 1분봉도 마지막 봉을 현재가로 키운다. remember 로 잡아 둔 배열을 고치면 값이
    // 쌓이므로 복사본에 얹는다 (봉 390개 복사는 무시할 만하다).
    val mClosesL = if (tickPx != null && mN > 0) mCloses.copyOf().also { it[mN - 1] = tickPx } else mCloses
    val mHighsL = if (tickPx != null && mN > 0) mHighs.copyOf().also { it[mN - 1] = maxOf(it[mN - 1], tickPx) } else mHighs
    val mLowsL = if (tickPx != null && mN > 0) mLows.copyOf().also { it[mN - 1] = minOf(it[mN - 1], tickPx) } else mLows
    val minMode = bar == "1m" && mN >= 2
    val empty = remember { DoubleArray(0) }

    // 확대 다이얼로그로 넘길 선택 차트 인덱스(-1=닫힘). 그리드/확대에서 동일 렌더 재사용.
    var zoom by remember(ticker) { mutableStateOf(-1) }
    // 확대 다이얼로그의 축 확대/이동 상태 — 차트를 바꾸면 원본 배율로 리셋
    var view by remember(ticker, zoom) { mutableStateOf(ChartView()) }
    // 길게 누른 지점(px). 떼면 null — 십자선과 시고저종 상자가 사라진다
    var insX by remember(ticker, zoom) { mutableStateOf<Float?>(null) }
    // 인덱스→확대 차트 렌더 (h=차트 높이, m=제스처 modifier). 0회귀 1Z·M궤적 2일봉 3Z·M 4MACD 5RSI
    val renderChart: @Composable (Int, androidx.compose.ui.unit.Dp, Modifier) -> Unit = { idx, h, m ->
        when (idx) {
            0 -> RegressionScatter(r.spyNorm, r.tickerNorm, r.predicted, r.bandUpper, r.bandLower, r.beta,
                markIdx = scatterIdx, height = h, view = view, zoomed = true, modifier = m)
            1 -> ZmScatter(r.zPct, r.mPct, scatterIdx, height = h, view = view, zoomed = true, modifier = m)
            2 -> if (minMode) {
                CandleChart(mOpens, mHighsL, mLowsL, mClosesL, empty, empty, empty,
                    currency = Tickers.currencySymbol(ticker), dates = mDates,
                    height = h, view = view, zoomed = true, inspectX = insX,
                    avgPrice = avgPrice, modifier = m)
                DateAxis(mDates, view, zoomed = true, timeOfDay = true)
            } else {
                if (closes.any { !it.isNaN() })
                    CandleChart(opens, highs, lows, closes, segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                        markers = priceMarks, currency = Tickers.currencySymbol(ticker), dates = dates,
                        dailyChgPct = dayPct ?: Double.NaN, height = h, view = view, zoomed = true,
                        inspectX = insX, avgPrice = avgPrice, modifier = m)
                else PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                    markers = priceMarks, currency = Tickers.currencySymbol(ticker),
                    height = h, view = view, zoomed = true, inspectX = insX, modifier = m)
                DateAxis(dates, view, zoomed = true)
            }
            3 -> { ZmChart(seg(r.zPct), seg(r.mPct), zmMarks, height = h, view = view, zoomed = true, inspectX = insX, modifier = m); DateAxis(dates, view, zoomed = true) }
            4 -> if (minMode) {
                MacdChart(mMacd.first, mMacd.second, height = h, view = view, zoomed = true,
                    inspectX = insX, modifier = m)
                DateAxis(mDates, view, zoomed = true, timeOfDay = true)
            } else {
                MacdChart(macdW, sigW, height = h, view = view, zoomed = true, inspectX = insX, modifier = m)
                DateAxis(dates, view, zoomed = true)
            }
            else -> if (minMode) {
                RsiChart(mRsi, height = h, view = view, zoomed = true, inspectX = insX, modifier = m)
                DateAxis(mDates, view, zoomed = true, timeOfDay = true)
            } else {
                RsiChart(seg(r.rsi), height = h, view = view, zoomed = true, inspectX = insX, modifier = m)
                DateAxis(dates, view, zoomed = true)
            }
        }
    }

    if (group == "scatter") {
        // ── 산점도 2개 — 전체 폭으로 위아래 배치 (반폭일 땐 점이 뭉개졌다) ──
        val sc = ((avail - chromeDp(118.dp)) / 2).coerceAtLeast(140.dp)
        charts.total = sc * 2
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
        // ── 시계열 — x축을 통일해 세로로 쌓는다 ──
        // ChartView 를 하나만 두고 전부 공유한다. 아무 차트에서나 핀치·드래그하면 같이 움직이고,
        // 날짜축은 공통이므로 맨 아래 한 번만 그린다.
        // [시계열] 가격·일봉 + Z·M / [보조지표] MACD + RSI — 넷을 한 화면에 넣으면
        // 하나하나가 너무 낮아 읽기 어려워 둘씩 나눴다.
        val sub = group == "sub"
        val totalMonths = ((dates.last() - dates.first()).toDouble() / 2_629_746.0).coerceAtLeast(1.0)
        // 사용자가 맞춘 구간은 ChartRange 가 들고 있다 — 종목을 바꿔도, 앱을 껐다 켜도 유지된다.
        // 종목마다 전체 기간이 다르므로 "보이는 개월 수"를 이 종목의 배율로 환산해 받는다.
        var sView by remember(ticker, n) { mutableStateOf(ChartRange.viewFor(totalMonths).snappedX(n)) }
        // 좌우 이동은 봉 1개 단위로 끊는다
        fun pushView(v: ChartView) {
            val snapped = v.snappedX(n)
            sView = snapped
            ChartRange.save(snapped, totalMonths)
        }
        // 화면을 벗어날 때 마지막 값을 파일에 확정 (제스처 중에는 throttle 되어 있다)
        DisposableEffect(ticker) { onDispose { ChartRange.flush() } }
        // 헤더줄 2개 + 날짜축 + 안내문을 뺀 나머지를 둘로 나눈다
        val body = (avail - chromeDp(120.dp)).coerceAtLeast(200.dp)
        val sh = (body / 2f).coerceAtLeast(100.dp)
        charts.total = sh * 2
        val gest = Modifier.chartGestures(sView, { pushView(it) }, xOnly = true,
            onInspect = { insX = it })
        // 1분봉은 구간 저장(개월 단위)과 기준이 달라 뷰를 따로 둔다 — 처음엔 전체(최근 390봉)
        var mView by remember(ticker) { mutableStateOf(ChartView()) }
        val mGest = Modifier.chartGestures(mView, { mView = it }, xOnly = true,
            onInspect = { insX = it })

        // 간격 없는 Column — 바깥 Column 의 spacedBy(8dp) 가 항목마다 붙으면
        // 제목줄·차트 사이가 벌어져 한 화면에 안 들어간다
        Column(Modifier.fillMaxWidth()) {
            if (!sub) {
                BarHeader(bar, { bar = it; Store.setBarMode(it) },
                    Tickers.priceLabel(ticker, LivePrices.price(ticker) ?: r.lastPrice)) { zoom = 2 }
                if (minMode) {
                    // Z·M 을 숨긴 만큼 가격 차트가 본문 전체를 쓴다
                    CandleChart(mOpens, mHighsL, mLowsL, mClosesL, empty, empty, empty,
                        currency = Tickers.currencySymbol(ticker), dates = mDates,
                        height = sh * 2, view = mView, zoomed = true, inspectX = insX,
                        avgPrice = avgPrice, modifier = mGest)
                } else if (bar == "1m") {
                    // 아직 안 받았거나 받지 못한 상태 — 일봉을 대신 보여주면 더 헷갈린다
                    Box(Modifier.fillMaxWidth().height(sh * 2), contentAlignment = Alignment.Center) {
                        Text(if (minutes.isEmpty()) "1분봉 불러오는 중…" else "1분봉이 없습니다",
                            color = TextMuted, fontSize = 12.sp)
                    }
                } else if (closes.any { !it.isNaN() }) {
                    CandleChart(opens, highs, lows, closes,
                        segDollar(r.predicted), segDollar(r.bandUpper), segDollar(r.bandLower),
                        markers = priceMarks, currency = Tickers.currencySymbol(ticker), topLabel = "",
                        dates = dates, dailyChgPct = dayPct ?: Double.NaN,
                        height = sh, view = sView, zoomed = true, inspectX = insX,
                        avgPrice = avgPrice, modifier = gest)
                } else {
                    PriceChart(segDollar(r.tickerNorm), segDollar(r.predicted), segDollar(r.bandUpper),
                        segDollar(r.bandLower), markers = priceMarks,
                        currency = Tickers.currencySymbol(ticker),
                        height = sh, view = sView, zoomed = true, inspectX = insX, modifier = gest)
                }

                if (!minMode) {
                    SeriesHeader("Z·M", "Z${"%.0f".format(r.lastZpct)}·M${"%.0f".format(r.lastMpct)}", TextPrimary) { zoom = 3 }
                    ZmChart(seg(r.zPct), seg(r.mPct), zmMarks, height = sh, view = sView, zoomed = true,
                        inspectX = insX, modifier = gest)
                }
            } else if (minMode) {
                SeriesHeader("MACD · 1분", "%.2f".format(mMacd.first.lastOrNull() ?: 0.0), TextPrimary) { zoom = 4 }
                MacdChart(mMacd.first, mMacd.second, height = sh, view = mView, zoomed = true,
                    inspectX = insX, modifier = mGest)

                SeriesHeader("RSI · 1분", "%.1f".format(mRsi.lastOrNull { !it.isNaN() } ?: 50.0), Teal) { zoom = 5 }
                RsiChart(mRsi, height = sh, view = mView, zoomed = true, inspectX = insX, modifier = mGest)
            } else {
                SeriesHeader("MACD", "${"%.2f".format(macdLast)}(${"%+.2f".format(macdLast - sigLast)})", TextPrimary) { zoom = 4 }
                MacdChart(macdW, sigW, height = sh, view = sView, zoomed = true, inspectX = insX, modifier = gest)

                SeriesHeader("RSI", "%.1f".format(rsiLast), Teal) { zoom = 5 }
                RsiChart(seg(r.rsi), height = sh, view = sView, zoomed = true, inspectX = insX, modifier = gest)
            }

            if (minMode) {
                DateAxis(mDates, mView, zoomed = true, timeOfDay = true)
                Text(
                    if (mView.isIdentity) "1분봉 ${mN}개" else "1분봉 ${(mN / mView.sx).toInt()}개 — 탭하면 전체",
                    color = TextMuted, fontSize = 11.sp,
                    modifier = Modifier.fillMaxWidth().clickable { mView = ChartView() },
                )
            } else {
                DateAxis(dates, sView, zoomed = true)
                Text(
                    if (sView.isIdentity) "전체 ${"%.1f".format(totalMonths)}개월"
                    else "${"%.1f".format(totalMonths / sView.sx)}개월 — 탭하면 전체",
                    color = TextMuted, fontSize = 11.sp,
                    modifier = Modifier.fillMaxWidth().clickable { pushView(ChartView()) },
                )
            }
        }
    }

    // ── 차트 확대 다이얼로그 (그리드에서 탭한 차트를 전체화면 크게) ──
    // Dialog 는 별도 창이라 측정 Column 의 높이에는 잡히지 않는다 — 안에 둬도 무방하다.
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
                                Modifier.chartGestures(view, { view = it }, onTap = { zoom = -1 }, xOnly = xOnly,
                                    onInspect = { insX = it }))
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
    }   // 측정 Column 끝
}

/** 가격 차트 헤더 — 제목 자리에 봉 주기 세그먼트(일봉/1분). 우측은 현재가 + 확대. */
@Composable
private fun BarHeader(bar: String, onBar: (String) -> Unit, value: String, onZoom: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(top = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        UnderlineSegments(
            listOf("1d" to "일봉", "1m" to "1분"),
            selected = bar, onSelect = onBar, fontSize = 11,
        )
        Spacer(Modifier.weight(1f))
        Text(value, color = TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Bold,
            fontFamily = Mono, maxLines = 1)
        Text(" ⤢", color = TextMuted, fontSize = 11.sp, modifier = Modifier.clickable { onZoom() })
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
        modifier.then(if (onClick != null) Modifier.clickable { onClick() } else Modifier)
            .padding(vertical = 4.dp),
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
