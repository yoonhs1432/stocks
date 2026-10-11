package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.AnalysisCache
import com.quant.dashboard.data.Candle
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.Server
import com.quant.dashboard.data.Trade
import com.quant.dashboard.quant.Quant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class UiState(
    val ticker: String = Tickers.DEFAULT.first(),
    val loading: Boolean = false,
    val error: String? = null,
    val result: Quant.Result? = null,
    val ohlc: List<Candle> = emptyList(),
    /** 1분봉 — 봉 주기를 1분으로 놓았을 때만 받는다(일봉 분석과 완전히 별개). */
    val minutes: List<Candle> = emptyList(),
    val minLoading: Boolean = false,
    // 아래 셋은 서버가 분석과 **같이** 준다. 화면이 따로 뒤지지 않아도 되게.
    val trades: List<Trade> = emptyList(),
    val avgPrice: Double = Double.NaN,
    val qty: Double = Double.NaN,
    /** 화면의 분석이 **서버에서 온 시각**(ms). 지난 값을 먼저 그렸을 때 얼마나 묵었는지 알리는 데 쓴다. */
    val asOf: Long = 0L,
)

/** 받아 둔 분석 한 벌을 화면 상태로. 종목이 바뀌는 순간에도, 새 응답이 왔을 때도 같은 모양으로 담는다. */
private fun UiState.with(a: Server.Analysis, ticker: String, at: Long) = copy(
    ticker = ticker, result = a.result, ohlc = a.candles,
    trades = a.trades, avgPrice = a.avgPrice, qty = a.qty, asOf = at,
)

class AnalysisViewModel : ViewModel() {
    var state by mutableStateOf(
        UiState(ticker = Store.loadTickers().firstOrNull() ?: Tickers.DEFAULT.first())
    )
        private set

    var overview by mutableStateOf<List<OverviewRepo.Row>>(emptyList())
        private set

    fun loadOverview(force: Boolean = false) {
        viewModelScope.launch {
            val rows = OverviewRepo.load(force)
            if (rows.isNotEmpty()) overview = rows
        }
    }

    private var loadedVersion = -1

    private companion object {
        /** 이 안에 받은 분석은 다시 묻지 않는다 — 서버도 그 사이엔 같은 값을 준다. */
        const val QUICK_MS = 20_000L
    }
    private var loadJob: Job? = null
    private var reqSeq = 0

    /**
     * AppState 변경 반영 — pending(비교/포트폴리오 탭에서 넘어온 종목)이 있으면 그 종목을 우선 로드.
     * 진입점을 하나로 합친 이유: 예전엔 dataVersion 효과와 pendingTicker 효과가 각각 load()를 걸어,
     * 늦게 끝난 이전 종목 응답이 새 종목 화면을 덮어쓰는 경우가 있었음.
     */
    fun sync(version: Int, pending: String? = null) {
        val changed = version != loadedVersion
        if (changed) loadedVersion = version
        when {
            pending != null -> select(pending)
            changed -> load()
            state.result == null && !state.loading -> load()
        }
        // ⚠️ 여기서 force 를 주면 **비교 탭의 20~30초짜리 첫 조회가 한 번 더** 돈다.
        //    칩 줄에 쓰는 값이라 캐시로 충분하다.
        if (changed || overview.isEmpty()) loadOverview(false)
    }

    fun select(ticker: String) {
        if (ticker == state.ticker) { load(ticker); return }
        // 받아 둔 게 있으면 **그 종목 자신의** 지난 분석을 바로 그리고, 조용히 새로 받는다.
        // 없으면 예전처럼 비운다 — 로드 중 이전 종목 차트가 남아 보이는 문제 방지.
        val hit = AnalysisCache.peek(ticker)
        state = if (hit != null) {
            state.with(hit.a, ticker, hit.at).copy(minutes = emptyList(), error = null, loading = false)
        } else {
            state.copy(ticker = ticker, result = null, ohlc = emptyList(),
                minutes = emptyList(), error = null, asOf = 0L)
        }
        // 방금 받은 것이면 다시 묻지 않는다 — 종목을 빠르게 넘겨 볼 때 요청이 쌓이지 않게
        if (hit != null && System.currentTimeMillis() - hit.at < QUICK_MS) {
            loadJob?.cancel(); ++reqSeq
            return
        }
        load(ticker, quiet = hit != null)
    }

    /**
     * 비교 표 순서대로 앞·뒤 종목으로. [order] 는 화면에 보이는 정렬 그대로다.
     * 지금 종목이 목록에 없으면(직접 입력) 첫 종목으로 간다.
     */
    fun step(order: List<String>, delta: Int) {
        if (order.isEmpty()) return
        val i = order.indexOf(state.ticker)
        val next = if (i < 0) order.first() else order[(i + delta + order.size) % order.size]
        if (next != state.ticker) select(next)
    }

    /** 자동(조용한) 새로고침 — 로딩 인디케이터 없이 현재 종목 갱신. */
    fun autoRefresh() {
        if (state.loading) return   // 사용자가 띄운 로드를 가로채지 않음
        load(quiet = true)
        loadOverview(false)   // 5분 캐시: 만료 시에만 실제 재요청
    }

    /**
     * 당겨서 새로고침 — 시세를 **실제로 다시 받는다**.
     *
     * 그냥 다시 그리기만 하면 일봉 캐시(6시간) 때문에 아무것도 안 바뀐다.
     * 자동 새로고침과 달리 사용자가 명시적으로 요청한 것이므로 캐시를 건너뛴다.
     */
    fun refresh() {
        load(force = true)
        loadOverview(force = true)
        if (state.minutes.isNotEmpty()) loadMinutes(force = true)
    }

    private var minJob: Job? = null

    /**
     * 1분봉 로드 — 분석 탭에서 봉 주기를 1분으로 놨을 때만 부른다.
     * 비교 탭처럼 여러 종목을 한꺼번에 받으면 429 가 나므로 **보고 있는 종목만** 받는다.
     */
    fun loadMinutes(ticker: String = state.ticker, force: Boolean = false) {
        minJob?.cancel()
        minJob = viewModelScope.launch {
            state = state.copy(minLoading = true)
            val bars = withContext(Dispatchers.IO) {
                runCatching { Server.minutes(ticker) }.getOrDefault(emptyList())
            }
            if (ticker != state.ticker) return@launch   // 그 사이 종목이 바뀌었으면 버린다
            state = state.copy(minutes = bars, minLoading = false)
        }
    }

    fun load(ticker: String = state.ticker, quiet: Boolean = false, force: Boolean = false) {
        val seq = ++reqSeq
        loadJob?.cancel()   // 진행 중이던 이전 요청 취소 (응답 순서가 뒤바뀌는 것 방지)
        if (!quiet) state = state.copy(loading = true, error = null)
        loadJob = viewModelScope.launch {
            // 분석은 **서버가 계산해서** 준다. 폰이 일봉을 받아 회귀를 돌리던 때와 달리
            // 요청 한 번이고, 종목을 눌렀을 때 이미 계산이 끝나 있다(서버가 미리 해 둔다).
            // 보여 줄 게 아직 없으면 **파일에 남겨 둔 지난 분석**부터 그린다(수십 ms).
            // 앱을 새로 켠 직후나, 미리 받아 두기만 하고 아직 안 열어 본 종목이 여기에 걸린다.
            if (state.result == null && !force) {
                val old = withContext(Dispatchers.IO) { runCatching { AnalysisCache.get(ticker) }.getOrNull() }
                if (seq != reqSeq) return@launch
                if (old != null) state = state.with(old.a, ticker, old.at).copy(loading = false, error = null)
            }
            val holder = withContext(Dispatchers.IO) {
                runCatching { AnalysisCache.fetch(ticker, force) }
            }
            if (seq != reqSeq) return@launch   // 더 최신 요청이 있으면 이 응답은 폐기
            val a = holder.getOrNull()
            val now = System.currentTimeMillis()
            state = when {
                a == null && quiet -> state.copy(loading = false)
                // 지난 값이라도 그려져 있으면 차트는 그대로 두고, 왜 못 받았는지만 알린다
                a == null -> state.copy(
                    loading = false,
                    error = holder.exceptionOrNull()?.message ?: "PC 에 연결되지 않습니다",
                )
                a.result == null -> state.copy(
                    loading = false, ticker = ticker, result = null, ohlc = a.candles,
                    trades = a.trades, avgPrice = a.avgPrice, qty = a.qty, asOf = now,
                    error = "분석 데이터 부족 — 상장 후 기간이 짧은 종목입니다",
                )
                else -> state.with(a, ticker, now).copy(loading = false, error = null)
            }
        }
    }
}
