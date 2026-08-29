package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.Candle
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.Yahoo
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
            val rows = withContext(Dispatchers.IO) { OverviewRepo.load(force) }
            // 비교 탭에서 미장 TOP30을 열어 뒀다면 그 캐시도 합친다 —
            // 워치리스트 밖 종목으로 넘어와도 헤더 일간 등락이 보이게 (추가 요청 없음)
            val merged = rows + OverviewRepo.cachedTop().filter { t -> rows.none { it.ticker == t.ticker } }
            if (merged.isNotEmpty()) overview = merged
        }
    }

    private var spyCache: List<Pair<Long, Double>> = emptyList()
    private var loadedVersion = -1
    private var loadJob: Job? = null
    private var reqSeq = 0

    /**
     * AppState 변경 반영 — pending(비교/포트폴리오 탭에서 넘어온 종목)이 있으면 그 종목을 우선 로드.
     * 진입점을 하나로 합친 이유: 예전엔 dataVersion 효과와 pendingTicker 효과가 각각 load()를 걸어,
     * 늦게 끝난 이전 종목 응답이 새 종목 화면을 덮어쓰는 경우가 있었음.
     */
    fun sync(version: Int, pending: String? = null) {
        val changed = version != loadedVersion
        if (changed) { loadedVersion = version; spyCache = emptyList() }
        when {
            pending != null -> select(pending)
            changed -> load()
            state.result == null && !state.loading -> load()
        }
        if (changed || overview.isEmpty()) loadOverview(changed)
    }

    fun select(ticker: String) {
        // 종목이 바뀌면 이전 종목 결과를 즉시 비움 — 로드 중 이전 종목 차트가 남아 보이는 문제 방지
        if (ticker != state.ticker) {
            state = state.copy(ticker = ticker, result = null, ohlc = emptyList(), error = null)
        }
        load(ticker)
    }

    /** 자동(조용한) 새로고침 — 로딩 인디케이터 없이 현재 종목 갱신. */
    fun autoRefresh() {
        if (state.loading) return   // 사용자가 띄운 로드를 가로채지 않음
        load(quiet = true)
        loadOverview(false)   // 5분 캐시: 만료 시에만 실제 재요청
    }

    fun load(ticker: String = state.ticker, quiet: Boolean = false) {
        val seq = ++reqSeq
        loadJob?.cancel()   // 진행 중이던 이전 요청 취소 (응답 순서가 뒤바뀌는 것 방지)
        if (!quiet) state = state.copy(loading = true, error = null)
        loadJob = viewModelScope.launch {
            val range = Store.lookbackRange()
            val interval = Store.candleInterval()
            val holder = withContext(Dispatchers.IO) {
                try {
                    if (spyCache.isEmpty()) spyCache = Yahoo.closes(Tickers.BASE, range, interval)
                    val spy = Store.sliceAsof(spyCache)
                    val candles = Store.sliceAsofCandles(Yahoo.ohlc(ticker, range, interval))
                    val tk = candles.map { Pair(it.t, it.close) }
                    when {
                        spy.isEmpty() -> Result.failure(Exception("SPY 시세를 가져오지 못했습니다"))
                        tk.isEmpty() -> Result.failure(Exception("$ticker 시세를 가져오지 못했습니다"))
                        else -> {
                            val r = Quant.analyze(spy, tk)
                            if (r == null) Result.failure(Exception("분석 데이터 부족"))
                            else Result.success(Pair(r, candles))
                        }
                    }
                } catch (e: Exception) {
                    Result.failure(e)
                }
            }
            if (seq != reqSeq) return@launch   // 더 최신 요청이 있으면 이 응답은 폐기
            state = if (holder.isSuccess) {
                val (r, candles) = holder.getOrNull()!!
                state.copy(loading = false, ticker = ticker, result = r, ohlc = candles, error = null)
            } else if (quiet) {
                state.copy(loading = false)   // 조용한 새로고침 실패는 기존 화면 유지
            } else {
                state.copy(loading = false, error = holder.exceptionOrNull()?.message ?: "오류")
            }
        }
    }
}
