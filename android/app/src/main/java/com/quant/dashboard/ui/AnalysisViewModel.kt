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
            if (rows.isNotEmpty()) overview = rows
        }
    }

    private var spyCache: List<Pair<Long, Double>> = emptyList()
    private var loadedVersion = -1

    /** AppState.dataVersion 변경(기준일·설정) 시 재로드, 아니면 최초 1회만. */
    fun sync(version: Int) {
        if (version != loadedVersion) {
            loadedVersion = version
            refresh()
        } else if (state.result == null && !state.loading) {
            load()
            if (overview.isEmpty()) loadOverview()
        }
    }

    fun select(ticker: String) {
        if (ticker != state.ticker) state = state.copy(ticker = ticker)
        load(ticker)
    }

    /** 자동(조용한) 새로고침 — 로딩 인디케이터 없이 현재 종목 갱신. */
    fun autoRefresh() {
        load(quiet = true)
        loadOverview(false)   // 5분 캐시: 만료 시에만 실제 재요청
    }

    fun load(ticker: String = state.ticker, quiet: Boolean = false) {
        if (!quiet) state = state.copy(loading = true, error = null)
        viewModelScope.launch {
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
            state = if (holder.isSuccess) {
                val (r, candles) = holder.getOrNull()!!
                state.copy(loading = false, result = r, ohlc = candles, error = null)
            } else if (quiet) {
                state   // 조용한 새로고침 실패는 기존 화면 유지
            } else {
                state.copy(loading = false, error = holder.exceptionOrNull()?.message ?: "오류")
            }
        }
    }

    fun refresh() {
        spyCache = emptyList()
        load()
        loadOverview(true)
    }
}
