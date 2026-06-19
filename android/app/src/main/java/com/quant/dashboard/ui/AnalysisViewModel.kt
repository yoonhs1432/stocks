package com.quant.dashboard.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.Yahoo
import com.quant.dashboard.quant.Quant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

data class UiState(
    val ticker: String = Tickers.DEFAULT.first(),
    val loading: Boolean = false,
    val error: String? = null,
    val result: Quant.Result? = null,
)

class AnalysisViewModel : ViewModel() {
    var state by mutableStateOf(
        UiState(ticker = Store.loadTickers().firstOrNull() ?: Tickers.DEFAULT.first())
    )
        private set

    // 종목 버튼용 전 종목 요약 (모멘텀 색/정렬)
    var overview by mutableStateOf<List<OverviewRepo.Row>>(emptyList())
        private set

    fun loadOverview(force: Boolean = false) {
        viewModelScope.launch {
            val rows = withContext(Dispatchers.IO) { OverviewRepo.load(force) }
            if (rows.isNotEmpty()) overview = rows
        }
    }

    // SPY 시세는 종목 간 재사용 (캐시)
    private var spyCache: List<Pair<Long, Double>> = emptyList()

    fun select(ticker: String) {
        if (ticker != state.ticker) state = state.copy(ticker = ticker)
        load(ticker)
    }

    fun load(ticker: String = state.ticker) {
        state = state.copy(loading = true, error = null)
        viewModelScope.launch {
            val res = withContext(Dispatchers.IO) {
                try {
                    if (spyCache.isEmpty()) spyCache = Yahoo.closes(Tickers.BASE)
                    val spy = spyCache
                    val tk = Yahoo.closes(ticker)
                    when {
                        spy.isEmpty() -> Result.failure(Exception("SPY 시세를 가져오지 못했습니다"))
                        tk.isEmpty() -> Result.failure(Exception("$ticker 시세를 가져오지 못했습니다"))
                        else -> {
                            val r = Quant.analyze(spy, tk)
                            if (r == null) Result.failure(Exception("분석 데이터 부족"))
                            else Result.success(r)
                        }
                    }
                } catch (e: Exception) {
                    Result.failure(e)
                }
            }
            state = if (res.isSuccess) {
                state.copy(loading = false, result = res.getOrNull(), error = null)
            } else {
                state.copy(loading = false, error = res.exceptionOrNull()?.message ?: "오류")
            }
        }
    }

    fun refresh() {
        spyCache = emptyList()
        load()
    }
}
