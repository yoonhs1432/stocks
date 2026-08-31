package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.Quotes
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.Yahoo
import com.quant.dashboard.quant.Portfolio
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class PortfolioState(
    val loading: Boolean = false,
    val empty: Boolean = false,
    val result: Portfolio.Result? = null,
    val rate: Double = 1400.0,   // USD/KRW
)

class PortfolioViewModel : ViewModel() {
    var state by mutableStateOf(PortfolioState())
        private set

    private var loadedVersion = -1

    /** AppState.dataVersion 변경(기준일·설정) 시 재로드, 아니면 최초 1회만. */
    fun sync(version: Int) {
        if (version != loadedVersion) {
            loadedVersion = version
            load()
        } else if (state.result == null && !state.empty && !state.loading) load()
    }

    fun load() {
        state = state.copy(loading = true)
        viewModelScope.launch {
            val pair = withContext(Dispatchers.IO) {
                val trades = Store.loadTrades()
                if (trades.isEmpty()) return@withContext null
                val tickers = trades.keys.toList()
                val months = Store.lookbackMonths()
                // 시세 소스 라우팅(토스↔Yahoo)을 타도록 Quotes 경유 — 예전엔 Yahoo 직접 호출이었다
                val series = tickers.map { tk ->
                    async { tk to Store.sliceAsof(Quotes.closes(tk, months)) }
                }.awaitAll().toMap()
                val hist = series.filterValues { it.isNotEmpty() }
                val lastClose = hist.mapValues { (_, v) -> v.last().second }
                val rate = Store.sliceAsof(Yahoo.closes("KRW=X", "1mo")).lastOrNull()?.second ?: 1400.0
                val res = Portfolio.compute(
                    trades = trades,
                    name = { Tickers.displayName(it) },
                    lastClose = lastClose,
                    hist = hist,
                    seed = Store.seedUsd(),
                )
                res to rate
            }
            state = if (pair == null) PortfolioState(loading = false, empty = true)
            else PortfolioState(loading = false, result = pair.first, rate = pair.second)
        }
    }
}
