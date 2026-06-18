package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
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
)

class PortfolioViewModel : ViewModel() {
    var state by mutableStateOf(PortfolioState())
        private set

    fun load() {
        state = state.copy(loading = true)
        viewModelScope.launch {
            val res = withContext(Dispatchers.IO) {
                val trades = Store.loadTrades()
                if (trades.isEmpty()) return@withContext null
                val tickers = trades.keys.toList()
                val series = tickers.map { tk ->
                    async { tk to Yahoo.closes(tk) }
                }.awaitAll().toMap()
                val hist = series.filterValues { it.isNotEmpty() }
                val lastClose = hist.mapValues { (_, v) -> v.last().second }
                Portfolio.compute(
                    trades = trades,
                    name = { Tickers.displayName(it) },
                    lastClose = lastClose,
                    hist = hist,
                )
            }
            state = if (res == null) PortfolioState(loading = false, empty = true)
            else PortfolioState(loading = false, result = res)
        }
    }
}
