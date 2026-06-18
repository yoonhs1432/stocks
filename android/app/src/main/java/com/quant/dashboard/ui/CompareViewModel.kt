package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.Yahoo
import com.quant.dashboard.quant.Quant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class CompareRow(
    val ticker: String,
    val name: String,
    val price: Double,
    val day: Double,
    val week: Double,
    val fromHigh: Double,
    val zPct: Double,
    val mPct: Double,
    val signal: String,
)

enum class SortKey { M, Z, DAY, WEEK, FROM_HIGH, PRICE, NAME }

data class CompareState(
    val loading: Boolean = false,
    val error: String? = null,
    val rows: List<CompareRow> = emptyList(),
    val sortKey: SortKey = SortKey.M,
    val sortDesc: Boolean = false,
)

class CompareViewModel : ViewModel() {
    var state by mutableStateOf(CompareState())
        private set

    fun loadIfEmpty() {
        if (state.rows.isEmpty() && !state.loading) load()
    }

    fun load() {
        state = state.copy(loading = true, error = null)
        viewModelScope.launch {
            val rows = withContext(Dispatchers.IO) {
                val spy = Yahoo.closes(Tickers.BASE)
                if (spy.isEmpty()) return@withContext null
                Store.loadTickers().map { tk ->
                    async {
                        val r = Quant.analyze(spy, Yahoo.closes(tk)) ?: return@async null
                        val p = r.price; val m = p.size
                        if (m < 2) return@async null
                        val prevD = p[m - 2]
                        val prevW = if (m > 5) p[m - 6] else prevD
                        val high = p.max()
                        CompareRow(
                            ticker = tk, name = Tickers.displayName(tk), price = p[m - 1],
                            day = if (prevD > 0) (p[m - 1] / prevD - 1) * 100 else 0.0,
                            week = if (prevW > 0) (p[m - 1] / prevW - 1) * 100 else 0.0,
                            fromHigh = if (high > 0) (p[m - 1] / high - 1) * 100 else 0.0,
                            zPct = r.lastZpct, mPct = r.lastMpct, signal = r.signal,
                        )
                    }
                }.awaitAll().filterNotNull()
            }
            state = if (rows == null) state.copy(loading = false, error = "시세를 가져오지 못했습니다")
            else state.copy(loading = false, rows = rows, error = null)
        }
    }

    fun setSort(key: SortKey) {
        val desc = if (state.sortKey == key) !state.sortDesc else false
        state = state.copy(sortKey = key, sortDesc = desc)
    }

    fun sorted(): List<CompareRow> {
        val sel = { r: CompareRow ->
            when (state.sortKey) {
                SortKey.M -> r.mPct; SortKey.Z -> r.zPct; SortKey.DAY -> r.day
                SortKey.WEEK -> r.week; SortKey.FROM_HIGH -> r.fromHigh; SortKey.PRICE -> r.price
                SortKey.NAME -> 0.0
            }
        }
        val base = if (state.sortKey == SortKey.NAME) state.rows.sortedBy { it.name }
        else state.rows.sortedBy { sel(it) }
        return if (state.sortDesc) base.reversed() else base
    }
}
