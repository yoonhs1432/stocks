package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.OverviewRepo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

typealias CompareRow = OverviewRepo.Row

enum class SortKey { M, Z, DAY, WEEK, FROM_HIGH, PRICE, NAME, BETA, SIGMA }

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

    fun load(force: Boolean = false) {
        state = state.copy(loading = true, error = null)
        viewModelScope.launch {
            val rows = withContext(Dispatchers.IO) { OverviewRepo.load(force) }
            state = if (rows.isEmpty()) state.copy(loading = false, error = "시세를 가져오지 못했습니다")
            else state.copy(loading = false, rows = rows, error = null)
        }
    }

    fun setSort(key: SortKey) {
        val desc = if (state.sortKey == key) !state.sortDesc else false
        state = state.copy(sortKey = key, sortDesc = desc)
    }

    fun sorted(): List<CompareRow> {
        val base = when (state.sortKey) {
            SortKey.NAME -> state.rows.sortedBy { it.name }
            SortKey.PRICE -> state.rows.sortedBy { it.price }
            SortKey.DAY -> state.rows.sortedBy { it.day }
            SortKey.WEEK -> state.rows.sortedBy { it.week }
            SortKey.FROM_HIGH -> state.rows.sortedBy { it.fromHigh }
            SortKey.Z -> state.rows.sortedBy { it.zPct }
            SortKey.M -> state.rows.sortedBy { it.mPct }
            SortKey.BETA -> state.rows.sortedBy { it.beta }
            SortKey.SIGMA -> state.rows.sortedBy { it.sigmaPct }
        }
        return if (state.sortDesc) base.reversed() else base
    }
}
