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
    val sortKey: SortKey = SortKey.DAY,
    val sortDesc: Boolean = true,
)

class CompareViewModel : ViewModel() {
    var state by mutableStateOf(CompareState())
        private set

    private var loadedVersion = -1

    /** AppState.dataVersion 변경(기준일·설정) 시 강제 재로드, 아니면 최초 1회만. */
    fun sync(version: Int) {
        if (version != loadedVersion) {
            loadedVersion = version
            load(force = true)
        } else loadIfEmpty()
    }

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

    /** 자동(조용한) 새로고침 — 로딩 표시 없이 명단 갱신(5분 캐시 만료 시에만 실제 재요청). */
    fun autoRefresh() {
        viewModelScope.launch {
            val rows = withContext(Dispatchers.IO) { OverviewRepo.load(false) }
            if (rows.isNotEmpty()) state = state.copy(rows = rows, error = null)
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
