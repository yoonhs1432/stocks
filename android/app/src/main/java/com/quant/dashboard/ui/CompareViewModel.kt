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
    val holdingsOnly: Boolean = false,
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

    /** 보유종목만 보기 토글 (탭 전환에도 유지되도록 VM에 저장). */
    fun toggleHoldings() {
        state = state.copy(holdingsOnly = !state.holdingsOnly)
    }

    fun setSort(key: SortKey) {
        val desc = if (state.sortKey == key) !state.sortDesc else false
        state = state.copy(sortKey = key, sortDesc = desc)
    }

    /** 보유 필터 적용된 표시 대상 행. */
    fun visibleRows(): List<CompareRow> =
        if (state.holdingsOnly) state.rows.filter { it.holding } else state.rows

    fun sorted(): List<CompareRow> {
        val src = visibleRows()
        val base = when (state.sortKey) {
            SortKey.NAME -> src.sortedBy { it.name }
            SortKey.PRICE -> src.sortedBy { it.price }
            SortKey.DAY -> src.sortedBy { it.day }
            SortKey.WEEK -> src.sortedBy { it.week }
            SortKey.FROM_HIGH -> src.sortedBy { it.fromHigh }
            SortKey.Z -> src.sortedBy { it.zPct }
            SortKey.M -> src.sortedBy { it.mPct }
            SortKey.BETA -> src.sortedBy { it.beta }
            SortKey.SIGMA -> src.sortedBy { it.sigmaPct }
        }
        return if (state.sortDesc) base.reversed() else base
    }
}
