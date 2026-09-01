package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.Rankings
import com.quant.dashboard.data.Store
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

typealias CompareRow = OverviewRepo.Row

/** RANK = 목록 원래 순서(미장 TOP에서는 랭킹 순위). */
enum class SortKey { M, Z, DAY, WEEK, FROM_HIGH, PRICE, NAME, BETA, SIGMA, RANK }

data class CompareState(
    val loading: Boolean = false,
    val error: String? = null,
    val rows: List<CompareRow> = emptyList(),
    val sortKey: SortKey = SortKey.DAY,
    val sortDesc: Boolean = true,
    val holdingsOnly: Boolean = false,
    // ── 미장 TOP 목록 (버튼으로 전환, 열 때만 로드) ──
    val showTop: Boolean = false,
    val topRows: List<CompareRow> = emptyList(),
    val topLoading: Boolean = false,
    val topError: String? = null,
    // 랭킹 기준 (토스 /rankings). 미연동이면 하드코딩 목록으로 폴백되고 그 사유가 rankNote 에 담긴다.
    val rankMarket: String = "US",
    val rankType: String = "MARKET_TRADING_AMOUNT",
    val rankDuration: String = "1d",
    val rankNote: String? = null,
)

class CompareViewModel : ViewModel() {
    var state by mutableStateOf(
        CompareState(
            rankMarket = Store.rankMarket(),
            rankType = Store.rankType(),
            rankDuration = Store.rankDuration(),
        )
    )
        private set

    private var loadedVersion = -1

    /** AppState.dataVersion 변경(기준일·설정) 시 강제 재로드, 아니면 최초 1회만. */
    fun sync(version: Int) {
        if (version != loadedVersion) {
            loadedVersion = version
            load(force = true)
            if (state.showTop) loadTop(force = true)
        } else {
            loadIfEmpty()
            if (state.showTop && state.topRows.isEmpty() && !state.topLoading) loadTop()
        }
    }

    /** 시장 전환(미국 ↔ 국내) — 종목 자체가 바뀌므로 강제 재로드. */
    fun setRankMarket(m: String) {
        if (m == state.rankMarket) return
        Store.setRankMarket(m)
        state = state.copy(rankMarket = m, topRows = emptyList())
        loadTop(force = true)
    }

    /** 랭킹 기준 변경 — 종목 자체가 바뀌므로 강제 재로드. */
    fun setRankType(t: String) {
        if (t == state.rankType) return
        Store.setRankType(t)
        // 급상승·급하락은 realtime 을 지원하지 않으므로 1일로 보정
        var d = state.rankDuration
        if (t.startsWith("TOP_") && d == "realtime") { d = "1d"; Store.setRankDuration(d) }
        state = state.copy(rankType = t, rankDuration = d, topRows = emptyList())
        loadTop(force = true)
    }

    fun setRankDuration(d: String) {
        if (d == state.rankDuration) return
        Store.setRankDuration(d)
        state = state.copy(rankDuration = d, topRows = emptyList())
        loadTop(force = true)
    }

    /** 워치리스트 ↔ 미장 TOP 전환. TOP은 처음 열 때만 실제로 받아온다(요청 30건). */
    fun toggleTop() {
        val on = !state.showTop
        state = state.copy(
            showTop = on,
            // TOP에서 보유 필터가 켜져 있으면 거의 빈 목록이 되므로 해제
            holdingsOnly = if (on) false else state.holdingsOnly,
            // 기본 정렬: TOP은 랭킹 순, 워치리스트는 일간 등락
            sortKey = if (on) SortKey.RANK else SortKey.DAY,
            sortDesc = !on,
        )
        if (on && state.topRows.isEmpty() && !state.topLoading) loadTop()
    }

    private fun loadTop(force: Boolean = false) {
        state = state.copy(topLoading = true, topError = null)
        viewModelScope.launch {
            val rows = withContext(Dispatchers.IO) { OverviewRepo.loadTop(force) }
            val note = Rankings.fallbackReason
            state = if (rows.isEmpty()) state.copy(topLoading = false, topError = "시세를 가져오지 못했습니다", rankNote = note)
            else state.copy(topLoading = false, topRows = rows, topError = null, rankNote = note)
        }
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
            if (state.showTop) {
                val top = withContext(Dispatchers.IO) { OverviewRepo.loadTop(false) }
                if (top.isNotEmpty()) state = state.copy(topRows = top, topError = null)
            }
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

    /** 현재 보고 있는 목록(워치리스트 또는 미장 TOP30). */
    fun activeRows(): List<CompareRow> = if (state.showTop) state.topRows else state.rows

    /** 목록 원래 순서(TOP30 = 시총 순위) — RANK 정렬·순위 표시용. */
    fun rankOf(ticker: String): Int = activeRows().indexOfFirst { it.ticker == ticker }

    /** 보유 필터 적용된 표시 대상 행. */
    fun visibleRows(): List<CompareRow> =
        if (state.holdingsOnly) activeRows().filter { it.holding } else activeRows()

    /**
     * 화면에 실제로 **표시되는** 현재가 — 실시간 틱이 있으면 그 값.
     * 정렬도 이 값으로 해야 표에 보이는 숫자와 순서가 맞는다.
     */
    fun shownPrice(r: CompareRow): Double = LivePrices.price(r.ticker) ?: r.price

    /**
     * 화면에 실제로 표시되는 등락률.
     *
     * 예전에는 정렬만 일봉 종가 기준(`r.day`)으로 하고 표시는 실시간가로 다시 계산해서,
     * 장중에 두 값이 벌어지면 **정렬이 표시값과 어긋나** 보였다 (예: -0.1% 가 +0.4% 와 +0.3% 사이).
     */
    fun shownDay(r: CompareRow): Double {
        val live = LivePrices.price(r.ticker)
        return if (live != null && r.prevClose > 0) (live / r.prevClose - 1) * 100 else r.day
    }

    fun sorted(): List<CompareRow> {
        val src = visibleRows()
        val base = when (state.sortKey) {
            SortKey.RANK -> src   // 목록 원래 순서 유지
            SortKey.NAME -> src.sortedBy { it.name }
            SortKey.PRICE -> src.sortedBy { shownPrice(it) }
            SortKey.DAY -> src.sortedBy { shownDay(it) }
            SortKey.WEEK -> src.sortedBy { it.week }
            SortKey.FROM_HIGH -> src.sortedBy { it.fromHigh }
            SortKey.Z -> src.sortedBy { it.zPct }
            SortKey.M -> src.sortedBy { it.mPct }
            SortKey.BETA -> src.sortedBy { it.beta }
            SortKey.SIGMA -> src.sortedBy { it.sigmaPct }
        }
        return if (state.sortKey != SortKey.RANK && state.sortDesc) base.reversed() else base
    }
}
