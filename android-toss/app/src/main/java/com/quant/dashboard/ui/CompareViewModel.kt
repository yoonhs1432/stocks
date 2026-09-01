package com.quant.dashboard.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

typealias CompareRow = OverviewRepo.Row

/** RANK 제거 — 목록이 워치리스트+보유 하나뿐이라 "원래 순서" 정렬이 의미 없다. */
enum class SortKey { M, Z, DAY, WEEK, FROM_HIGH, PRICE, NAME, BETA, SIGMA }

data class CompareState(
    val loading: Boolean = false,
    val error: String? = null,
    val rows: List<CompareRow> = emptyList(),
    val sortKey: SortKey = SortKey.DAY,
    val sortDesc: Boolean = true,
    val holdingsOnly: Boolean = false,
    /** 보고 있는 시장 — 미국·국내를 한 화면에 섞지 않고 버튼으로 전환한다. */
    val market: String = "US",
)

class CompareViewModel : ViewModel() {
    var state by mutableStateOf(CompareState(market = Store.compareMarket()))
        private set

    private var loadedVersion = -1

    /** AppState.dataVersion 변경(기준일·설정) 시 강제 재로드, 아니면 최초 1회만. */
    fun sync(version: Int) {
        if (version != loadedVersion) {
            loadedVersion = version
            load(force = true)
        } else {
            loadIfEmpty()
        }
    }

    fun setMarket(m: String) {
        if (m == state.market) return
        Store.setCompareMarket(m)
        state = state.copy(market = m)
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

    /** 워치리스트에 두 시장이 다 있는지 — 하나뿐이면 전환 버튼을 감춘다. */
    fun hasBothMarkets(): Boolean =
        state.rows.any { Tickers.isKrw(it.ticker) } && state.rows.any { !Tickers.isKrw(it.ticker) }

    /** 선택한 시장 + 보유 필터를 적용한 표시 대상. */
    fun visibleRows(): List<CompareRow> {
        val byMarket =
            if (!hasBothMarkets()) state.rows
            else state.rows.filter { Tickers.isKrw(it.ticker) == (state.market == "KR") }
        return if (state.holdingsOnly) byMarket.filter { it.holding } else byMarket
    }

    /**
     * 화면에 실제로 **표시되는** 현재가 — 실시간 틱이 있으면 그 값.
     * 정렬도 이 값으로 해야 표에 보이는 숫자와 순서가 맞는다.
     */
    fun shownPrice(r: CompareRow): Double = LivePrices.price(r.ticker) ?: r.price

    /**
     * 화면에 실제로 표시되는 등락률.
     *
     * 예전에는 정렬만 일봉 종가 기준(`r.day`)으로 하고 표시는 실시간가로 다시 계산해서,
     * 장중에 두 값이 벌어지면 **정렬이 표시값과 어긋나** 보였다.
     */
    fun shownDay(r: CompareRow): Double {
        val live = LivePrices.price(r.ticker)
        return if (live != null && r.prevClose > 0) (live / r.prevClose - 1) * 100 else r.day
    }

    fun sorted(): List<CompareRow> {
        val src = visibleRows()
        val base = when (state.sortKey) {
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
        return if (state.sortDesc) base.reversed() else base
    }
}
