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
import kotlinx.coroutines.launch

typealias CompareRow = OverviewRepo.Row

/** RANK 제거 — 목록이 워치리스트+보유 하나뿐이라 "원래 순서" 정렬이 의미 없다. */
enum class SortKey { M, Z, DAY, WEEK, FROM_HIGH, PRICE, NAME, BETA, SIGMA }

data class CompareState(
    val loading: Boolean = false,
    val error: String? = null,
    // 앱을 켜자마자 **지난번 표**로 그린다 — 서버 첫 조회는 PC 에서도 20~30초다
    val rows: List<CompareRow> = OverviewRepo.cached(),
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

    /**
     * AppState.dataVersion 변경(기준일·설정) 시 재로드, 아니면 최초 1회만.
     *
     * ⚠️ **종목 목록을 서버에서 받기 전에는 부르지 않는다.** 예전에는 기본 목록으로 한 번
     * 부르고, 목록이 도착하면 또 한 번 불렀다. 서버의 첫 조회가 20~30초라 그 한 번이
     * 그대로 대기 시간으로 쌓였다(미국·국내 각각이라 분 단위가 되기도 했다).
     *
     * ⚠️ **첫 로드는 강제하지 않는다.** `force=true` 는 서버 캐시(5분)와 일봉 캐시를
     * 둘 다 건너뛰고 전부 다시 받게 한다. 설정이 바뀌면 서버 캐시 키(기간·종목 수)가
     * 어차피 달라져서 새로 계산되므로, 강제는 사용자가 직접 당겨서 새로고침할 때만 쓴다.
     */
    fun sync(version: Int) {
        if (!Store.synced()) return
        if (version != loadedVersion) {
            val first = loadedVersion < 0
            loadedVersion = version
            load(force = !first)
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

    /**
     * **보고 있는 시장을 먼저** 받아 그리고, 반대쪽은 그 뒤에 채운다.
     * 둘 다 기다렸다 한꺼번에 그리면 보이지도 않는 시장 때문에 화면이 두 배로 늦었다.
     */
    fun load(force: Boolean = false) {
        state = state.copy(loading = true, error = null)
        viewModelScope.launch {
            val order = if (state.market == "KR") listOf("KR", "US") else listOf("US", "KR")
            var any = false
            for (mk in order) {
                val ok = OverviewRepo.loadMarket(mk, force)
                if (ok) {
                    any = true
                    state = state.copy(loading = false, rows = OverviewRepo.cached(), error = null)
                }
            }
            // 왜 안 되는지를 그대로 보여준다 — "가져오지 못했습니다" 만으로는 고칠 수가 없다
            if (!any) state = state.copy(loading = false,
                error = OverviewRepo.lastError ?: "시세를 가져오지 못했습니다")
        }
    }

    /** 자동(조용한) 새로고침 — 로딩 표시 없이 명단 갱신(5분 캐시 만료 시에만 실제 재요청). */
    fun autoRefresh() {
        viewModelScope.launch {
            val rows = OverviewRepo.load(false)
            // ⚠️ 성공 여부는 **rows 가 비었는지가 아니라** lastError 로 판단한다.
            //    파일에서 꺼낸 지난번 표가 깔려 있으면 실패해도 rows 는 비어 있지 않다.
            state = state.copy(
                rows = if (rows.isNotEmpty()) rows else state.rows,
                error = OverviewRepo.lastError,
            )
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
