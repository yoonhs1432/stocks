package com.quant.dashboard.data

import com.quant.dashboard.quant.Portfolio
import com.quant.dashboard.quant.Quant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope

/**
 * 전 종목 요약(현재가·등락·Z·M·β·σ·신호·보유)을 한 번 받아 공유 캐시.
 * 비교 표·σ·β 산점도·Z·M 산점도·분석 탭 종목 버튼이 공통으로 사용.
 */
object OverviewRepo {
    data class Row(
        val ticker: String, val name: String, val price: Double,
        val prevClose: Double,     // 전일 종가 — 실시간 현재가로 등락률을 다시 계산할 때 사용
        val day: Double, val week: Double, val fromHigh: Double,
        val zPct: Double, val mPct: Double, val signal: String,
        val beta: Double, val sigmaPct: Double,
        val holding: Boolean,      // 현재 보유 중 (★)
        val hasHistory: Boolean,   // 과거 매매 이력만 (☆)
        val zHist: DoubleArray = DoubleArray(0),   // 주간 Z 시계열(최근 N주)
        val mHist: DoubleArray = DoubleArray(0),   // 주간 M 시계열(최근 N주)
    )

    @Volatile private var weekDatesArr: LongArray = LongArray(0)
    /** Z·M 산점도 스크럽 타임라인 (epoch sec, 오래된→최근). 최근 6개월 거래일(일별). */
    fun weekDates(): LongArray = weekDatesArr

    /** 목록별 캐시 슬롯 (워치리스트 / 미장 TOP30). range·interval·asof가 바뀌면 무효화. */
    private class Slot {
        @Volatile var rows: List<Row> = emptyList()
        @Volatile var ts = 0L
        @Volatile var key = ""
    }
    private val watchSlot = Slot()
    private val topSlot = Slot()

    fun cached(): List<Row> = watchSlot.rows

    /** 미장 TOP30 캐시 — 이미 받아둔 게 있으면 재요청 없이 조회(분석 탭 일간등락 표시용). */
    fun cachedTop(): List<Row> = topSlot.rows

    /** 사용자 워치리스트. force=false면 5분 캐시 재사용. IO 디스패처에서 호출 권장. */
    suspend fun load(force: Boolean = false): List<Row> =
        loadInto(watchSlot, Store.loadTickers(), force)

    /** 미국 시총 상위 30개. 비교 탭에서 해당 목록을 열 때만 호출(요청 30건). */
    suspend fun loadTop(force: Boolean = false): List<Row> =
        loadInto(topSlot, Tickers.US_TOP30, force)

    private suspend fun loadInto(slot: Slot, tickers: List<String>, force: Boolean): List<Row> {
        val now = System.currentTimeMillis()
        val range = Store.lookbackRange()
        val interval = Store.candleInterval()
        val curKey = "$range|$interval|${Store.asofDate() ?: ""}"
        if (!force && slot.rows.isNotEmpty() && slot.key == curKey && now - slot.ts < 300_000) return slot.rows
        val spy = Store.sliceAsof(Quotes.closes(Tickers.BASE, range, interval))
        if (spy.isEmpty()) return slot.rows
        val trades = Store.visibleTrades()
        // 토스 모드에서는 ★(보유)를 매매기록 계산이 아니라 실제 계좌 잔고로 판정한다
        val tossHeld: Set<String>? =
            if (Store.tossMode()) TossSync.cachedAccount()?.holdings?.items
                ?.filter { it.quantity > 0 }?.map { it.symbol }?.toSet()
            else null
        // 스크럽 타임라인 — 최근 6개월 거래일(일별), 모든 종목 공유
        val latest = spy.last().first
        val cutoff = latest - 182L * 86400
        val wd = spy.asSequence().map { it.first }.filter { it >= cutoff }.toList().toLongArray()
        val WN = wd.size
        weekDatesArr = wd
        val rows = coroutineScope {
            tickers.map { tk ->
                async(Dispatchers.IO) {
                    val r = Quant.analyze(spy, Store.sliceAsof(Quotes.closes(tk, range, interval))) ?: return@async null
                    val p = r.price; val m = p.size
                    if (m < 2) return@async null
                    val prevD = p[m - 2]
                    val prevW = if (m > 5) p[m - 6] else prevD
                    val high = p.max()
                    val held = tossHeld?.contains(tk) ?: (Portfolio.currentHoldQty(trades[tk].orEmpty()) > 0)
                    // 거래일별 Z·M 샘플 (각 날짜 이하의 최근 값)
                    val zh = DoubleArray(WN) { Double.NaN }
                    val mh = DoubleArray(WN) { Double.NaN }
                    for (w in 0 until WN) {
                        var idx = -1
                        for (i in r.dates.indices) { if (r.dates[i] <= wd[w]) idx = i else break }
                        if (idx >= 0) { zh[w] = r.zPct[idx]; mh[w] = r.mPct[idx] }
                    }
                    Row(
                        ticker = tk, name = Tickers.displayName(tk), price = p[m - 1],
                        prevClose = prevD,
                        day = if (prevD > 0) (p[m - 1] / prevD - 1) * 100 else 0.0,
                        week = if (prevW > 0) (p[m - 1] / prevW - 1) * 100 else 0.0,
                        fromHigh = if (high > 0) (p[m - 1] / high - 1) * 100 else 0.0,
                        zPct = r.lastZpct, mPct = r.lastMpct, signal = r.signal,
                        beta = r.beta, sigmaPct = r.sigmaPct, holding = held,
                        hasHistory = trades[tk].orEmpty().isNotEmpty(),
                        zHist = zh, mHist = mh,
                    )
                }
            }.awaitAll().filterNotNull()
        }
        if (rows.isNotEmpty()) { slot.rows = rows; slot.ts = now; slot.key = curKey }
        return rows
    }
}
