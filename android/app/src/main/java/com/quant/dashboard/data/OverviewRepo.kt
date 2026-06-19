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
        val day: Double, val week: Double, val fromHigh: Double,
        val zPct: Double, val mPct: Double, val signal: String,
        val beta: Double, val sigmaPct: Double, val holding: Boolean,
    )

    @Volatile private var cache: List<Row> = emptyList()
    @Volatile private var ts = 0L

    fun cached(): List<Row> = cache

    /** force=false면 5분 캐시 재사용. IO 디스패처에서 호출 권장. */
    suspend fun load(force: Boolean = false): List<Row> {
        val now = System.currentTimeMillis()
        if (!force && cache.isNotEmpty() && now - ts < 300_000) return cache
        val spy = Yahoo.closes(Tickers.BASE)
        if (spy.isEmpty()) return cache
        val trades = Store.loadTrades()
        val rows = coroutineScope {
            Store.loadTickers().map { tk ->
                async(Dispatchers.IO) {
                    val r = Quant.analyze(spy, Yahoo.closes(tk)) ?: return@async null
                    val p = r.price; val m = p.size
                    if (m < 2) return@async null
                    val prevD = p[m - 2]
                    val prevW = if (m > 5) p[m - 6] else prevD
                    val high = p.max()
                    val held = Portfolio.currentHoldQty(trades[tk].orEmpty()) > 0
                    Row(
                        ticker = tk, name = Tickers.displayName(tk), price = p[m - 1],
                        day = if (prevD > 0) (p[m - 1] / prevD - 1) * 100 else 0.0,
                        week = if (prevW > 0) (p[m - 1] / prevW - 1) * 100 else 0.0,
                        fromHigh = if (high > 0) (p[m - 1] / high - 1) * 100 else 0.0,
                        zPct = r.lastZpct, mPct = r.lastMpct, signal = r.signal,
                        beta = r.beta, sigmaPct = r.sigmaPct, holding = held,
                    )
                }
            }.awaitAll().filterNotNull()
        }
        if (rows.isNotEmpty()) { cache = rows; ts = now }
        return rows
    }
}
