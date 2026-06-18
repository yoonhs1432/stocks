package com.quant.dashboard.quant

import com.quant.dashboard.data.Trade
import java.time.LocalDate

/**
 * 포트폴리오 집계 — backend/analysis.py(resolve_all_cycles / compute_portfolio_equity /
 * compute_drawdown / calc_portfolio_total_pnl)를 Kotlin으로 포팅.
 */
object Portfolio {
    const val SEED_USD = 20_000.0

    data class Holding(
        val ticker: String, val name: String, val qty: Int,
        val avg: Double, val cur: Double, val eval: Double,
        val pnl: Double, val retPct: Double,
    )

    data class Realized(val ticker: String, val name: String, val realized: Double)

    data class Result(
        val totalPnl: Double,
        val seed: Double,
        val holdings: List<Holding>,
        val realized: List<Realized>,
        val equity: List<Pair<Long, Double>>,  // (epochSec, 누적손익)
        val currentDd: Double,
        val mdd: Double,
        val mddDate: Long?,
    )

    private data class Cyc(
        val holdQty: Int, val buyQty: Int, val buyCost: Double,
        val cumulative: Double, val currentPnl: Double?,
    )

    /** 매매 기록 → 현재 사이클 + 누적 실현손익 (resolve_all_cycles). */
    private fun resolve(trades: List<Trade>): Cyc {
        val sorted = trades.filter { it.qty > 0 && it.price > 0 }.sortedBy { it.date }
        var holdQty = 0; var buyQty = 0; var buyCost = 0.0; var sellProceeds = 0.0
        var cumulative = 0.0; var realizedPartial = 0.0
        var hasSell = false; var started = false; var ended = false
        for (r in sorted) {
            if (r.type == "buy") {
                if (holdQty == 0) {
                    if (started && ended) cumulative += sellProceeds - buyCost
                    buyQty = 0; buyCost = 0.0; sellProceeds = 0.0
                    realizedPartial = 0.0; hasSell = false
                    started = true; ended = false
                }
                holdQty += r.qty; buyQty += r.qty; buyCost += r.qty * r.price
            } else if (r.type == "sell" && holdQty > 0) {
                val avg = if (buyQty > 0) buyCost / buyQty else 0.0
                realizedPartial += r.qty * (r.price - avg); hasSell = true
                sellProceeds += r.qty * r.price
                holdQty = maxOf(holdQty - r.qty, 0)
                if (holdQty == 0) ended = true
            }
        }
        val currentPnl = when {
            ended -> sellProceeds - buyCost
            hasSell -> realizedPartial
            else -> null
        }
        return Cyc(holdQty, buyQty, buyCost, cumulative, currentPnl)
    }

    fun compute(
        trades: Map<String, List<Trade>>,
        name: (String) -> String,
        lastClose: Map<String, Double>,
        hist: Map<String, List<Pair<Long, Double>>>,
        seed: Double = SEED_USD,
    ): Result {
        val holdings = ArrayList<Holding>()
        val realized = ArrayList<Realized>()
        var total = 0.0
        for ((tk, list) in trades) {
            val c = resolve(list)
            if (c.buyQty == 0) continue
            val real = c.cumulative + (c.currentPnl ?: 0.0)
            if (real != 0.0) realized.add(Realized(tk, name(tk), real))
            var unreal = 0.0
            if (c.holdQty > 0) {
                val avg = c.buyCost / c.buyQty
                val cur = lastClose[tk] ?: avg
                unreal = (cur - avg) * c.holdQty
                holdings.add(
                    Holding(tk, name(tk), c.holdQty, avg, cur, cur * c.holdQty,
                        unreal, if (avg > 0) (cur / avg - 1) * 100 else 0.0)
                )
            }
            total += real + unreal
        }
        val (equity, dd) = equityAndDrawdown(trades, hist, seed)
        return Result(
            totalPnl = total, seed = seed,
            holdings = holdings.sortedByDescending { it.eval },
            realized = realized.sortedByDescending { it.realized },
            equity = equity, currentDd = dd.first, mdd = dd.second, mddDate = dd.third,
        )
    }

    private fun parseDay(date: String): Long? =
        try { LocalDate.parse(date).toEpochDay() } catch (e: Exception) { null }

    private fun equityAndDrawdown(
        trades: Map<String, List<Trade>>,
        hist: Map<String, List<Pair<Long, Double>>>,
        seed: Double,
    ): Pair<List<Pair<Long, Double>>, Triple<Double, Double, Long?>> {
        data class Ev(val day: Long, val ticker: String, val type: String, val qty: Int, val price: Double)
        val events = ArrayList<Ev>()
        for ((tk, list) in trades) for (t in list) {
            if (t.qty > 0 && t.price > 0) {
                val d = parseDay(t.date) ?: continue
                events.add(Ev(d, tk, t.type, t.qty, t.price))
            }
        }
        if (events.isEmpty()) return Pair(emptyList(), Triple(0.0, 0.0, null))
        events.sortBy { it.day }

        val closeMap = HashMap<String, HashMap<Long, Double>>()
        val allDays = sortedSetOf<Long>()
        for ((tk, series) in hist) {
            val m = HashMap<Long, Double>()
            for ((sec, c) in series) { val d = sec / 86400L; m[d] = c; allDays.add(d) }
            closeMap[tk] = m
        }
        if (allDays.isEmpty()) return Pair(emptyList(), Triple(0.0, 0.0, null))

        val holdings = HashMap<String, Int>()
        val avgCosts = HashMap<String, Double>()
        val lastKnown = HashMap<String, Double>()
        var realizedTotal = 0.0
        var ei = 0
        val equity = ArrayList<Pair<Long, Double>>()
        for (day in allDays) {
            // 오늘 종가로 ffill 갱신
            for ((tk, m) in closeMap) m[day]?.let { lastKnown[tk] = it }
            // 이 날짜까지 이벤트 처리
            while (ei < events.size && events[ei].day <= day) {
                val e = events[ei]
                val cq = holdings.getOrDefault(e.ticker, 0)
                val ca = avgCosts.getOrDefault(e.ticker, 0.0)
                if (e.type == "buy") {
                    val nq = cq + e.qty
                    avgCosts[e.ticker] = if (nq > 0) ((ca * cq) + (e.price * e.qty)) / nq else 0.0
                    holdings[e.ticker] = nq
                } else if (e.type == "sell" && cq > 0) {
                    val sq = minOf(e.qty, cq)
                    realizedTotal += (e.price - ca) * sq
                    holdings[e.ticker] = cq - sq
                    if (holdings[e.ticker] == 0) avgCosts[e.ticker] = 0.0
                }
                ei++
            }
            var unreal = 0.0
            for ((tk, q) in holdings) {
                if (q == 0) continue
                val c = lastKnown[tk] ?: continue
                unreal += (c - (avgCosts[tk] ?: 0.0)) * q
            }
            equity.add(Pair(day * 86400L, realizedTotal + unreal))
        }

        var runMax = Double.NEGATIVE_INFINITY
        var curDd = 0.0; var mdd = 0.0; var mddDay: Long? = null
        for ((sec, pnl) in equity) {
            val pv = pnl + seed
            if (pv > runMax) runMax = pv
            val dd = if (runMax > 0) (pv - runMax) / runMax * 100 else 0.0
            curDd = dd
            if (dd < mdd) { mdd = dd; mddDay = sec }
        }
        return Pair(equity, Triple(curDd, mdd, mddDay))
    }
}
