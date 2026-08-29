package com.quant.dashboard.data

import org.json.JSONArray
import org.json.JSONObject
import java.time.LocalDate

/**
 * 계좌 잔고 스냅샷 — 자산추이·MDD 용.
 *
 * 토스 API 에는 **과거 잔고 시계열이 없다.** 그래서 앱이 열릴 때 하루 1회 현재 잔고를
 * 기록해 앞으로 쌓아 간다. 즉 그래프는 전환 시점부터 시작하며, 앱을 며칠 안 열면 그 기간은 비어 있다.
 */
data class Snapshot(
    val date: String,      // YYYY-MM-DD
    val krwEval: Double,   // 국내 종목 평가금액 (KRW)
    val usdEval: Double,   // 해외 종목 평가금액 (USD)
    val krwCash: Double,   // 원화 매수가능금액
    val usdCash: Double,   // 달러 매수가능금액
    val rate: Double,      // 기록 시점 USD/KRW
) {
    /** 원화 환산 총자산 (평가금액 + 현금). */
    fun totalKrw(): Double = krwEval + krwCash + (usdEval + usdCash) * rate
}

object Snapshots {
    private const val FILE = "toss_snapshots.json"
    private const val MAX = 1500   // 약 4년치

    fun load(): List<Snapshot> {
        val f = Store.fileIn(FILE) ?: return emptyList()
        if (!f.exists()) return emptyList()
        return try {
            val arr = JSONArray(f.readText())
            (0 until arr.length()).mapNotNull { i ->
                val o = arr.optJSONObject(i) ?: return@mapNotNull null
                val d = o.optString("date").trim()
                if (d.length != 10) return@mapNotNull null
                Snapshot(
                    date = d,
                    krwEval = o.optDouble("krw_eval", 0.0),
                    usdEval = o.optDouble("usd_eval", 0.0),
                    krwCash = o.optDouble("krw_cash", 0.0),
                    usdCash = o.optDouble("usd_cash", 0.0),
                    rate = o.optDouble("rate", 1400.0),
                )
            }.sortedBy { it.date }
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun save(list: List<Snapshot>) {
        val f = Store.fileIn(FILE) ?: return
        val arr = JSONArray()
        for (s in list.takeLast(MAX)) {
            arr.put(
                JSONObject()
                    .put("date", s.date).put("krw_eval", s.krwEval).put("usd_eval", s.usdEval)
                    .put("krw_cash", s.krwCash).put("usd_cash", s.usdCash).put("rate", s.rate)
            )
        }
        try { f.writeText(arr.toString()) } catch (e: Exception) {}
    }

    /** 오늘 자 기록을 남긴다(하루 1회, 같은 날 다시 부르면 최신 값으로 덮어씀). */
    fun record(s: Snapshot) {
        val list = load().filter { it.date != s.date } + s
        save(list.sortedBy { it.date })
    }

    fun recordToday(krwEval: Double, usdEval: Double, krwCash: Double, usdCash: Double, rate: Double) {
        record(Snapshot(LocalDate.now().toString(), krwEval, usdEval, krwCash, usdCash, rate))
    }

    /** 원화 환산 총자산 시계열. */
    fun totals(): List<Pair<String, Double>> = load().map { it.date to it.totalKrw() }

    /** 고점 대비 현재 낙폭(%)과 최대 낙폭(%)·그 날짜. 기록이 2개 미만이면 null. */
    data class Drawdown(val current: Double, val max: Double, val maxDate: String?)

    fun drawdown(): Drawdown? {
        val t = totals()
        if (t.size < 2) return null
        var peak = Double.NEGATIVE_INFINITY
        var mdd = 0.0
        var mddDate: String? = null
        for ((d, v) in t) {
            if (v > peak) peak = v
            if (peak > 0) {
                val dd = (v / peak - 1) * 100
                if (dd < mdd) { mdd = dd; mddDate = d }
            }
        }
        val last = t.last().second
        val cur = if (peak > 0) (last / peak - 1) * 100 else 0.0
        return Drawdown(cur, mdd, mddDate)
    }

    fun clear() { Store.fileIn(FILE)?.delete() }
}
