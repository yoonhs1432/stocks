package com.quant.dashboard.data

import org.json.JSONArray
import org.json.JSONObject
import java.time.LocalDate

/**
 * 계좌 잔고 스냅샷 — 자산추이·수익률·MDD 용.
 *
 * 토스 API 에는 **과거 잔고 시계열이 없다.** 그래서 앱이 열릴 때 하루 1회 현재 잔고를
 * 기록해 앞으로 쌓아 간다. 즉 그래프는 전환 시점부터 시작하며, 앱을 며칠 안 열면 그 기간은 비어 있다.
 *
 * ⚠️ **총자산 곡선은 수익률이 아니다.** 입금하면 그대로 올라간다.
 *    순수 손익은 [pnls](평가손익) 곡선을 보면 된다.
 *
 * 예전에는 매매기록으로 입출금을 역산해 TWR 수익률을 냈는데, 매매기록이 불완전하면
 * 값이 어긋나서 걷어냈다. 지금은 **토스가 준 값을 그대로** 저장하고 그린다.
 */
data class Snapshot(
    val date: String,      // YYYY-MM-DD
    val krwEval: Double,   // 국내 종목 평가금액 (KRW)
    val usdEval: Double,   // 해외 종목 평가금액 (USD)
    val krwCash: Double,   // 원화 매수가능금액
    val usdCash: Double,   // 달러 매수가능금액
    val rate: Double,      // 기록 시점 USD/KRW
    // ── 아래는 2026-09-02 부터 기록. 그 전 스냅샷은 NaN 이라 수익률 계산에서 빠진다 ──
    val krwPnl: Double = Double.NaN,        // 평가손익(미실현), 통화별
    val usdPnl: Double = Double.NaN,
    val krwPurchase: Double = Double.NaN,   // 매입금액, 통화별
    val usdPurchase: Double = Double.NaN,
) {
    /** 원화 환산 총자산 (평가금액 + 현금). 입출금이 그대로 반영되는 값. */
    fun totalKrw(): Double = krwEval + krwCash + (usdEval + usdCash) * rate

    /** 평가손익이 기록된 스냅샷인가 (2026-09-02 이전 기록에는 없다). */
    val hasPnl: Boolean get() = !krwPnl.isNaN() && !usdPnl.isNaN()

    /** 평가손익(미실현) 원화 환산. */
    fun pnlKrw(): Double = krwPnl + usdPnl * rate

    /** 평가금액(원화 환산). */
    fun evalKrw(): Double = krwEval + usdEval * rate

    /** 예수금(원화 환산). */
    fun cashKrw(): Double = krwCash + usdCash * rate
}

object Snapshots {
    private const val FILE = "toss_snapshots.json"
    private const val MAX = 1500   // 약 4년치

    private fun JSONObject.optNaN(key: String): Double =
        if (has(key)) optDouble(key, Double.NaN) else Double.NaN

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
                    krwPnl = o.optNaN("krw_pnl"),
                    usdPnl = o.optNaN("usd_pnl"),
                    krwPurchase = o.optNaN("krw_buy"),
                    usdPurchase = o.optNaN("usd_buy"),
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
            val o = JSONObject()
                .put("date", s.date).put("krw_eval", s.krwEval).put("usd_eval", s.usdEval)
                .put("krw_cash", s.krwCash).put("usd_cash", s.usdCash).put("rate", s.rate)
            // NaN 은 JSON 에 담을 수 없다 — 값이 있는 것만 넣고, 없으면 키 자체를 뺀다
            if (!s.krwPnl.isNaN()) o.put("krw_pnl", s.krwPnl)
            if (!s.usdPnl.isNaN()) o.put("usd_pnl", s.usdPnl)
            if (!s.krwPurchase.isNaN()) o.put("krw_buy", s.krwPurchase)
            if (!s.usdPurchase.isNaN()) o.put("usd_buy", s.usdPurchase)
            arr.put(o)
        }
        try { f.writeText(arr.toString()) } catch (e: Exception) {}
    }

    /** 오늘 자 기록을 남긴다(하루 1회, 같은 날 다시 부르면 최신 값으로 덮어씀). */
    fun record(s: Snapshot) {
        val list = load().filter { it.date != s.date } + s
        save(list.sortedBy { it.date })
    }

    fun recordToday(
        krwEval: Double, usdEval: Double, krwCash: Double, usdCash: Double, rate: Double,
        krwPnl: Double = Double.NaN, usdPnl: Double = Double.NaN,
        krwPurchase: Double = Double.NaN, usdPurchase: Double = Double.NaN,
    ) {
        record(
            Snapshot(
                LocalDate.now().toString(), krwEval, usdEval, krwCash, usdCash, rate,
                krwPnl = krwPnl, usdPnl = usdPnl,
                krwPurchase = krwPurchase, usdPurchase = usdPurchase,
            )
        )
    }

    /** 그래프용 시계열 — 토스가 준 값을 그대로 쓴다(파생·역산 없음). */
    data class Series(
        val dates: List<String>,
        val eval: DoubleArray,   // 평가금액 (원화 환산)
        val cash: DoubleArray,   // 예수금 (원화 환산)
        val total: DoubleArray,  // 총자산 = 평가 + 예수금
    )

    fun series(): Series {
        val l = load()
        return Series(
            l.map { it.date },
            DoubleArray(l.size) { l[it].evalKrw() },
            DoubleArray(l.size) { l[it].cashKrw() },
            DoubleArray(l.size) { l[it].totalKrw() },
        )
    }

    /** 평가손익(원) 시계열. 손익이 기록된 날만 — 옛 스냅샷에는 없다. */
    fun pnls(): List<Pair<String, Double>> =
        load().filter { it.hasPnl }.map { it.date to it.pnlKrw() }

    fun clear() { Store.fileIn(FILE)?.delete() }
}
