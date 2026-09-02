package com.quant.dashboard.data

import com.quant.dashboard.quant.Portfolio
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
 *    수익만 보려면 [investPnlKrw](누적 투자손익) 이나 [twr](입출금 제거 수익률 지수) 를 쓸 것.
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
    val krwRealized: Double = Double.NaN,   // 기록 시점까지의 누적 실현손익 (매매기록에서 산출)
    val usdRealized: Double = Double.NaN,
) {
    /** 원화 환산 총자산 (평가금액 + 현금). 입출금이 그대로 반영되는 값. */
    fun totalKrw(): Double = krwEval + krwCash + (usdEval + usdCash) * rate

    /** 수익률 계산에 필요한 필드가 다 있는가 (구 버전 스냅샷은 없다). */
    val hasPnl: Boolean
        get() = !krwPnl.isNaN() && !usdPnl.isNaN() && !krwRealized.isNaN() && !usdRealized.isNaN()

    /** 평가손익(미실현) 원화 환산. */
    fun pnlKrw(): Double = krwPnl + usdPnl * rate

    /** 누적 투자손익 = 미실현 + 실현. **입출금과 무관하다.** */
    fun investPnlKrw(): Double = (krwPnl + krwRealized) + (usdPnl + usdRealized) * rate
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
                    krwRealized = o.optNaN("krw_real"),
                    usdRealized = o.optNaN("usd_real"),
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
            if (!s.krwRealized.isNaN()) o.put("krw_real", s.krwRealized)
            if (!s.usdRealized.isNaN()) o.put("usd_real", s.usdRealized)
            arr.put(o)
        }
        try { f.writeText(arr.toString()) } catch (e: Exception) {}
    }

    /** 오늘 자 기록을 남긴다(하루 1회, 같은 날 다시 부르면 최신 값으로 덮어씀). */
    fun record(s: Snapshot) {
        val list = load().filter { it.date != s.date } + s
        save(list.sortedBy { it.date })
    }

    /**
     * 매매기록에서 통화별 누적 실현손익을 뽑는다.
     *
     * 실현손익은 매도한 순간 평가손익에서 빠져 예수금으로 옮겨 간다. 이 값을 같이 남겨 두지 않으면
     * 매도가 입금과 구분되지 않아 수익률이 어긋난다. 통화 판정은 종목코드 기준(`Tickers.isKrw`).
     */
    private fun realizedByCurrency(): Pair<Double, Double> {
        var kr = 0.0; var us = 0.0
        return try {
            for ((tk, list) in Store.loadTrades()) {
                val v = Portfolio.realizedOf(list)
                if (v == 0.0) continue
                if (Tickers.isKrw(tk)) kr += v else us += v
            }
            Pair(kr, us)
        } catch (e: Exception) {
            Pair(Double.NaN, Double.NaN)
        }
    }

    fun recordToday(
        krwEval: Double, usdEval: Double, krwCash: Double, usdCash: Double, rate: Double,
        krwPnl: Double = Double.NaN, usdPnl: Double = Double.NaN,
        krwPurchase: Double = Double.NaN, usdPurchase: Double = Double.NaN,
    ) {
        val (kr, ur) = realizedByCurrency()
        record(
            Snapshot(
                LocalDate.now().toString(), krwEval, usdEval, krwCash, usdCash, rate,
                krwPnl = krwPnl, usdPnl = usdPnl,
                krwPurchase = krwPurchase, usdPurchase = usdPurchase,
                krwRealized = kr, usdRealized = ur,
            )
        )
    }

    /** 원화 환산 총자산 시계열. **입출금 포함** — 수익률로 읽으면 안 된다. */
    fun totals(): List<Pair<String, Double>> = load().map { it.date to it.totalKrw() }

    /** 누적 투자손익(원) 시계열. 입출금과 무관. 기록에 손익 필드가 있는 구간만. */
    fun investPnl(): List<Pair<String, Double>> =
        load().filter { it.hasPnl }.map { it.date to it.investPnlKrw() }

    /**
     * 두 스냅샷 사이의 **외부 입출금**(원화 환산) 추정.
     *
     * 통화별로 `Δ(평가금액+예수금) − Δ평가손익 − Δ실현손익` 이 남으면 그건 매매로 설명되지 않는
     * 돈이므로 입출금이다. 원화↔달러 환전은 양쪽이 부호만 반대로 잡혀 합치면 거의 0 이 된다.
     */
    private fun flowKrw(a: Snapshot, b: Snapshot): Double {
        fun f(v1: Double, v0: Double, p1: Double, p0: Double, r1: Double, r0: Double) =
            (v1 - v0) - (p1 - p0) - (r1 - r0)
        val fk = f(b.krwEval + b.krwCash, a.krwEval + a.krwCash, b.krwPnl, a.krwPnl, b.krwRealized, a.krwRealized)
        val fu = f(b.usdEval + b.usdCash, a.usdEval + a.usdCash, b.usdPnl, a.usdPnl, b.usdRealized, a.usdRealized)
        return fk + fu * b.rate
    }

    /**
     * 시간가중수익률(TWR) 지수 — 시작을 100 으로 두고 구간 수익률을 곱해 나간다.
     *
     * `r = (기말자산 − 그 구간 입출금) / 기초자산 − 1`. 입금해도 지수는 움직이지 않으므로
     * **순수하게 사고판 결과만** 남는다. 손익 필드가 없는 옛 스냅샷은 빠지므로 곡선은 그때부터 시작한다.
     */
    fun twr(): List<Pair<String, Double>> {
        val l = load().filter { it.hasPnl }
        if (l.size < 2) return emptyList()
        var idx = 100.0
        val out = ArrayList<Pair<String, Double>>(l.size)
        out.add(l[0].date to idx)
        for (i in 1 until l.size) {
            val a = l[i - 1]; val b = l[i]
            val v0 = a.totalKrw()
            if (v0 > 0) {
                val r = (b.totalKrw() - flowKrw(a, b)) / v0 - 1.0
                // 전액 출금 같은 극단값에서 지수가 0/음수로 죽지 않게 하한을 둔다
                if (r.isFinite()) idx *= (1 + r).coerceAtLeast(0.01)
            }
            out.add(b.date to idx)
        }
        return out
    }

    /** 고점 대비 현재 낙폭(%)과 최대 낙폭(%)·그 날짜. 기록이 2개 미만이면 null. */
    data class Drawdown(val current: Double, val max: Double, val maxDate: String?)

    /** 총자산 기준 낙폭 — 입출금이 섞이므로 참고용. 수익률 낙폭은 `drawdown(twr())`. */
    fun drawdown(): Drawdown? = drawdown(totals())

    fun drawdown(t: List<Pair<String, Double>>): Drawdown? {
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
