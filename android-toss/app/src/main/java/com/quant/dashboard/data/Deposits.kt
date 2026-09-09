package com.quant.dashboard.data

import org.json.JSONArray
import org.json.JSONObject

/** 입금(양수)·출금(음수) 한 건. 금액은 **원화 고정**. */
data class Deposit(val date: String, val krw: Double)

/**
 * 원금 장부 — 사용자가 직접 적는다.
 *
 * 토스 API 에는 입출금 엔드포인트가 없다(33개 경로 전수 확인). 예전엔 `총자산 − 평가손익`
 * 으로 원금을 추정했는데, **매도로 확정한 실현손익이 예수금에 섞여** 원금이 부풀고 수익률이
 * 실제보다 낮게 나왔다. 적어 둔 입금액이 있으면 그게 유일하게 정확한 기준이다.
 *
 * 날짜별 누적이라 나중에 추가 입금을 적으면 **그 날짜부터** 원금선이 계단으로 올라간다.
 */
object Deposits {
    private const val FILE = "toss_deposits.json"

    /** 날짜 오름차순. 같은 날 여러 건은 적은 순서를 유지한다. */
    fun load(): List<Deposit> {
        val f = Store.fileIn(FILE) ?: return emptyList()
        if (!f.exists()) return emptyList()
        return try {
            val arr = JSONArray(f.readText())
            (0 until arr.length()).mapNotNull { i ->
                val o = arr.optJSONObject(i) ?: return@mapNotNull null
                val d = o.optString("date").trim()
                if (d.length != 10) null else Deposit(d, o.optDouble("krw", 0.0))
            }.sortedBy { it.date }
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun save(list: List<Deposit>) {
        val f = Store.fileIn(FILE) ?: return
        val arr = JSONArray()
        list.sortedBy { it.date }.forEach {
            arr.put(JSONObject().put("date", it.date).put("krw", it.krw))
        }
        try { f.writeText(arr.toString()) } catch (e: Exception) { }
    }

    fun add(date: String, krw: Double) { save(load() + Deposit(date, krw)) }

    /** [load] 순서 기준 index 삭제. */
    fun removeAt(index: Int) {
        val l = load()
        if (index in l.indices) save(l.filterIndexed { i, _ -> i != index })
    }

    /** 지금까지 넣은 원금 합계(원). 기록이 없으면 0. */
    fun total(): Double = load().sumOf { it.krw }

    /**
     * 날짜별 누적 원금 — 각 날짜에 대해 **그 날짜까지의** 입금 합계.
     *
     * 기록이 아직 없는 구간은 `NaN` 이라 선이 그려지지 않는다(0원으로 찍히면 손익이 총자산
     * 전체가 되어 버린다). `dates` 는 오름차순이어야 한다.
     *
     * @param rates 그날 환율 — usd=true 일 때만 쓴다. 과거 금액을 오늘 환율로 바꾸면
     *              환율 변동이 원금 변동처럼 보이므로 스냅샷에 저장된 값을 그대로 쓴다.
     */
    fun seriesFor(dates: List<String>, usd: Boolean, rates: DoubleArray): DoubleArray {
        val l = load()
        if (l.isEmpty()) return DoubleArray(dates.size) { Double.NaN }
        var k = 0
        var acc = 0.0
        return DoubleArray(dates.size) { i ->
            while (k < l.size && l[k].date <= dates[i]) { acc += l[k].krw; k++ }
            val r = rates.getOrNull(i) ?: 1400.0
            when {
                k == 0 -> Double.NaN            // 첫 입금 이전 구간
                usd -> acc / r
                else -> acc
            }
        }
    }
}
