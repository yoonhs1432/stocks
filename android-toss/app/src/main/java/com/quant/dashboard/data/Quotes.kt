package com.quant.dashboard.data

/**
 * 시세 소스 라우팅 — 토스 연동이 켜져 있으면 토스, 아니면 Yahoo.
 *
 * 토스로 갈 수 없는 것:
 *  - 암호화폐(BTC-USD, ETH-USD 등) — 토스 API 범위 밖이라 항상 Yahoo
 *  - 토스에서 조회되지 않는 종목 — 빈 응답/에러 시 자동으로 Yahoo 로 폴백
 *
 * 기본값은 **꺼짐**이다. 토스 일봉은 200봉/요청이라 2년치면 종목당 3회 호출이 필요하고
 * (`MARKET_DATA_CHART` 레이트리밋), 회귀 기준인 SPY 를 포함해 어떤 종목이 실제로
 * 조회되는지 확인되기 전까지는 기존 분석 결과를 바꾸지 않는 편이 안전하다.
 */
object Quotes {

    /** 토스에서 조회 불가능한 심볼 (암호화폐 등). */
    private fun tossUnsupported(symbol: String): Boolean =
        symbol.contains("-")   // BTC-USD, ETH-USD …

    private fun useToss(symbol: String): Boolean =
        Store.tossQuotes() && BrokerCreds.isLinked() && !tossUnsupported(symbol)

    /** 토스는 봉 수로 요청한다 — 개월당 약 22 거래일 + 경계 여유. */
    private fun barCount(months: Int): Int = (months * 22 + 15).coerceAtLeast(40)

    /**
     * 요청 개월보다 오래된 봉을 잘라낸다.
     * Yahoo range 토큰(3mo/6mo/1y/2y)은 1개월 단위가 아니라 넉넉히 받아 여기서 정확히 맞춘다.
     * 자른 결과가 2봉 미만이면(상장 직후 등) 자르지 않은 원본을 돌려준다.
     */
    private fun trimMonths(list: List<Candle>, months: Int): List<Candle> {
        if (list.size < 2) return list
        val cut = list.last().t - months.toLong() * 2_629_746L   // 1개월 = 30.44일
        val out = list.filter { it.t >= cut }
        return if (out.size >= 2) out else list
    }

    /**
     * 토스 일봉 동시 요청 제한.
     *
     * 비교 탭은 20여 종목을 **병렬로** 분석하는데 종목당 200봉씩 3페이지를 받으므로
     * 한꺼번에 60여 건이 `MARKET_DATA_CHART` 로 몰린다. 그러면 일부가 429 로 떨어지고
     * **그 종목만 Yahoo 로 폴백**해, 한 화면 안에서 종목마다 시세 출처가 뒤섞인다.
     * (실제로 이 때문에 같은 종목의 전일 종가가 호출 시점마다 달라져 등락률이 크게 어긋났다.)
     */
    private val chartGate = java.util.concurrent.Semaphore(3, true)

    /** 심볼별로 마지막에 어느 소스를 썼는지 — 진단 화면에서 출처를 확인하기 위해. */
    private val srcMap = java.util.concurrent.ConcurrentHashMap<String, String>()

    fun sourceOf(symbol: String): String = srcMap[symbol] ?: "-"

    /** OHLC 봉. 실패 시 Yahoo 로 폴백하며, 그래도 실패하면 빈 리스트. */
    fun ohlc(
        symbol: String,
        months: Int = Store.lookbackMonths(),
        interval: String = "1d",
    ): List<Candle> {
        // 주봉 등 일봉이 아닌 요청은 토스가 지원하지 않으므로(1m/1d 뿐) Yahoo 사용
        if (interval == "1d" && useToss(symbol)) {
            var out: List<Candle> = emptyList()
            var err: String? = null
            chartGate.acquire()
            try {
                var attempt = 0
                while (attempt < 3) {
                    try {
                        out = TossApi.dailyOhlc(symbol, barCount(months)); break
                    } catch (e: TossException) {
                        err = e.code
                        // 한도면 잠깐 쉬었다 재시도 — 여기서 포기하면 이 종목만 다른 소스가 된다
                        if (e.http == 429) { Thread.sleep(1200L * (attempt + 1)); attempt++ } else break
                    } catch (e: Exception) {
                        err = e.message; break
                    }
                }
            } finally {
                chartGate.release()
            }
            if (out.size >= 2) {
                srcMap[symbol] = "토스"
                return trimMonths(out, months)
            }
            srcMap[symbol] = "Yahoo(토스실패:${err ?: "빈응답"})"
        } else {
            srcMap[symbol] = "Yahoo"
        }
        return trimMonths(Yahoo.ohlc(symbol, Store.rangeToken(months), interval), months)
    }

    fun closes(
        symbol: String,
        months: Int = Store.lookbackMonths(),
        interval: String = "1d",
    ): List<Pair<Long, Double>> = ohlc(symbol, months, interval).map { Pair(it.t, it.close) }
}
