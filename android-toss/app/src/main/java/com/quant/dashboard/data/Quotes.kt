package com.quant.dashboard.data

/**
 * 시세 — **토스 일봉만** 쓴다.
 *
 * 예전에는 Yahoo 를 대체 소스로 뒀는데, 토스가 429 로 실패한 종목만 조용히 Yahoo 로 넘어가면서
 * **한 화면 안에서 종목마다 시세 출처가 뒤섞였다.** 두 소스의 전일 종가가 다르면 등락률이
 * 그만큼 어긋난다(GDXU 가 -7.6% 대신 -19.4% 로 나왔다). 그래서 소스를 하나로 못 박았다.
 * 실패하면 값을 지어내지 않고 **빈 리스트**를 돌려준다 — 화면에 문제가 드러나는 편이 낫다.
 */
/** 봉 1개 (일봉). */
data class Candle(val t: Long, val open: Double, val high: Double, val low: Double, val close: Double)

object Quotes {

    /** 토스는 봉 수로 요청한다 — 개월당 약 22 거래일 + 경계 여유. */
    private fun barCount(months: Int): Int = (months * 22 + 15).coerceAtLeast(40)

    /**
     * 요청 개월보다 오래된 봉을 잘라낸다 (캐시를 더 긴 기간으로 받아 뒀을 때).
     * 자른 결과가 2봉 미만이면(상장 직후 등) 자르지 않은 원본을 돌려준다.
     */
    private fun trimMonths(list: List<Candle>, months: Int): List<Candle> {
        if (list.size < 2) return list
        val cut = list.last().t - months.toLong() * 2_629_746L   // 1개월 = 30.44일
        val out = list.filter { it.t >= cut }
        return if (out.size >= 2) out else list
    }

    /**
     * 일봉 동시 요청 제한.
     *
     * 비교 탭은 20여 종목을 **병렬로** 분석하는데 종목당 200봉씩 3페이지를 받으므로
     * 한꺼번에 60여 건이 `MARKET_DATA_CHART` 로 몰린다. 그러면 일부가 429 로 떨어진다.
     */
    private val chartGate = java.util.concurrent.Semaphore(3, true)

    // ── 일봉 캐시 ──
    // 일봉은 하루에 한 번만 바뀌는데 예전에는 화면을 새로 그릴 때마다(5분 캐시 만료 시마다)
    // 종목당 200봉×3페이지를 다시 받았다. 현재가는 틱(`LivePrices`)이 따로 갱신하므로
    // 일봉은 한 번 받아 두고 재사용한다.
    private class Cached(val months: Int, val at: Long, val bars: List<Candle>)

    private val cache = java.util.concurrent.ConcurrentHashMap<String, Cached>()

    /** 캐시 수명. 미국장 마감(05:00 KST) 뒤 새 봉이 생기므로 반나절이면 충분하다. */
    private const val TTL = 6 * 3600_000L

    /** 받아 둔 일봉을 모두 버린다 (조회기간을 늘렸거나 손으로 다시 받을 때). */
    fun clearCache() { cache.clear(); todayCache.clear() }

    // ── 당일 캔들 (비교 탭 미니 캔들용) ──
    // 분석용 일봉은 2년치라 종목당 3페이지 + 6시간 캐시다. 장중에 그걸 자주 받으면
    // 예전처럼 429 가 나므로 건드리지 않고, **3봉만** 따로 받는다(1페이지, 5분 캐시).
    private class Today(val at: Long, val bar: Candle?)

    private val todayCache = java.util.concurrent.ConcurrentHashMap<String, Today>()
    private const val TODAY_TTL = 5 * 60_000L

    /** 마지막 일봉 1개. 장중이면 오늘 봉이 갱신돼 온다. 실패하면 null. */
    fun todayCandle(symbol: String): Candle? {
        val c = todayCache[symbol]
        if (c != null && System.currentTimeMillis() - c.at < TODAY_TTL) return c.bar
        if (!BrokerCreds.isLinked()) return null
        var bar: Candle? = null
        chartGate.acquire()
        try {
            bar = runCatching { TossApi.dailyOhlc(symbol, 3).lastOrNull() }.getOrNull()
        } finally {
            chartGate.release()
        }
        todayCache[symbol] = Today(System.currentTimeMillis(), bar)
        return bar
    }

    /** 일봉. 조회 실패 시 빈 리스트. */
    fun ohlc(
        symbol: String,
        months: Int = Store.lookbackMonths(),
        force: Boolean = false,
    ): List<Candle> {
        val c = cache[symbol]
        // 더 짧은 기간을 요청하면 받아 둔 것을 잘라 쓰면 된다 — 다시 받을 필요 없음
        if (!force && c != null && c.months >= months &&
            System.currentTimeMillis() - c.at < TTL
        ) {
            return trimMonths(c.bars, months)
        }
        if (!BrokerCreds.isLinked()) return emptyList()

        var out: List<Candle> = emptyList()
        chartGate.acquire()
        try {
            var attempt = 0
            while (attempt < 3) {
                try {
                    out = TossApi.dailyOhlc(symbol, barCount(months)); break
                } catch (e: TossException) {
                    // 한도면 잠깐 쉬었다 재시도
                    if (e.http == 429) { Thread.sleep(1200L * (attempt + 1)); attempt++ } else break
                } catch (e: Exception) {
                    break
                }
            }
        } finally {
            chartGate.release()
        }
        if (out.size < 2) return emptyList()
        cache[symbol] = Cached(months, System.currentTimeMillis(), out)
        return trimMonths(out, months)
    }

    fun closes(
        symbol: String,
        months: Int = Store.lookbackMonths(),
        force: Boolean = false,
    ): List<Pair<Long, Double>> = ohlc(symbol, months, force).map { Pair(it.t, it.close) }
}
