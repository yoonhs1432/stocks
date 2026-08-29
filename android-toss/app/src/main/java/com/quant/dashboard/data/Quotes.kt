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

    /** 분석 기간(range 토큰)에 맞는 대략적인 거래일 수 — 토스는 봉 수로 요청한다. */
    private fun barCount(range: String): Int = when (range) {
        "6mo" -> 130
        "1y" -> 260
        else -> 520   // 2y
    }

    /** OHLC 봉. 실패 시 Yahoo 로 폴백하며, 그래도 실패하면 빈 리스트. */
    fun ohlc(symbol: String, range: String = "2y", interval: String = "1d"): List<Candle> {
        // 주봉 등 일봉이 아닌 요청은 토스가 지원하지 않으므로(1m/1d 뿐) Yahoo 사용
        if (interval == "1d" && useToss(symbol)) {
            val out = try { TossApi.dailyOhlc(symbol, barCount(range)) } catch (e: Exception) { emptyList() }
            if (out.size >= 2) return out
        }
        return Yahoo.ohlc(symbol, range, interval)
    }

    fun closes(symbol: String, range: String = "2y", interval: String = "1d"): List<Pair<Long, Double>> =
        ohlc(symbol, range, interval).map { Pair(it.t, it.close) }
}
