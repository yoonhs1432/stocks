package com.quant.dashboard.data

/** 기본 종목 + 표시명 (app.py DEFAULT_TICKERS / TICKER_DISPLAY_NAMES 미러). */
object Tickers {
    const val BASE = "SPY"  // 회귀 기준 자산

    val DEFAULT = listOf(
        "FNGU", "TQQQ", "SOXL", "HIBL", "QPUX", "LABU", "DFEN", "DPST",
        "GDXU", "KORU", "005930", "AVXX", "SPYU", "TARK", "URTY", "TNA",
        "BNKU", "BTC-USD", "ETH-USD", "GLD",
    )

    /**
     * 미국 시가총액 상위 30개 — **폴백 목록**.
     * 평소에는 토스 랭킹 API(`Rankings`)가 비교 탭 "미장 TOP" 목록을 채우고,
     * 토스 미연동·조회 실패·집계 없음일 때만 이 정적 목록을 쓴다. (기준: 2026년 상반기)
     */
    val US_TOP30 = listOf(
        "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "LLY",
        "JPM", "WMT", "V", "XOM", "ORCL", "MA", "UNH", "COST", "JNJ", "HD",
        "PG", "NFLX", "ABBV", "BAC", "AMD", "CRM", "KO", "CVX", "TMUS", "WFC",
    )

    private val DISPLAY = mapOf(
        "BTC-USD" to "BTC", "ETH-USD" to "ETH", "005930" to "삼전", "000660" to "하닉",
    )

    /**
     * 사용자 override > 하드코딩 표시명 > (국내 종목만) 토스 유니버스 이름 > 코드 그대로.
     *
     * 국내는 6자리 숫자라 코드만 보면 무슨 종목인지 알 수 없어 이름을 채운다.
     * 미국은 티커 자체가 읽히므로 그대로 둔다 — 표에서 "AAPL"이 "애플"보다 낫다.
     */
    fun displayName(ticker: String): String =
        Store.nameOverrides()[ticker] ?: DISPLAY[ticker]
            ?: (if (isKrw(ticker)) Universe.nameOf(ticker) else null) ?: ticker

    /**
     * 원화로 표시할 종목인가.
     *
     * 6자리 숫자 코드가 기본이지만, `.KS`/`.KQ` 를 붙여 넣은 경우와
     * 6자리가 아닌 국내 종목까지 잡으려고 토스 유니버스(KOSPI·KOSDAQ)도 함께 본다.
     */
    fun isKrw(ticker: String): Boolean {
        val t = ticker.substringBefore('.')
        if (t.length == 6 && t.all { it.isDigit() }) return true
        return Universe.isKr(t)
    }

    fun currencySymbol(ticker: String): String = if (isKrw(ticker)) "₩" else "$"

    /** 통화 기호 + 천단위 가격 문자열. 원화는 정수, 달러는 소수 2자리. */
    fun priceLabel(ticker: String, value: Double): String =
        if (isKrw(ticker)) "₩${"%,.0f".format(value)}" else "$${"%,.2f".format(value)}"
}
