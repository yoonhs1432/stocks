package com.quant.dashboard.data

/** 기본 종목 + 표시명 (app.py DEFAULT_TICKERS / TICKER_DISPLAY_NAMES 미러). */
object Tickers {
    const val BASE = "SPY"  // 회귀 기준 자산

    val DEFAULT = listOf(
        "FNGU", "TQQQ", "SOXL", "HIBL", "QPUX", "LABU", "DFEN", "DPST",
        "GDXU", "KORU", "005930", "AVXX", "SPYU", "TARK", "URTY", "TNA",
        "BNKU", "GLD",
    )

    private val DISPLAY = mapOf(
        "005930" to "삼전", "000660" to "하닉",
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
