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
     * 미국 시가총액 상위 30개 (비교 탭 "미장 TOP30" 목록).
     * 온디바이스라 시총 랭킹을 받아올 데이터 소스가 없어 정적 목록으로 둔다 —
     * 순위가 바뀌면 이 목록만 갱신하면 됨. (기준: 2026년 상반기)
     */
    val US_TOP30 = listOf(
        "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "LLY",
        "JPM", "WMT", "V", "XOM", "ORCL", "MA", "UNH", "COST", "JNJ", "HD",
        "PG", "NFLX", "ABBV", "BAC", "AMD", "CRM", "KO", "CVX", "TMUS", "WFC",
    )

    private val DISPLAY = mapOf(
        "BTC-USD" to "BTC", "ETH-USD" to "ETH", "005930" to "삼전", "000660" to "하닉",
    )

    /** 사용자 override > 하드코딩 표시명 > 코드 그대로. */
    fun displayName(ticker: String): String =
        Store.nameOverrides()[ticker] ?: DISPLAY[ticker] ?: ticker

    /** 한국 종목(6자리 코드 = Yahoo .KS) → 원화 표시. */
    fun isKrw(ticker: String): Boolean = ticker.length == 6 && ticker.all { it.isDigit() }

    fun currencySymbol(ticker: String): String = if (isKrw(ticker)) "₩" else "$"

    /** 통화 기호 + 천단위 가격 문자열. 원화는 정수, 달러는 소수 2자리. */
    fun priceLabel(ticker: String, value: Double): String =
        if (isKrw(ticker)) "₩${"%,.0f".format(value)}" else "$${"%,.2f".format(value)}"
}
