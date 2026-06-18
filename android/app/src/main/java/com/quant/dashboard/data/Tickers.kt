package com.quant.dashboard.data

/** 기본 종목 + 표시명 (app.py DEFAULT_TICKERS / TICKER_DISPLAY_NAMES 미러). */
object Tickers {
    const val BASE = "SPY"  // 회귀 기준 자산

    val DEFAULT = listOf(
        "FNGU", "TQQQ", "SOXL", "HIBL", "QPUX", "LABU", "DFEN", "DPST",
        "GDXU", "KORU", "005930", "AVXX", "SPYU", "TARK", "URTY", "TNA",
        "BNKU", "BTC-USD", "ETH-USD", "GLD",
    )

    private val DISPLAY = mapOf(
        "BTC-USD" to "BTC", "ETH-USD" to "ETH", "005930" to "삼전", "000660" to "하닉",
    )

    fun displayName(ticker: String): String = DISPLAY[ticker] ?: ticker
}
