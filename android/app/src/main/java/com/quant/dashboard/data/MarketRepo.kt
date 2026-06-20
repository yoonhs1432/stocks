package com.quant.dashboard.data

/**
 * 헤더 시장 배지용 — SPY 체제(SMA200+6M) + VIX + 미 10년물 + USD/KRW.
 * app.py get_market_regime / get_macro_indicators 미러 (Yahoo 직접).
 */
object MarketRepo {
    data class Info(
        val spy: Double?,       // SPY 일간 등락 %
        val nasdaq: Double?,    // NASDAQ(^IXIC) 일간 등락 %
        val kospi: Double?,     // KOSPI(^KS11) 일간 등락 %
        val us10y: Double?,     // 미 10년물 금리 %
        val usdkrw: Double?,    // USD/KRW
    )

    @Volatile private var cache: Info? = null
    @Volatile private var ts = 0L

    /** 직전 종가 대비 당일 등락률(%). */
    private fun dayPct(symbol: String): Double? {
        val c = Yahoo.closes(symbol, "5d").map { it.second }
        return if (c.size >= 2 && c[c.size - 2] > 0) (c.last() / c[c.size - 2] - 1) * 100 else null
    }

    /** IO 디스패처에서 호출. 1시간 캐시. */
    suspend fun load(force: Boolean = false): Info {
        val now = System.currentTimeMillis()
        val c = cache
        if (!force && c != null && now - ts < 3_600_000) return c

        var us10y = Yahoo.closes("^TNX", "5d").lastOrNull()?.second
        if (us10y != null && us10y > 50) us10y /= 10.0
        val info = Info(
            spy = dayPct("SPY"),
            nasdaq = dayPct("^IXIC"),
            kospi = dayPct("^KS11"),
            us10y = us10y,
            usdkrw = Yahoo.closes("KRW=X", "5d").lastOrNull()?.second,
        )
        cache = info; ts = now
        return info
    }
}
