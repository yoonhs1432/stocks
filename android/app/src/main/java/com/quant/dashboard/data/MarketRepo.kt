package com.quant.dashboard.data

/**
 * 헤더 시장 배지용 — SPY 체제(SMA200+6M) + VIX + 미 10년물 + USD/KRW.
 * app.py get_market_regime / get_macro_indicators 미러 (Yahoo 직접).
 */
object MarketRepo {
    data class Info(
        val regime: String,        // bull/bear/correction/neutral/unknown
        val spyRet6m: Double?,     // 6개월 수익률 (소수)
        val vix: Double?,
        val us10y: Double?,
        val usdkrw: Double?,
    )

    @Volatile private var cache: Info? = null
    @Volatile private var ts = 0L

    /** IO 디스패처에서 호출. 1시간 캐시. */
    suspend fun load(force: Boolean = false): Info {
        val now = System.currentTimeMillis()
        val c = cache
        if (!force && c != null && now - ts < 3_600_000) return c

        var regime = "unknown"; var ret6m: Double? = null
        val spy = Yahoo.closes("SPY", "2y")
        if (spy.size >= 200) {
            val closes = spy.map { it.second }
            val last = closes.last()
            val sma200 = closes.takeLast(200).average()
            ret6m = if (closes.size >= 126) last / closes[closes.size - 126] - 1 else null
            val above = last > sma200
            regime = when {
                ret6m == null -> "neutral"
                above && ret6m > 0.05 -> "bull"
                !above && ret6m < -0.10 -> "bear"
                !above && ret6m <= 0.0 -> "correction"
                else -> "neutral"
            }
        }
        var us10y = Yahoo.closes("^TNX", "1mo").lastOrNull()?.second
        if (us10y != null && us10y > 50) us10y /= 10.0
        val info = Info(
            regime = regime, spyRet6m = ret6m,
            vix = Yahoo.closes("^VIX", "1mo").lastOrNull()?.second,
            us10y = us10y,
            usdkrw = Yahoo.closes("KRW=X", "1mo").lastOrNull()?.second,
        )
        cache = info; ts = now
        return info
    }
}
