package com.quant.dashboard.data

/**
 * 비교 탭 "미장 TOP" 목록의 종목 선정 — `GET /api/v1/rankings` (미국 시장 전용).
 *
 * 토스에는 **시가총액 랭킹이 없다.** 거래대금 상위(1일)가 대형주 목록에 가장 가까워 기본값으로 둔다.
 * 토스 미연동이거나 조회에 실패하면 하드코딩 목록(`Tickers.US_TOP30`)으로 폴백하므로
 * 이 기능 때문에 비교 탭이 비는 일은 없다.
 */
object Rankings {
    const val COUNT = 30

    /** (API enum, 표시명). */
    val TYPES = listOf(
        "MARKET_TRADING_AMOUNT" to "거래대금",
        "MARKET_TRADING_VOLUME" to "거래량",
        "TOSS_SECURITIES_TRADING_AMOUNT" to "토스대금",
        "TOSS_SECURITIES_TRADING_VOLUME" to "토스수량",
        "TOP_GAINERS" to "급상승",
        "TOP_LOSERS" to "급하락",
    )

    val DURATIONS = listOf(
        "realtime" to "실시간", "1d" to "1일", "1w" to "1주",
        "1mo" to "1개월", "3mo" to "3개월", "6mo" to "6개월", "1y" to "1년",
    )

    /** 급상승·급하락은 realtime 미지원 (400 unsupported-ranking-duration). */
    fun durationsFor(type: String): List<Pair<String, String>> =
        if (type.startsWith("TOP_")) DURATIONS.drop(1) else DURATIONS

    fun typeLabel(type: String): String = TYPES.firstOrNull { it.first == type }?.second ?: type
    fun durationLabel(d: String): String = DURATIONS.firstOrNull { it.first == d }?.second ?: d

    /**
     * 목록 제목 — 예: "미장 거래대금 TOP 30 (1일)".
     * 화면에서 매 리컴포지션마다 호출되므로 설정 파일을 읽지 않고 인자로만 만든다.
     */
    fun titleOf(type: String, duration: String, fallback: Boolean): String =
        if (fallback) "미장 TOP $COUNT"
        else "미장 ${typeLabel(type)} TOP $COUNT (${durationLabel(duration)})"

    private fun usable(): Boolean = BrokerCreds.isLinked()

    // ── 캐시 (10분) ──
    @Volatile private var symbols: List<String> = emptyList()
    @Volatile private var key = ""
    @Volatile private var ts = 0L

    /** 마지막 조회가 폴백(하드코딩 목록)이었는지 — 화면에 사유를 알리기 위해. */
    @Volatile var fallbackReason: String? = null
        private set

    /** 목록 식별자 — OverviewRepo 캐시 무효화 키에 쓴다. */
    fun cacheKey(): String = if (usable()) "${Store.rankType()}|${Store.rankDuration()}" else "static"

    /**
     * 랭킹 종목 심볼 목록. IO 디스패처에서 호출.
     * 실패·빈 응답이면 `Tickers.US_TOP30` 을 돌려주고 사유를 `fallbackReason` 에 남긴다.
     */
    fun symbols(force: Boolean = false): List<String> {
        if (!usable()) {
            fallbackReason = "토스 미연동 — 기본 목록"
            return Tickers.US_TOP30
        }
        val cur = cacheKey()
        val now = System.currentTimeMillis()
        if (!force && symbols.isNotEmpty() && key == cur && now - ts < 600_000) {
            fallbackReason = null
            return symbols
        }
        return try {
            val list = TossApi.rankings(
                type = Store.rankType(),
                marketCountry = "US",
                duration = Store.rankDuration(),
                count = COUNT,
            ).map { it.symbol }
            if (list.isEmpty()) {
                // 집계가 없는 조합(휴장 직후 등)은 에러가 아니라 빈 배열로 온다
                fallbackReason = "랭킹 집계 없음 — 기본 목록"
                Tickers.US_TOP30
            } else {
                symbols = list; key = cur; ts = now; fallbackReason = null
                list
            }
        } catch (e: TossException) {
            fallbackReason = "랭킹 조회 실패(${e.code}) — 기본 목록"
            Tickers.US_TOP30
        } catch (e: Exception) {
            fallbackReason = "랭킹 조회 실패 — 기본 목록"
            Tickers.US_TOP30
        }
    }
}
