package com.quant.dashboard.data

/**
 * 비교 탭 TOP 목록의 종목 선정 — `GET /api/v1/rankings`. 미국·국내 둘 다.
 *
 * 토스에는 **시가총액 랭킹이 없다.** 거래대금 상위(1일)가 대형주 목록에 가장 가까워 기본값으로 둔다.
 * 미국은 토스 미연동·조회 실패 시 하드코딩 목록(`Tickers.US_TOP30`)으로 폴백한다.
 * 국내는 폴백 목록을 두지 않았다 — 지어낸 종목코드를 보여주느니 연동이 필요하다고 알리는 편이 낫다.
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

    /** (API 값, 표시명). */
    val MARKETS = listOf("US" to "미국", "KR" to "국내")

    fun marketLabel(m: String): String = MARKETS.firstOrNull { it.first == m }?.second ?: m

    fun typeLabel(type: String): String = TYPES.firstOrNull { it.first == type }?.second ?: type
    fun durationLabel(d: String): String = DURATIONS.firstOrNull { it.first == d }?.second ?: d

    /**
     * 목록 제목 — 예: "미장 거래대금 TOP 30 (1일)".
     * 화면에서 매 리컴포지션마다 호출되므로 설정 파일을 읽지 않고 인자로만 만든다.
     */
    fun titleOf(market: String, type: String, duration: String, fallback: Boolean): String =
        if (fallback) "${marketLabel(market)} TOP $COUNT"
        else "${marketLabel(market)} ${typeLabel(type)} TOP $COUNT (${durationLabel(duration)})"

    private fun usable(): Boolean = BrokerCreds.isLinked()

    // ── 캐시 (10분) ──
    @Volatile private var symbols: List<String> = emptyList()
    @Volatile private var key = ""
    @Volatile private var ts = 0L

    /** 마지막 조회가 폴백(하드코딩 목록)이었는지 — 화면에 사유를 알리기 위해. */
    @Volatile var fallbackReason: String? = null
        private set

    /** 목록 식별자 — OverviewRepo 캐시 무효화 키에 쓴다. */
    fun cacheKey(): String =
        if (usable()) "${Store.rankMarket()}|${Store.rankType()}|${Store.rankDuration()}"
        else "static|${Store.rankMarket()}"

    /**
     * 랭킹 종목 심볼 목록. IO 디스패처에서 호출.
     * 실패·빈 응답이면 미국은 `Tickers.US_TOP30`, 국내는 빈 목록을 돌려주고
     * 사유를 `fallbackReason` 에 남긴다 (화면에 배지로 표시).
     */
    fun symbols(force: Boolean = false): List<String> {
        val market = Store.rankMarket()
        if (!usable()) {
            // 국내는 대체할 정적 목록이 없다
            if (market != "US") {
                fallbackReason = "국내 랭킹은 토스 연동이 필요합니다"
                return emptyList()
            }
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
                marketCountry = market,
                duration = Store.rankDuration(),
                count = COUNT,
            ).map { it.symbol }
            if (list.isEmpty()) {
                // 집계가 없는 조합(휴장 직후 등)은 에러가 아니라 빈 배열로 온다
                fallbackReason = "랭킹 집계 없음" + (if (market == "US") " — 기본 목록" else "")
                fallbackList(market)
            } else {
                symbols = list; key = cur; ts = now; fallbackReason = null
                list
            }
        } catch (e: TossException) {
            fallbackReason = "랭킹 조회 실패(${e.code})" + (if (market == "US") " — 기본 목록" else "")
            fallbackList(market)
        } catch (e: Exception) {
            fallbackReason = "랭킹 조회 실패" + (if (market == "US") " — 기본 목록" else "")
            fallbackList(market)
        }
    }

    private fun fallbackList(market: String): List<String> =
        if (market == "US") Tickers.US_TOP30 else emptyList()
}
