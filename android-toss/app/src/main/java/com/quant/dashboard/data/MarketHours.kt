package com.quant.dashboard.data

/**
 * 장 운영시간 — 실시간 조회를 **언제 돌릴지** 판정.
 *
 * 판정은 서버가 한다(`/api/market`, 토스 `market-calendar` 를 하루 1회 받아 캐시).
 * 폰은 그 결과를 잠깐 들고 있다가 쓴다. 못 받으면 **장중으로 보고** 돈다 —
 * 잘못 멈춰서 값이 안 바뀌는 것보다, 조금 더 부르는 쪽이 낫다.
 */
object MarketHours {
    @Volatile private var open = true
    @Volatile private var label = ""
    @Volatile private var sessions: List<TossApi.Session> = emptyList()
    @Volatile private var at = 0L

    private const val TTL = 5 * 60 * 1000L

    /** 5분마다 한 번만 실제로 묻는다. IO 디스패처에서 호출할 것. */
    fun ensure(force: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!force && now - at < TTL) return
        if (!ServerConfig.isSet()) return
        try {
            val m = Server.market()
            open = m.open
            label = m.label
            sessions = m.sessions
            at = now
        } catch (e: Exception) {
            at = now - TTL / 2      // 실패하면 조금 뒤에 다시 (매번 두드리지 않게)
        }
    }

    /** 시세 틱을 돌려야 하는가. */
    fun anyOpen(): Boolean = open

    /** 그 시장에서 지금 열린 세션 이름 (`US 정규장`). 닫혀 있으면 null. */
    fun labelFor(market: String): String? {
        val now = System.currentTimeMillis() / 1000
        val on = sessions.filter { it.market == market && now in it.start..it.end }
        if (on.isEmpty()) return null
        return "$market " + on.joinToString(" · ") { it.name }
    }

    /** 화면 표시용 전체 라벨. */
    fun label(): String = label

    /**
     * 그 시장에서 **지금 또는 마지막으로** 시작된 세션의 시작 시각.
     * 장 마감 중에는 마지막 세션의 종가가 정상값이라, 그보다 앞선 체결만 '낡음'으로 본다.
     */
    fun sessionStart(market: String): Long? {
        val now = System.currentTimeMillis() / 1000
        return sessions.filter { it.market == market && it.start <= now }.maxOfOrNull { it.start }
    }
}
