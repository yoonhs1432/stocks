package com.quant.dashboard.data

import java.time.LocalDate

/**
 * 장 운영 시간 — 실시간 시세 틱을 언제 돌릴지 판정.
 *
 * 토스 `/market-calendar` 를 하루 1회 받아 캐시한다. 한국(프리·정규·애프터)과
 * 미국(데이마켓·프리·정규·애프터) 세션을 모두 포함하므로, 기존의 "미국 정규장 09:30~16:00 ET"
 * 하드코딩과 달리 **시간외와 한국장, 휴장일까지 정확히** 판정된다.
 *
 * 토스 미연동이거나 조회 실패 시에는 대략 판정(fallbackOpen)으로 물러난다.
 */
object MarketHours {
    @Volatile private var sessions: List<TossApi.Session> = emptyList()
    @Volatile private var loadedDay = ""

    /** 하루 1회 갱신. IO 디스패처에서 호출. */
    fun ensure(force: Boolean = false) {
        if (!BrokerCreds.isLinked()) return
        val today = LocalDate.now().toString()
        if (!force && loadedDay == today && sessions.isNotEmpty()) return
        val out = ArrayList<TossApi.Session>()
        runCatching { out.addAll(TossApi.marketSessions("KR")) }
        runCatching { out.addAll(TossApi.marketSessions("US")) }
        if (out.isNotEmpty()) { sessions = out; loadedDay = today }
    }

    /** 지금 열려 있는 세션들 (한·미 동시 개장 가능). 캐시가 없으면 빈 리스트. */
    fun openNow(): List<TossApi.Session> {
        val now = System.currentTimeMillis() / 1000
        return sessions.filter { now in it.start..it.end }
    }

    /** 시세 틱을 돌려야 하는지. 캐시가 없으면 대략 판정으로 폴백. */
    fun anyOpen(): Boolean =
        if (sessions.isEmpty()) fallbackOpen() else openNow().isNotEmpty()

    /**
     * 해당 시장(KR/US)에서 **지금 진행 중인 세션의 시작 시각**(epoch 초).
     * 열린 세션이 없으면 이미 끝난 세션 중 가장 최근 것의 시작을 돌려준다 —
     * 장 마감 중에는 마지막 세션의 종가가 정상값이므로 "낡음"으로 보면 안 된다.
     * 캘린더가 없으면 null (판정하지 않음).
     */
    fun sessionStart(market: String): Long? {
        val now = System.currentTimeMillis() / 1000
        return sessions.asSequence()
            .filter { it.market == market && it.start <= now }
            .maxByOrNull { it.start }
            ?.start
    }

    /** 특정 시장(KR/US)에서 지금 열린 세션 이름. 닫혀 있으면 null. */
    fun labelFor(market: String): String? {
        val open = openNow().filter { it.market == market }
        if (open.isEmpty()) return null
        return "$market ${open.joinToString(" · ") { it.name }}"
    }

    /** 화면 표시용 라벨 (예: "US 정규장", "KR 정규장 · US 프리마켓"). 닫혀 있으면 null. */
    fun label(): String? {
        val open = openNow()
        if (open.isEmpty()) return null
        return open.joinToString(" · ") { "${it.market} ${it.name}" }
    }

    /**
     * 토스 미연동/조회 실패 시 폴백 — 한국 정규장(09:00~15:30 KST) 또는
     * 미국 프리~애프터(04:00~20:00 ET) 를 평일에만 열린 것으로 본다. 휴장일은 걸러내지 못한다.
     */
    private fun fallbackOpen(): Boolean {
        val kst = java.time.ZonedDateTime.now(java.time.ZoneId.of("Asia/Seoul"))
        val et = java.time.ZonedDateTime.now(java.time.ZoneId.of("America/New_York"))
        fun weekday(z: java.time.ZonedDateTime) =
            z.dayOfWeek != java.time.DayOfWeek.SATURDAY && z.dayOfWeek != java.time.DayOfWeek.SUNDAY
        val krOpen = weekday(kst) && kst.toLocalTime().let {
            !it.isBefore(java.time.LocalTime.of(9, 0)) && !it.isAfter(java.time.LocalTime.of(15, 30))
        }
        val usOpen = weekday(et) && et.toLocalTime().let {
            !it.isBefore(java.time.LocalTime.of(4, 0)) && !it.isAfter(java.time.LocalTime.of(20, 0))
        }
        return krOpen || usOpen
    }
}
