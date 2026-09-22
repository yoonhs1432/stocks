package com.quant.dashboard.data

/**
 * 계좌 — **PC 서버에서 받아 둔다.** 이 앱은 토스를 직접 부르지 않는다.
 *
 * 체결내역 가져오기도 서버가 한다(`POST /api/fills`). 매매기록은 PC 에 쌓이므로
 * 폰을 바꿔도 남고, 두 기기에서 따로 세지 않는다.
 */
object TossSync {

    data class Account(
        val holdings: TossApi.Holdings,
        val krwCash: Double,
        val usdCash: Double,
        val rate: Double,
    ) {
        val krwEval: Double get() = holdings.krwEval
        val usdEval: Double get() = holdings.usdEval

        /** 원화 환산 총자산 = 평가금액 + 매수가능금액. */
        fun totalKrw(): Double = krwEval + krwCash + (usdEval + usdCash) * rate

        /** 평가손익 합계(원화 환산). 서버가 이미 원화로 합쳐 준다. */
        fun pnlKrw(): Double = holdings.krwPnl + holdings.usdPnl * rate

        /** 수수료·세금 공제 후 — 서버가 따로 주지 않아 같은 값을 쓴다. */
        fun pnlAfterCostKrw(): Double = pnlKrw()

        /** 당일 손익(원화 환산). */
        fun dailyPnlKrw(): Double = holdings.krwDailyPnl + holdings.usdDailyPnl * rate
    }

    @Volatile private var accountCache: Account? = null
    @Volatile private var accountAt = 0L

    private const val TTL = 60_000L      // 서버도 20초 캐시를 두고 있다

    fun cachedAccount(): Account? = accountCache

    /** 보유 + 예수금 + 환율. IO 디스패처에서 호출할 것. */
    fun account(force: Boolean = false): Account {
        val now = System.currentTimeMillis()
        val hit = accountCache
        if (!force && hit != null && now - accountAt < TTL) return hit
        val a = Server.account(force)
        accountCache = a
        accountAt = now
        return a
    }

    fun clear() {
        accountCache = null
        accountAt = 0L
    }

    /** 체결내역 가져오기 — 서버가 토스에서 받아 PC 에 쌓는다. 누적 건수를 돌려준다. */
    fun importFills(): Int {
        val total = Server.fetchFills()
        Store.refreshTrades()
        return total
    }
}
