package com.quant.dashboard.data

/**
 * 자산 추이 — **PC 서버가 쌓고 계산한 것**을 받아 둔다.
 *
 * 토스에는 과거 잔고 API 가 없어 누군가는 매일 남겨야 한다. 예전에는 앱이 열릴 때
 * 폰이 남겼고, 그래서 **앱을 안 연 날은 비었다.** 지금은 PC 가 30분마다 남기므로
 * 폰을 안 켜도 빠지지 않는다.
 *
 * 화면은 그리는 중에 이 값을 읽으므로 **여기서 네트워크를 타면 안 된다.**
 * `ensure()` 로 미리 받아 두고, 읽기는 메모리에서만 한다.
 */
object Snapshots {

    private val EMPTY = Server.Series(
        emptyList(), DoubleArray(0), DoubleArray(0), DoubleArray(0), DoubleArray(0), DoubleArray(0),
    )

    @Volatile private var krw: Server.Series = EMPTY
    @Volatile private var usd: Server.Series = EMPTY
    @Volatile private var at = 0L

    private const val TTL = 5 * 60 * 1000L

    /** 원·달러 두 벌을 받아 둔다. IO 디스패처에서 호출할 것. */
    fun ensure(force: Boolean = false) {
        val now = System.currentTimeMillis()
        if (!force && now - at < TTL && krw.dates.isNotEmpty()) return
        if (!ServerConfig.isSet()) return
        try {
            krw = Server.snapshots(false)
            usd = Server.snapshots(true)
            at = now
        } catch (e: Exception) {
            // 못 받으면 있던 값을 그대로 쓴다 — 빈 화면보다 낫다
        }
    }

    /** 그래프용 시계열. 원/달러 환산까지 서버가 끝내서 준다. */
    fun series(usd: Boolean = false): Server.Series = if (usd) this.usd else krw

    /** 평가손익 시계열 — 값이 있는 날만. */
    fun pnls(usd: Boolean = false): List<Pair<String, Double>> {
        val s = series(usd)
        return s.dates.indices
            .filter { it < s.pnl.size && !s.pnl[it].isNaN() }
            .map { s.dates[it] to s.pnl[it] }
    }

    /** [pnls] 와 같은 날짜들의 원금 — 손익률의 기준선. */
    fun pnlPrincipal(usd: Boolean = false): DoubleArray {
        val s = series(usd)
        return s.dates.indices
            .filter { it < s.pnl.size && !s.pnl[it].isNaN() }
            .map { s.principal.getOrElse(it) { Double.NaN } }
            .toDoubleArray()
    }

    fun clear() {
        krw = EMPTY; usd = EMPTY; at = 0L
    }
}
