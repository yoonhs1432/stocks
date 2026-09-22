package com.quant.dashboard.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * 비교 표 한 벌 — **PC 서버가 계산해 준 것**을 받아 둔다.
 *
 * 예전에는 폰이 종목마다 일봉을 받아 회귀·Z·M 을 직접 돌렸다. 종목이 스무 개면 일봉
 * 요청이 60번이라 한도(429)에 걸렸고, 캐시·동시요청 제한·백오프를 폰에서 다 떠안아야 했다.
 * 지금은 서버가 그 일을 한다(`web/repo.py`). 폰은 한 번 받아 그리기만 한다.
 */
object OverviewRepo {

    data class Row(
        val ticker: String, val name: String, val price: Double,
        val prevClose: Double,     // 전일 종가 — 실시간 현재가로 등락률을 다시 계산할 때 사용
        val day: Double, val week: Double, val fromHigh: Double,
        val zPct: Double, val mPct: Double, val signal: String,
        val beta: Double, val sigmaPct: Double,
        // 미니 캔들용 당일 시/고/저 (없으면 NaN)
        val open: Double = Double.NaN,
        val high: Double = Double.NaN,
        val low: Double = Double.NaN,
        val holding: Boolean,      // 현재 보유 중 (★)
        val hasHistory: Boolean,   // 과거 매매 이력만 (☆)
    )

    @Volatile private var rows: List<Row> = emptyList()
    @Volatile private var ts = 0L

    /**
     * 마지막 실패 이유. 화면에 **그대로** 띄운다.
     *
     * 예전에는 여기서 예외를 삼키고 화면은 "시세를 가져오지 못했습니다" 한 줄만 보여줬다.
     * PC 가 꺼진 건지, 암호가 틀린 건지, 터널이 내려간 건지 알 길이 없었다.
     */
    @Volatile var lastError: String? = null
        private set

    private const val TTL = 60_000L      // 이보다 자주 물어도 같은 값이다

    fun cached(): List<Row> = rows

    /**
     * 미국·국내를 **둘 다** 받아 하나의 목록으로 준다. 화면이 버튼으로 걸러 쓴다.
     * 서버는 한쪽을 받으면 반대쪽을 미리 계산해 두므로 두 번째 호출은 거의 즉시 온다.
     */
    suspend fun load(force: Boolean = false): List<Row> = withContext(Dispatchers.IO) {
        val now = System.currentTimeMillis()
        if (!force && rows.isNotEmpty() && now - ts < TTL) return@withContext rows
        if (!ServerConfig.isSet()) {
            lastError = "설정 탭에서 집 PC 주소와 접속 암호를 입력하세요"
            return@withContext rows
        }

        val out = LinkedHashMap<String, Row>()
        var ok = false
        var err: String? = null
        for (mk in listOf("US", "KR")) {
            try {
                for (r in Server.compare(mk, force)) out[r.ticker] = r
                ok = true
            } catch (e: Exception) {
                // 한쪽이 실패해도 다른 쪽은 보여준다. 다만 이유는 남긴다.
                err = e.message ?: "PC 에 연결되지 않습니다"
            }
        }
        lastError = if (ok) null else err
        if (!ok) return@withContext rows
        val list = out.values.toList()
        if (list.isNotEmpty()) { rows = list; ts = now }
        list
    }

    fun clear() { rows = emptyList(); ts = 0L; lastError = null }
}
