package com.quant.dashboard.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * 비교 표 한 벌 — **PC 서버가 계산해 준 것**을 받아 둔다.
 *
 * 예전에는 폰이 종목마다 일봉을 받아 회귀·Z·M 을 직접 돌렸다. 종목이 스무 개면 일봉
 * 요청이 60번이라 한도(429)에 걸렸고, 캐시·동시요청 제한·백오프를 폰에서 다 떠안아야 했다.
 * 지금은 서버가 그 일을 한다(`web/repo.py`). 폰은 한 번 받아 그리기만 한다.
 *
 * ── 오래간만에 열어도 바로 보이게 ──
 * 서버의 첫 조회는 PC 에서도 20~30초다(일봉 스무 종목). 그동안 빈 화면을 보여 줄 이유가
 * 없어서 **마지막으로 받은 표를 파일에 남겨 두고**, 앱을 켜면 그걸 먼저 그린다.
 * 새 값이 오면 조용히 갈아 끼운다.
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

    /** 시장별로 따로 담는다 — 보고 있는 쪽이 오면 반대쪽을 기다리지 않고 바로 그린다. */
    @Volatile private var byMarket: Map<String, List<Row>> = emptyMap()
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
    private const val FILE = "compare_cache.json"

    fun cached(): List<Row> = byMarket.values.flatten()

    /** 받아 둔 표가 **언제 것인지**. 0 이면 이번에 받은 적 없다(파일에서 꺼낸 것). */
    fun cachedAt(): Long = ts

    /**
     * 한 시장만 받아 합친다. 보고 있는 시장을 먼저 부르면 **반쪽이라도 먼저** 그릴 수 있다.
     * @return 받아 왔으면 true
     */
    suspend fun loadMarket(mk: String, force: Boolean = false): Boolean = withContext(Dispatchers.IO) {
        if (!ServerConfig.isSet()) {
            lastError = "설정 탭에서 집 PC 주소와 접속 암호를 입력하세요"
            return@withContext false
        }
        try {
            val rows = Server.compare(mk, force)
            byMarket = byMarket + (mk to rows)
            ts = System.currentTimeMillis()
            lastError = null
            persist()
            true
        } catch (e: Exception) {
            lastError = e.message ?: "PC 에 연결되지 않습니다"
            false
        }
    }

    /**
     * 미국·국내를 **둘 다** 받아 하나의 목록으로 준다. 화면이 버튼으로 걸러 쓴다.
     * 서버는 한쪽을 받으면 반대쪽을 미리 계산해 두므로 두 번째 호출은 거의 즉시 온다.
     */
    suspend fun load(force: Boolean = false): List<Row> = withContext(Dispatchers.IO) {
        val now = System.currentTimeMillis()
        // ⚠️ 실패 중이면 TTL 을 보지 않는다 — 안 그러면 "8초마다 다시 시도"가 1분 동안
        //    캐시만 돌려주고 실제로는 아무것도 묻지 않는다.
        if (!force && lastError == null && byMarket.isNotEmpty() && now - ts < TTL)
            return@withContext cached()
        var ok = false
        for (mk in listOf("US", "KR")) if (loadMarket(mk, force)) ok = true
        if (ok) lastError = null
        cached()
    }

    fun clear() { byMarket = emptyMap(); ts = 0L; lastError = null }

    // ── 파일에 남겨 두기 ──
    // 앱을 껐다 켜도 **첫 화면이 비어 있지 않게**. 값 자체는 곧 새로 받아 덮어쓴다.

    /** 앱 시작 시 1회. 파일 한 개(수십 KB)라 메인 스레드에서 읽어도 티가 안 난다. */
    fun restore() {
        if (byMarket.isNotEmpty()) return
        val f = Store.fileIn(FILE) ?: return
        if (!f.exists()) return
        try {
            val o = JSONObject(f.readText())
            val out = LinkedHashMap<String, List<Row>>()
            for (mk in o.keys()) {
                val arr = o.optJSONArray(mk) ?: continue
                out[mk] = (0 until arr.length()).mapNotNull { arr.optJSONObject(it)?.let(::rowOf) }
            }
            if (out.isNotEmpty()) byMarket = out
        } catch (e: Exception) {
            // 형식이 바뀌었거나 깨졌다 — 없는 셈 치고 새로 받는다
        }
    }

    private fun persist() {
        val f = Store.fileIn(FILE) ?: return
        try {
            val o = JSONObject()
            for ((mk, rows) in byMarket) {
                val arr = JSONArray()
                for (r in rows) arr.put(jsonOf(r))
                o.put(mk, arr)
            }
            f.writeText(o.toString())
        } catch (e: Exception) {
            // 못 남겨도 화면에는 영향이 없다
        }
    }

    /** ⚠️ JSONObject 는 NaN 을 넣으면 예외다 — 없는 값은 null 로 둔다. */
    private fun JSONObject.putNum(key: String, v: Double): JSONObject =
        put(key, if (v.isNaN()) JSONObject.NULL else v)

    private fun jsonOf(r: Row) = JSONObject()
        .put("ticker", r.ticker).put("name", r.name)
        .putNum("price", r.price).putNum("prevClose", r.prevClose)
        .putNum("day", r.day).putNum("week", r.week).putNum("fromHigh", r.fromHigh)
        .putNum("zPct", r.zPct).putNum("mPct", r.mPct).put("signal", r.signal)
        .putNum("beta", r.beta).putNum("sigmaPct", r.sigmaPct)
        .putNum("open", r.open).putNum("high", r.high).putNum("low", r.low)
        .put("holding", r.holding).put("hasHistory", r.hasHistory)

    private fun rowOf(o: JSONObject): Row? {
        val tk = o.optString("ticker")
        if (tk.isBlank()) return null
        fun n(k: String) = o.optDouble(k, Double.NaN)
        return Row(
            ticker = tk, name = o.optString("name").ifBlank { tk },
            price = n("price"), prevClose = n("prevClose"),
            day = n("day"), week = n("week"), fromHigh = n("fromHigh"),
            zPct = n("zPct"), mPct = n("mPct"), signal = o.optString("signal"),
            beta = n("beta"), sigmaPct = n("sigmaPct"),
            open = n("open"), high = n("high"), low = n("low"),
            holding = o.optBoolean("holding"), hasHistory = o.optBoolean("hasHistory"),
        )
    }
}
