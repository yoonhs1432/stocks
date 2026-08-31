package com.quant.dashboard.data

import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * 종목 이름 검색용 유니버스 캐시 — `GET /api/v1/stocks/all`.
 *
 * 티커를 정확히 몰라도 종목을 추가할 수 있게 한다. 토스는 **미국 종목도 한글 이름**을 주므로
 * ("애플" → AAPL) 한글·영문·티커 어느 쪽으로 쳐도 찾히고, 국내 종목은 6자리 코드를
 * 외우지 않아도 된다("삼성전자" → 005930).
 *
 * 일 배치로 갱신되는 저변동 데이터라 **하루 1회**만 받아 `toss_universe.json` 에 저장한다.
 * (마켓당 수천 건, gzip 약 30KB × 3마켓)
 */
object Universe {
    private const val FILE = "toss_universe.json"
    /** 미국 3개 거래소 + 국내 2개. 국내 종목도 이름으로 찾을 수 있어야 한다("삼성전자" → 005930). */
    private val MARKETS = listOf("NASDAQ", "NYSE", "AMEX", "KOSPI", "KOSDAQ")

    /** 캐시 스키마/대상 마켓이 바뀌면 오늘자 캐시라도 다시 받게 하는 표식. */
    private val VERSION = MARKETS.joinToString(",")

    data class Item(val symbol: String, val name: String, val market: String, val type: String)

    @Volatile private var items: List<Item> = emptyList()
    @Volatile private var bySymbol: Map<String, String> = emptyMap()   // 심볼 → 이름 (표시명용)
    @Volatile private var date = ""       // 캐시 생성일 (yyyy-MM-dd)
    @Volatile private var loaded = false

    fun count(): Int { loadFile(); return items.size }
    fun cachedDate(): String { loadFile(); return date }

    private fun today(): String =
        SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())

    /** 파일 캐시를 메모리로 (앱 실행당 1회). */
    @Synchronized
    private fun loadFile() {
        if (loaded) return
        loaded = true
        val f = Store.fileIn(FILE) ?: return
        if (!f.exists()) return
        try {
            val o = JSONObject(f.readText())
            val arr = o.optJSONArray("items") ?: return
            val out = ArrayList<Item>(arr.length())
            for (i in 0 until arr.length()) {
                val e = arr.optJSONObject(i) ?: continue
                out.add(Item(e.optString("s"), e.optString("n"), e.optString("m"), e.optString("t")))
            }
            // 대상 마켓이 늘어났으면(미장 전용 캐시 등) 오늘자여도 다시 받는다
            if (o.optString("v") != VERSION) return
            items = out
            bySymbol = out.associate { it.symbol to it.name }
            date = o.optString("date")
        } catch (e: Exception) {
            // 캐시가 깨졌으면 없는 셈 치고 다시 받는다
        }
    }

    @Synchronized
    private fun save() {
        val f = Store.fileIn(FILE) ?: return
        try {
            val arr = JSONArray()
            for (it in items) {
                arr.put(JSONObject().put("s", it.symbol).put("n", it.name)
                    .put("m", it.market).put("t", it.type))
            }
            f.writeText(JSONObject().put("v", VERSION).put("date", date).put("items", arr).toString())
        } catch (e: Exception) {}
    }

    /** 오늘자 캐시가 있으면 그대로 사용. IO 디스패처에서 호출. 종목 수 반환. */
    fun ensure(force: Boolean = false): Int {
        loadFile()
        if (!force && items.isNotEmpty() && date == today()) return items.size
        if (!BrokerCreds.isLinked()) return items.size
        val out = ArrayList<Item>(9000)
        var ok = 0
        for (m in MARKETS) {
            try {
                TossApi.listStocks(m).forEach { out.add(Item(it.symbol, it.name, m, it.securityType)) }
                ok++
            } catch (e: Exception) {
                // 한 마켓이 실패해도 나머지는 쓴다
            }
        }
        if (ok == 0) return items.size
        // 마켓 일부만 받아졌으면 기존 캐시의 나머지 마켓을 살려 둔다
        val gotMarkets = out.map { it.market }.toSet()
        val kept = items.filter { it.market !in gotMarkets }
        items = out + kept
        bySymbol = items.associate { it.symbol to it.name }
        date = if (ok == MARKETS.size) today() else ""   // 부분 성공은 다음에 다시 시도
        save()
        return items.size
    }

    /**
     * 심볼 → 종목명. **메모리에 이미 올라와 있을 때만** 돌려준다 —
     * 표시명은 화면 그리는 중에 불리므로 여기서 파일을 읽으면 안 된다.
     */
    fun nameOf(symbol: String): String? = if (!loaded) null else bySymbol[symbol]

    /**
     * 티커·이름 검색. 정확도 순: 티커 완전일치 → 티커 접두 → 이름 접두 → 이름/티커 포함.
     * 보통주·ETF 를 우선하고 우선주·워런트류는 뒤로 민다.
     */
    fun search(query: String, limit: Int = 12): List<Item> {
        loadFile()
        val q = query.trim()
        if (q.isEmpty() || items.isEmpty()) return emptyList()
        val u = q.uppercase(Locale.US)
        val scored = ArrayList<Pair<Int, Item>>()
        for (it in items) {
            val sym = it.symbol.uppercase(Locale.US)
            val nm = it.name
            val s = when {
                sym == u -> 0
                sym.startsWith(u) -> 1
                nm.startsWith(q) -> 2
                nm.contains(q) -> 3
                sym.contains(u) -> 4
                else -> continue
            }
            val penalty = if (it.type == "STOCK" || it.type == "ETF" || it.type == "FOREIGN_ETF") 0 else 5
            scored.add((s + penalty) to it)
        }
        return scored.sortedWith(compareBy({ it.first }, { it.second.symbol }))
            .take(limit).map { it.second }
    }
}
