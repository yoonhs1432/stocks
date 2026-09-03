package com.quant.dashboard.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** 매매 기록 1건. desktop trade_history.json과 동일 필드. */
data class Trade(
    val date: String,   // YYYY-MM-DD
    val type: String,   // buy | sell
    val qty: Int,
    val price: Double,
    val memo: String? = null,
    /** 토스 체결 자동 가져오기 출처 (orderId). 중복 가져오기 방지용. 수동 입력은 null. */
    val srcId: String? = null,
)

/**
 * 폰 로컬 영속화 — filesDir에 JSON 저장 (외부 의존성 없음, org.json).
 * MainActivity.onCreate에서 init(applicationContext) 1회 호출.
 */
object Store {
    private var dir: File? = null

    fun init(ctx: Context) {
        dir = ctx.filesDir
    }

    private fun f(name: String) = File(dir, name)

    /** 다른 data 클래스가 filesDir 에 파일을 두기 위한 접근자 (init 전이면 null). */
    fun fileIn(name: String): File? = if (dir == null) null else File(dir, name)

    // ── 매매 기록 ──
    fun loadTrades(): LinkedHashMap<String, MutableList<Trade>> {
        val out = LinkedHashMap<String, MutableList<Trade>>()
        val file = f("trades.json")
        if (dir == null || !file.exists()) return out
        try {
            val obj = JSONObject(file.readText())
            for (key in obj.keys()) {
                val arr = obj.getJSONArray(key)
                val list = ArrayList<Trade>()
                for (i in 0 until arr.length()) {
                    val t = arr.optJSONObject(i) ?: continue
                    val date = t.optString("date", "").trim()
                    val type = t.optString("type", "").trim().lowercase()
                    if (date.isEmpty() || (type != "buy" && type != "sell")) continue
                    list.add(
                        Trade(
                            date = date,
                            type = type,
                            qty = t.optDouble("qty", 0.0).toInt(),
                            price = t.optDouble("price", 0.0),
                            memo = if (t.has("memo") && !t.isNull("memo")) t.optString("memo").ifBlank { null } else null,
                            srcId = if (t.has("src_id") && !t.isNull("src_id")) t.optString("src_id").ifBlank { null } else null,
                        )
                    )
                }
                out[key] = list
            }
        } catch (e: Exception) {
            // 손상 시 빈 맵
        }
        return out
    }

    fun saveTrades(map: Map<String, List<Trade>>) {
        val obj = JSONObject()
        for ((k, v) in map) {
            val arr = JSONArray()
            for (t in v) {
                val o = JSONObject()
                o.put("date", t.date); o.put("type", t.type)
                o.put("qty", t.qty); o.put("price", t.price)
                if (!t.memo.isNullOrBlank()) o.put("memo", t.memo)
                if (!t.srcId.isNullOrBlank()) o.put("src_id", t.srcId)
                arr.put(o)
            }
            obj.put(k, arr)
        }
        try { f("trades.json").writeText(obj.toString(2)) } catch (e: Exception) {}
    }

    /**
     * 화면에 표시할 매매기록 — **토스에서 가져온 체결만**(srcId 있음).
     *
     * 이관 전 다른 증권사에서 손으로 적어 둔 기록은 지금 계좌의 손익과 무관하므로 숨긴다.
     * 파일에서 지우지는 않는다.
     */
    fun visibleTrades(): LinkedHashMap<String, MutableList<Trade>> {
        val all = loadTrades()
        val out = LinkedHashMap<String, MutableList<Trade>>()
        for ((k, v) in all) {
            val kept = v.filter { it.srcId != null }
            if (kept.isNotEmpty()) out[k] = kept.toMutableList()
        }
        return out
    }

    fun addTrade(ticker: String, t: Trade) {
        val m = loadTrades()
        m.getOrPut(ticker) { ArrayList() }.add(t)
        saveTrades(m)
    }

    fun deleteTrade(ticker: String, index: Int) {
        val m = loadTrades()
        m[ticker]?.let {
            if (index in it.indices) {
                it.removeAt(index)
                if (it.isEmpty()) m.remove(ticker)
                saveTrades(m)
            }
        }
    }

    /** 기존 매매 기록의 메모만 수정 (app.py 매매 메모 편집 미러). */
    fun updateTradeMemo(ticker: String, index: Int, memo: String?) {
        val m = loadTrades()
        m[ticker]?.let {
            if (index in it.indices) {
                it[index] = it[index].copy(memo = memo?.ifBlank { null })
                saveTrades(m)
            }
        }
    }

    // ── 종목 리스트 ──
    fun loadTickers(): MutableList<String> {
        val file = f("tickers.json")
        if (dir != null && file.exists()) {
            try {
                val arr = JSONObject(file.readText()).getJSONArray("tickers")
                val out = ArrayList<String>()
                val seen = HashSet<String>()
                for (i in 0 until arr.length()) {
                    val t = arr.getString(i)
                    if (seen.add(t.uppercase())) out.add(t)   // 대소문자 무시 중복 제거
                }
                if (out.isNotEmpty()) return out
            } catch (e: Exception) {}
        }
        return Tickers.DEFAULT.toMutableList()
    }

    fun saveTickers(list: List<String>) {
        val obj = JSONObject()
        obj.put("tickers", JSONArray(list))
        try { f("tickers.json").writeText(obj.toString(2)) } catch (e: Exception) {}
    }

    fun addTicker(t: String) {
        val u = t.trim().uppercase()
        if (u.isEmpty()) return
        val l = loadTickers()
        if (u !in l) { l.add(u); saveTickers(l) }
    }

    /**
     * 여러 종목을 한 번에 추가 — 콤마·줄바꿈·공백·세미콜론 아무거나 구분자로 받는다
     * (토스 앱 관심종목을 옮겨 붙여넣는 용도).
     * 반환: (추가된 수, 이미 있어서 건너뛴 수).
     */
    fun addTickers(text: String): Pair<Int, Int> {
        val tokens = text.split(',', '\n', '\r', '\t', ' ', ';')
            .map { it.trim().uppercase() }
            // 국내 코드에 붙여 넣은 거래소 접미사는 떼어 낸다 (005930.KS → 005930).
            // 안 떼면 6자리 판정에 걸리지 않아 원화 대신 달러로 표시된다.
            .map { if (it.endsWith(".KS") || it.endsWith(".KQ")) it.dropLast(3) else it }
            .filter { it.isNotEmpty() }
        if (tokens.isEmpty()) return 0 to 0
        val l = loadTickers()
        val have = l.map { it.uppercase() }.toMutableSet()
        var added = 0; var dup = 0
        for (t in tokens) {
            if (have.add(t)) { l.add(t); added++ } else dup++
        }
        if (added > 0) saveTickers(l)
        return added to dup
    }

    fun removeTicker(t: String) {
        val l = loadTickers()
        if (l.size > MIN_TICKERS) { l.remove(t); saveTickers(l) }
    }

    const val MIN_TICKERS = 3   // app.py MIN_TICKERS 미러

    // ── 종목 표시명 override (app.py 사용자 이름 override 미러) ──
    fun nameOverrides(): Map<String, String> {
        val o = settings().optJSONObject("names") ?: return emptyMap()
        val out = LinkedHashMap<String, String>()
        for (k in o.keys()) o.optString(k).takeIf { it.isNotBlank() }?.let { out[k] = it }
        return out
    }

    fun setNameOverride(ticker: String, name: String?) {
        val s = settings()
        val o = s.optJSONObject("names") ?: JSONObject()
        if (name.isNullOrBlank()) o.remove(ticker) else o.put(ticker, name.trim())
        saveSettings(s.put("names", o))
    }

    /** 차트 조회기간(개월) — 1개월 단위, 최대 MAX_MONTHS. 기본 2개월. */
    fun chartMonths(): Int = settings().optInt("chart_months", 2).coerceIn(1, MAX_MONTHS)
    fun setChartMonths(v: Int) { saveSettings(settings().put("chart_months", v.coerceIn(1, MAX_MONTHS))) }

    // ── 설정 ──
    private fun settings(): JSONObject {
        val file = f("settings.json")
        if (dir != null && file.exists()) {
            try { return JSONObject(file.readText()) } catch (e: Exception) {}
        }
        return JSONObject()
    }

    private fun saveSettings(o: JSONObject) {
        try { f("settings.json").writeText(o.toString(2)) } catch (e: Exception) {}
    }

    /** MACD·RSI 산점도 X축 정규화 민감도 K (tanh). 기본 0.25. */
    fun macdK(): Double = settings().optDouble("macd_k", 0.25)
    fun setMacdK(v: Double) { saveSettings(settings().put("macd_k", v)) }

    /** 모든 기간 설정의 상한(개월). 슬라이더 최대치이자 시세 조회 상한. */
    const val MAX_MONTHS = 24

    /**
     * 분석 조회기간(개월) — 1개월 단위. 예전에는 6mo/1y/2y 세 단계였고,
     * 그 설정이 남아 있으면 개월 수로 한 번 이관한다.
     */
    fun lookbackMonths(): Int {
        val o = settings()
        if (o.has("range_months")) return o.optInt("range_months", MAX_MONTHS).coerceIn(1, MAX_MONTHS)
        return when (o.optString("range", "2y")) { "6mo" -> 6; "1y" -> 12; else -> MAX_MONTHS }
    }
    fun setLookbackMonths(m: Int) { saveSettings(settings().put("range_months", m.coerceIn(1, MAX_MONTHS))) }

    /**
     * 실시간 시세 갱신 주기(초). 0 = 끔. 장이 열려 있을 때만 동작한다.
     * `/prices` 는 전 종목을 요청 1번으로 받으므로 짧은 주기도 감당되지만,
     * 너무 짧으면 레이트리밋(MARKET_DATA)과 배터리에 부담이 된다.
     */
    fun tickSeconds(): Int = settings().optInt("tick_seconds", 0)
    fun setTickSeconds(v: Int) { saveSettings(settings().put("tick_seconds", v)) }

    /** 자산 추이 곡선 종류: "return"(TWR 수익률) | "pnl"(누적 투자손익) | "total"(총자산). */
    fun equityMode(): String = settings().optString("equity_mode", "return")
    fun setEquityMode(v: String) { saveSettings(settings().put("equity_mode", v)) }

    /**
     * 분석 탭 시계열에서 사용자가 두 손가락으로 맞춘 x축 구간.
     *
     * 배율이 아니라 **보이는 기간(개월)** 과 **오른쪽 끝이 최신에서 떨어진 정도(개월)** 로 저장한다.
     * 배율은 종목의 전체 기간에 대한 비율이라, 데이터 길이가 다른 종목에 그대로 옮기면
     * 보이는 기간이 달라진다. (미설정 = -1 → chartMonths() 를 기본값으로)
     */
    fun chartRangeMonths(): Double = settings().optDouble("chart_range_months", -1.0)
    fun chartRangeEnd(): Double = settings().optDouble("chart_range_end", 0.0)
    fun setChartRange(months: Double, endOffset: Double) {
        saveSettings(settings()
            .put("chart_range_months", months)
            .put("chart_range_end", endOffset))
    }

    /** 분석 탭에서 보고 있는 차트 묶음: "scatter"(산점도 2개) | "series"(시계열 4개). */
    fun chartGroup(): String = settings().optString("chart_group", "series")
    fun setChartGroup(v: String) { saveSettings(settings().put("chart_group", v)) }

    /** 비교 탭에서 보고 있는 시장: "US" | "KR". 한 번에 한쪽만 보여준다. */
    fun compareMarket(): String = settings().optString("compare_market", "US")
    fun setCompareMarket(v: String) { saveSettings(settings().put("compare_market", v)) }
}
