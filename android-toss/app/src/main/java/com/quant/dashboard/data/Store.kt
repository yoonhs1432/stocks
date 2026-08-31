package com.quant.dashboard.data

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.time.LocalDate

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
     * 화면에 표시할 매매기록. 토스 모드에서는 이관 전 수기 기록(srcId 없음)을 숨긴다.
     * 파일에서 지우지는 않으므로 토스 모드를 끄면 그대로 돌아온다.
     */
    fun visibleTrades(): LinkedHashMap<String, MutableList<Trade>> {
        val all = loadTrades()
        if (!tossMode()) return all
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

    // ── 개별 종목 표식 (개별/ETF 필터용, app.py individual_tickers 미러) ──
    fun individualTickers(): Set<String> {
        val arr = settings().optJSONArray("individual") ?: return emptySet()
        val out = HashSet<String>()
        for (i in 0 until arr.length()) out.add(arr.getString(i))
        return out
    }

    /** 한국 6자리 종목은 기본적으로 개별로 간주 + 사용자 지정. */
    fun isIndividual(ticker: String): Boolean =
        ticker in individualTickers() || (ticker.length == 6 && ticker.all { it.isDigit() })

    fun setIndividual(ticker: String, individual: Boolean) {
        val cur = individualTickers().toMutableSet()
        if (individual) cur.add(ticker) else cur.remove(ticker)
        saveSettings(settings().put("individual", JSONArray(cur.toList())))
    }

    /** 차트 조회기간(개월) — 1개월 단위, 최대 MAX_MONTHS. 기본 2개월. */
    fun chartMonths(): Int = settings().optInt("chart_months", 2).coerceIn(1, MAX_MONTHS)
    fun setChartMonths(v: Int) { saveSettings(settings().put("chart_months", v.coerceIn(1, MAX_MONTHS))) }

    /**
     * 포트폴리오 자산추이 x축 기간(개월) — 1개월 단위, 최대 MAX_MONTHS.
     * 예전 "전체"(600) 설정이 남아 있으면 상한으로 맞춘다.
     */
    fun equityMonths(): Int = settings().optInt("equity_months", 2).coerceIn(1, MAX_MONTHS)
    fun setEquityMonths(v: Int) { saveSettings(settings().put("equity_months", v.coerceIn(1, MAX_MONTHS))) }

    // ── 기준일(As-of) 시뮬레이션 ──
    /** 설정된 기준일 (ISO yyyy-MM-dd). 미설정/공백이면 null. */
    fun asofDate(): String? = settings().optString("asof", "").ifBlank { null }

    fun setAsofDate(s: String?) { saveSettings(settings().put("asof", s?.trim() ?: "")) }

    private fun asofCutoffSec(): Long? {
        val d = asofDate() ?: return null
        return try { LocalDate.parse(d).toEpochDay() * 86400L + 86_399L } catch (e: Exception) { null }
    }

    /** 기준일 이후 데이터를 잘라낸 시계열. 기준일 미설정 시 원본 그대로. */
    fun sliceAsof(series: List<Pair<Long, Double>>): List<Pair<Long, Double>> {
        val cut = asofCutoffSec() ?: return series
        return series.filter { it.first <= cut }
    }

    fun sliceAsofCandles(series: List<Candle>): List<Candle> {
        val cut = asofCutoffSec() ?: return series
        return series.filter { it.t <= cut }
    }

    // ── 설정 (시드, 분석 기간) ──
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

    fun seedUsd(): Double = settings().optDouble("seed", 20_000.0)
    fun setSeedUsd(v: Double) { saveSettings(settings().put("seed", v)) }

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
     * 요청 개월 수를 덮는 가장 작은 Yahoo range 토큰.
     * Yahoo 는 1개월 단위 구간을 받지 않으므로 넉넉히 받아 `Quotes` 가 정확히 잘라낸다.
     */
    fun rangeToken(months: Int): String = when {
        months <= 3 -> "3mo"
        months <= 6 -> "6mo"
        months <= 12 -> "1y"
        else -> "2y"
    }


    /**
     * 토스 기반 모드 — 포트폴리오·총자산을 토스 계좌 실측으로 표시하고,
     * 이관 전 수기 매매기록(srcId 없는 기록)은 화면에서 숨긴다. 기록 파일은 지우지 않는다.
     */
    fun tossMode(): Boolean = settings().optBoolean("toss_mode", false)
    fun setTossMode(v: Boolean) { saveSettings(settings().put("toss_mode", v)) }

    /**
     * 실시간 시세 갱신 주기(초). 0 = 끔. 장이 열려 있을 때만 동작한다.
     * `/prices` 는 전 종목을 요청 1번으로 받으므로 짧은 주기도 감당되지만,
     * 너무 짧으면 레이트리밋(MARKET_DATA)과 배터리에 부담이 된다.
     */
    fun tickSeconds(): Int = settings().optInt("tick_seconds", 0)
    fun setTickSeconds(v: Int) { saveSettings(settings().put("tick_seconds", v)) }

    /**
     * 비교 탭 미장 TOP 목록의 랭킹 기준 (`GET /api/v1/rankings`).
     * 토스에 시가총액 랭킹이 없어 거래대금 상위(1일)를 기본값으로 둔다.
     */
    /** 비교 탭 TOP 목록의 시장: "US" | "KR". */
    fun rankMarket(): String = settings().optString("rank_market", "US")
    fun setRankMarket(v: String) { saveSettings(settings().put("rank_market", v)) }

    fun rankType(): String = settings().optString("rank_type", "MARKET_TRADING_AMOUNT")
    fun setRankType(v: String) { saveSettings(settings().put("rank_type", v)) }

    fun rankDuration(): String = settings().optString("rank_duration", "1d")
    fun setRankDuration(v: String) { saveSettings(settings().put("rank_duration", v)) }

    /**
     * 토스 조회가 실패했을 때 Yahoo 로 대신 받을지. 기본 켬.
     *
     * **끄면** 토스가 실패한 종목은 값이 아예 안 나온다. 두 소스의 종가가 달라
     * 등락률이 어긋나는 상황에서, 조용히 다른 소스 값을 보여주는 대신
     * "토스가 실패했다"는 사실 자체를 드러내려는 스위치다.
     */
    fun yahooFallback(): Boolean = settings().optBoolean("yahoo_fallback", true)
    fun setYahooFallback(v: Boolean) { saveSettings(settings().put("yahoo_fallback", v)) }

    /** 시세를 토스 API로 받을지 (기본 꺼짐 — 레이트리밋·종목 커버리지 확인 전까지 Yahoo 유지). */
    fun tossQuotes(): Boolean = settings().optBoolean("toss_quotes", false)
    fun setTossQuotes(v: Boolean) { saveSettings(settings().put("toss_quotes", v)) }

    // ── 데스크톱 JSON 가져오기 ──
    // Gist 연동을 제거해 지금은 호출부가 없다. 파일 선택으로 가져오기를 붙일 때 재사용하려고 남겨 둔다.

    /** 데스크톱 quant_trade_history.json 포맷 → 로컬 매매기록 덮어쓰기. 종목 수 반환.
     *  데스크톱은 누락 필드를 r.get('qty',0)처럼 관대하게 다루므로 여기서도 opt*로 안전 파싱. */
    fun saveTradesFromJson(text: String): Int {
        val obj = JSONObject(text)
        val map = LinkedHashMap<String, MutableList<Trade>>()
        for (key in obj.keys()) {
            val arr = obj.optJSONArray(key) ?: continue
            val list = ArrayList<Trade>()
            for (i in 0 until arr.length()) {
                val t = arr.optJSONObject(i) ?: continue
                val date = t.optString("date", "").trim()
                val type = t.optString("type", "").trim().lowercase()
                if (date.isEmpty() || (type != "buy" && type != "sell")) continue
                // qty: 정수/실수/문자열 무엇이든 관대하게
                val qty = when {
                    t.has("qty") && !t.isNull("qty") -> t.optDouble("qty", 0.0).toInt()
                    t.has("quantity") && !t.isNull("quantity") -> t.optDouble("quantity", 0.0).toInt()
                    t.has("shares") && !t.isNull("shares") -> t.optDouble("shares", 0.0).toInt()
                    else -> 0
                }
                val price = t.optDouble("price", 0.0)
                val memo = if (t.has("memo") && !t.isNull("memo")) t.optString("memo").ifBlank { null } else null
                list.add(Trade(date, type, qty, price, memo))
            }
            if (list.isNotEmpty()) map[key] = list
        }
        saveTrades(map)
        return map.size
    }

    /** 데스크톱 quant_target_tickers.json 포맷 → 로컬 종목 리스트 덮어쓰기. */
    fun saveTickersFromJson(text: String): Int {
        val obj = JSONObject(text)
        val arr = obj.optJSONArray("tickers") ?: return 0
        val out = ArrayList<String>()
        for (i in 0 until arr.length()) {
            val s = arr.optString(i, "").trim().uppercase()
            if (s.isNotEmpty()) out.add(s)
        }
        if (out.isNotEmpty()) { saveTickers(out); return out.size }
        return 0
    }

    /** 데스크톱 quant_settings.json → 개별/ETF·이름·시드·차트기간 반영. 개별 종목 수 반환. */
    fun saveSettingsFromJson(text: String): Int {
        val obj = JSONObject(text)
        val s = settings()
        var nIndiv = 0
        // 개별 종목 (individual_tickers) → ETF/개별 구분
        obj.optJSONArray("individual_tickers")?.let { arr ->
            val set = ArrayList<String>()
            for (i in 0 until arr.length()) arr.optString(i, "").trim().uppercase()
                .takeIf { it.isNotEmpty() }?.let { set.add(it) }
            s.put("individual", JSONArray(set)); nIndiv = set.size
        }
        // 종목 표시명 (display_name_overrides)
        obj.optJSONObject("display_name_overrides")?.let { names ->
            val out = JSONObject()
            for (k in names.keys()) names.optString(k, "").takeIf { it.isNotBlank() }
                ?.let { out.put(k.trim().uppercase(), it) }
            s.put("names", out)
        }
        // 시드 / 차트 조회기간 (있으면)
        if (obj.has("seed_usd")) obj.optDouble("seed_usd", 0.0).takeIf { it > 0 }?.let { s.put("seed", it) }
        if (obj.has("view_months")) obj.optInt("view_months", 0).takeIf { it > 0 }?.let { s.put("chart_months", it) }
        saveSettings(s)
        return nIndiv
    }
}
