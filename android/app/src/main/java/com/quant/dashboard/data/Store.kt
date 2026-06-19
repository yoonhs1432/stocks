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
                    val t = arr.getJSONObject(i)
                    list.add(
                        Trade(
                            date = t.getString("date"),
                            type = t.getString("type"),
                            qty = t.getInt("qty"),
                            price = t.getDouble("price"),
                            memo = if (t.has("memo")) t.optString("memo") else null,
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
                arr.put(o)
            }
            obj.put(k, arr)
        }
        try { f("trades.json").writeText(obj.toString(2)) } catch (e: Exception) {}
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

    // ── 종목 리스트 ──
    fun loadTickers(): MutableList<String> {
        val file = f("tickers.json")
        if (dir != null && file.exists()) {
            try {
                val arr = JSONObject(file.readText()).getJSONArray("tickers")
                val out = ArrayList<String>()
                for (i in 0 until arr.length()) out.add(arr.getString(i))
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

    fun removeTicker(t: String) {
        val l = loadTickers()
        if (l.size > 1) { l.remove(t); saveTickers(l) }
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

    /** Yahoo range 문자열: "6mo" / "1y" / "2y". 기본 2y. */
    fun lookbackRange(): String = settings().optString("range", "2y")
    fun setLookbackRange(r: String) { saveSettings(settings().put("range", r)) }

    // ── Gist 연동 ──
    fun gistToken(): String = settings().optString("gist_token", "")
    fun gistId(): String = settings().optString("gist_id", "")
    fun setGist(token: String, id: String) {
        saveSettings(settings().put("gist_token", token.trim()).put("gist_id", id.trim()))
    }

    /** 데스크톱 quant_trade_history.json 포맷 → 로컬 매매기록 덮어쓰기. 종목 수 반환. */
    fun saveTradesFromJson(text: String): Int {
        val obj = JSONObject(text)
        val map = LinkedHashMap<String, MutableList<Trade>>()
        for (key in obj.keys()) {
            val arr = obj.getJSONArray(key)
            val list = ArrayList<Trade>()
            for (i in 0 until arr.length()) {
                val t = arr.getJSONObject(i)
                list.add(Trade(
                    t.getString("date"), t.getString("type"),
                    t.getInt("qty"), t.getDouble("price"),
                    if (t.has("memo")) t.optString("memo") else null,
                ))
            }
            map[key] = list
        }
        saveTrades(map)
        return map.size
    }

    /** 데스크톱 quant_target_tickers.json 포맷 → 로컬 종목 리스트 덮어쓰기. */
    fun saveTickersFromJson(text: String): Int {
        val arr = JSONObject(text).getJSONArray("tickers")
        val out = ArrayList<String>()
        for (i in 0 until arr.length()) out.add(arr.getString(i).trim().uppercase())
        if (out.isNotEmpty()) { saveTickers(out); return out.size }
        return 0
    }
}
