package com.quant.dashboard.data

import android.content.Context

import org.json.JSONObject
import java.io.File

/** 매매 기록 1건. desktop trade_history.json과 동일 필드. */
data class Trade(
    val date: String,   // YYYY-MM-DD
    val type: String,   // buy | sell
    /** ⚠️ 소수다. 미국은 0.5주 같은 체결이 있어 Int 로 받으면 0주가 된다. */
    val qty: Double,
    val price: Double,
    /** 체결 출처(orderId) — 서버가 붙여 준다. */
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
    // PC 서버가 토스 체결내역을 받아 쌓는다. 폰은 받아 둔 것을 읽기만 한다
    // (그래서 폰을 바꿔도 남고, 두 기기에서 따로 세지 않는다).

    @Volatile private var trades: LinkedHashMap<String, MutableList<Trade>> = LinkedHashMap()

    fun loadTrades(): LinkedHashMap<String, MutableList<Trade>> = trades

    /** 화면에 보여 줄 기록 — 지금은 전부 체결내역이라 그대로다. */
    fun visibleTrades(): LinkedHashMap<String, MutableList<Trade>> = trades

    /** 서버에서 매매 일지를 다시 받아 둔다. IO 디스패처에서 호출할 것. */
    fun refreshTrades() {
        if (!ServerConfig.isSet()) return
        try { trades = Server.journal() } catch (e: Exception) { /* 있던 걸 그대로 쓴다 */ }
    }

    // ── 종목 목록 · 분석 설정 ── (PC 서버에 있다. 웹으로 봐도 같은 목록이 나온다)

    @Volatile private var serverTickers: List<String> = emptyList()
    @Volatile private var serverMonths = MAX_MONTHS
    @Volatile private var serverTick = 10
    @Volatile private var syncAt = 0L
    @Volatile private var syncOk = false

    /** 서버에서 설정을 **한 번이라도 제대로 받았는지**. 못 받았으면 계속 다시 물어야 한다. */
    fun synced(): Boolean = syncOk

    const val MIN_TICKERS = 3

    /**
     * 서버에서 설정·종목·입금·매매기록을 받아 둔다. **IO 디스패처에서** 호출할 것.
     * @return 목록이 실제로 달라졌으면 true (화면을 다시 그리라는 뜻)
     */
    fun syncFromServer(force: Boolean = false): Boolean {
        if (!ServerConfig.isSet()) return false
        val now = System.currentTimeMillis()
        if (!force && now - syncAt < 60_000L) return false
        return try {
            val s = Server.settings()
            val changed = s.tickers != serverTickers || s.months != serverMonths
            serverTickers = s.tickers
            serverMonths = s.months
            serverTick = s.tickSeconds
            Deposits.set(s.deposits, s.principal)
            syncAt = now
            syncOk = true
            refreshTrades()
            changed
        } catch (e: Exception) {
            false
        }
    }

    fun loadTickers(): MutableList<String> =
        if (serverTickers.isEmpty()) Tickers.DEFAULT.toMutableList()
        else serverTickers.toMutableList()

    /** 종목 추가 — 서버 목록에 넣는다. IO 디스패처에서. */
    fun addTicker(t: String) {
        val u = t.trim().uppercase()
        if (u.isEmpty() || u in serverTickers) return
        Server.addTicker(u)
        syncFromServer(force = true)
    }

    /**
     * 여러 종목을 한 번에 — 콤마·줄바꿈·공백 아무거나 구분자로 받는다
     * (증권사 앱 관심종목을 옮겨 붙여넣는 용도).
     * 반환: (추가된 수, 이미 있어서 건너뛴 수).
     */
    fun addTickers(text: String): Pair<Int, Int> {
        val tokens = text.split(',', '\n', '\r', '\t', ' ', ';')
            .map { it.trim().uppercase() }
            // 국내 코드에 붙은 거래소 접미사는 뗀다 (005930.KS → 005930)
            .map { if (it.endsWith(".KS") || it.endsWith(".KQ")) it.dropLast(3) else it }
            .filter { it.isNotEmpty() }
        if (tokens.isEmpty()) return 0 to 0
        val have = serverTickers.map { it.uppercase() }.toMutableSet()
        var added = 0; var dup = 0
        for (t in tokens) {
            if (have.add(t)) { runCatching { Server.addTicker(t) }.onSuccess { added++ } } else dup++
        }
        if (added > 0) syncFromServer(force = true)
        return added to dup
    }

    fun removeTicker(t: String) {
        if (serverTickers.size <= MIN_TICKERS) return
        Server.removeTicker(t)
        syncFromServer(force = true)
    }

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

    /** 기간 설정의 상한(개월). */
    const val MAX_MONTHS = 24

    /** 분석 조회기간(개월) — 서버 설정. 바꾸면 서버가 일봉을 다시 받는다. */
    fun lookbackMonths(): Int = serverMonths.coerceIn(1, MAX_MONTHS)

    /** IO 디스패처에서 호출할 것. */
    fun setLookbackMonths(m: Int) {
        Server.setMonths(m.coerceIn(1, MAX_MONTHS))
        syncFromServer(force = true)
    }

    /**
     * 실시간 시세 갱신 주기(초). 0 = 끔. 장이 열려 있을 때만 동작한다.
     * `/prices` 는 전 종목을 요청 1번으로 받으므로 짧은 주기도 감당되지만,
     * 너무 짧으면 레이트리밋(MARKET_DATA)과 배터리에 부담이 된다.
     */
    fun tickSeconds(): Int = serverTick

    /** IO 디스패처에서 호출할 것. */
    fun setTickSeconds(v: Int) {
        Server.setTickSeconds(v)
        syncFromServer(force = true)
    }

    /**
     * 분석 탭 시계열에서 사용자가 두 손가락으로 맞춘 x축 구간.
     *
     * 배율이 아니라 **보이는 기간(개월)** 과 **오른쪽 끝이 최신에서 떨어진 정도(개월)** 로 저장한다.
     * 배율은 종목의 전체 기간에 대한 비율이라, 데이터 길이가 다른 종목에 그대로 옮기면
     * 보이는 기간이 달라진다. (미설정 = -1 → 2개월)
     */
    fun chartRangeMonths(): Double = settings().optDouble("chart_range_months", -1.0)
    fun chartRangeEnd(): Double = settings().optDouble("chart_range_end", 0.0)
    fun setChartRange(months: Double, endOffset: Double) {
        saveSettings(settings()
            .put("chart_range_months", months)
            .put("chart_range_end", endOffset))
    }

    /** 포트폴리오 탭 표시 통화 — true = 달러, false = 원. */
    fun portfolioUsd(): Boolean = settings().optBoolean("portfolio_usd", false)
    fun setPortfolioUsd(v: Boolean) { saveSettings(settings().put("portfolio_usd", v)) }

    /** 분석 탭에서 보고 있는 차트 묶음: "scatter"(산점도 2개) | "series"(시계열 4개). */
    fun chartGroup(): String = settings().optString("chart_group", "series")
    fun setChartGroup(v: String) { saveSettings(settings().put("chart_group", v)) }

    /**
     * 분석 탭 시계열의 봉 주기: "1d" | "1m". 토스가 받아 주는 주기는 이 둘뿐이다.
     * 1분 모드에서는 Z·M 을 숨긴다 — 일봉 회귀로 만든 값이라 분봉에 의미가 없다.
     */
    fun barMode(): String = settings().optString("bar_mode", "1d")
    fun setBarMode(v: String) { saveSettings(settings().put("bar_mode", v)) }

    /** 비교 탭에서 보고 있는 시장: "US" | "KR". 한 번에 한쪽만 보여준다. */
    fun compareMarket(): String = settings().optString("compare_market", "US")
    fun setCompareMarket(v: String) { saveSettings(settings().put("compare_market", v)) }
}
