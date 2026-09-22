package com.quant.dashboard.data

import com.quant.dashboard.quant.Quant
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.IOException
import java.net.ConnectException
import java.net.CookieHandler
import java.net.CookieManager
import java.net.CookiePolicy
import java.net.HttpURLConnection
import java.net.URL
import java.net.SocketTimeoutException
import java.net.URLEncoder
import java.net.UnknownHostException
import javax.net.ssl.SSLException
import java.util.zip.GZIPInputStream

/**
 * **집 PC 서버 하나만 부른다.** 이 앱은 더 이상 토스를 직접 부르지 않는다.
 *
 * 왜 이렇게 바꿨나 —
 * ① 토스 API 는 **허용 IP** 안에서만 열린다. 폰 IP 는 계속 바뀌어서 밖에 나가면 막혔다.
 *    PC 는 IP 가 고정이라 거기서만 부르면 된다.
 * ② 앱 키·시크릿이 폰에 없어도 된다. 폰에는 **서버 주소와 접속 암호**만 둔다.
 * ③ 일봉 캐시·장 시간·자산 기록 같은 무거운 일은 PC 가 이미 하고 있다. 폰은 받아 그리기만.
 *
 * 서버는 `web/server.py`. 화면이 쓰는 값의 모양은 그쪽 응답을 그대로 따른다.
 */
object Server {

    /**
     * @param code HTTP 응답 코드. **0 이면 서버가 대답을 못 한 것**(주소·연결 문제).
     * @param unreachable 주소에 **닿지도 못했다**는 뜻 — 보조 주소로 넘어가도 되는 경우.
     *   응답을 받은 뒤의 실패(4xx·5xx·읽기 중 끊김)는 여기 해당하지 않는다. 특히 POST 는
     *   이미 서버에 닿았을 수 있어 다시 보내면 **두 번 저장**될 수 있다.
     */
    class HttpError(val code: Int, message: String, val unreachable: Boolean = false) :
        Exception(message)

    // ── 연결 정보 ──

    fun isSet(): Boolean = ServerConfig.isSet()

    // ── 공통 호출 ──

    private fun enc(v: String): String = URLEncoder.encode(v, "UTF-8")

    /**
     * 옛 서버 대응 — 헤더 인증(`X-Quant-Key`)은 나중에 붙였다. 그 전 서버는 `?key=` 로
     * 쿠키를 심는 길밖에 없어서, 401 이 오면 한 번은 그쪽으로 다시 물어본다.
     * 쿠키를 받아 두려면 프로세스 전역 쿠키 저장소가 필요하다.
     */
    private val cookies = CookieManager().also {
        it.setCookiePolicy(CookiePolicy.ACCEPT_ALL)
        CookieHandler.setDefault(it)
    }

    private fun call(path: String, method: String = "GET", body: String? = null): String {
        try {
            return once(path, method, body)
        } catch (e: HttpError) {
            if (e.code != 401) throw e
        }
        // 옛 서버다 — `?key=` 로 **쿠키만 한 번 받아 두고** 원래 요청을 다시 보낸다.
        // (`?key=` 를 원래 요청에 붙이면 안 된다. 서버가 303 으로 되돌려 보내면서
        //  POST 가 GET 으로 바뀌어, 저장이 조용히 안 되는 일이 생긴다.)
        val tk = ServerConfig.token()
        if (tk.isBlank()) throw HttpError(401, "접속 암호가 없습니다")
        runCatching { once("/api/health?key=" + enc(tk)) }
        return once(path, method, body)
    }

    /**
     * 주소를 **차례로** 시도한다 — 터널이 내려가도 집 와이파이면 랜 주소로 간다.
     * 서버가 대답을 한 순간(4xx 포함) 거기서 끝낸다. 닿지도 못한 경우에만 다음 주소로.
     */
    private fun once(path: String, method: String = "GET", body: String? = null): String {
        val bases = ServerConfig.bases()
        if (bases.isEmpty()) throw HttpError(0, "PC 주소가 설정되지 않았습니다")
        var last: HttpError? = null
        for (b in bases) {
            try {
                val text = attempt(b, path, method, body)
                ServerConfig.noteOk(b)
                return text
            } catch (e: HttpError) {
                if (!e.unreachable) throw e
                last = e
            }
        }
        throw last ?: HttpError(0, "PC 에 연결할 수 없습니다")
    }

    private fun attempt(base: String, path: String, method: String, body: String?): String {
        var conn: HttpURLConnection? = null
        try {
            conn = (URL(base + path).openConnection() as HttpURLConnection).apply {
                requestMethod = method
                connectTimeout = 8_000
                // 첫 조회(일봉 20여 종목)는 PC 에서도 20~30초 걸린다
                readTimeout = 90_000
                setRequestProperty("Accept", "application/json")
                setRequestProperty("Accept-Encoding", "gzip")
                val tk = ServerConfig.token()
                if (tk.isNotBlank()) setRequestProperty("X-Quant-Key", tk)
                if (body != null) {
                    doOutput = true
                    setRequestProperty("Content-Type", "application/json")
                }
            }
            // 연결만 따로 세운다 — 여기서 나는 실패는 "닿지도 못했다"가 확실하다.
            // (한 덩어리로 두면 읽는 중에 난 타임아웃과 구분이 안 되고, 그걸 다른 주소로
            //  다시 보내면 POST 가 두 번 들어갈 수 있다.)
            try { conn.connect() } catch (e: Exception) { throw unreachable(e) }

            if (body != null) conn.outputStream.use { it.write(body.toByteArray()) }
            val code = conn.responseCode
            val raw = (if (code in 200..299) conn.inputStream else conn.errorStream)
                ?: return if (code in 200..299) "" else throw HttpError(code, "요청 실패 ($code)")
            val stream = if (conn.contentEncoding.equals("gzip", true)) GZIPInputStream(raw) else raw
            val text = stream.bufferedReader().use(BufferedReader::readText)
            if (code !in 200..299) {
                val msg = runCatching { JSONObject(text).optString("error") }.getOrNull()
                throw HttpError(
                    code,
                    when {
                        !msg.isNullOrBlank() -> msg
                        // 헤더 인증을 모르는 옛 서버도 여기로 온다 — 둘 다 알려 준다
                        code == 401 -> "접속 암호가 맞지 않거나 PC 서버가 옛 버전입니다"
                        else -> "요청 실패 ($code)"
                    },
                )
            }
            return text
        } catch (e: HttpError) {
            throw e
        } catch (e: SocketTimeoutException) {
            // 연결은 됐는데 읽다가 끊겼다 — 다른 주소로 다시 보내지 않는다
            throw HttpError(0, "PC 가 응답하지 않습니다 · 잠시 뒤 다시 시도합니다")
        } catch (e: IOException) {
            throw HttpError(0, "PC 와 통신하지 못했습니다 · 인터넷 연결을 확인하세요")
        } finally {
            conn?.disconnect()
        }
    }

    /** 연결 단계의 실패를 **뭘 해야 할지 알 수 있는 한국어**로. 자바 원문은 안 보여준다. */
    private fun unreachable(e: Exception): HttpError {
        val msg = when (e) {
            // "Unable to resolve host …" — 이름을 못 찾는다 = PC 가 꺼졌거나 터널이 내려갔다
            is UnknownHostException ->
                "PC 를 찾을 수 없습니다 · PC 가 켜져 있는지, 터널이 살아 있는지 확인하세요"
            is SocketTimeoutException -> "PC 가 응답하지 않습니다 · 잠시 뒤 다시 시도합니다"
            is ConnectException -> "PC 에 연결할 수 없습니다 · 서버가 떠 있는지 확인하세요"
            is SSLException -> "보안 연결에 실패했습니다 · 주소가 https 인지 확인하세요"
            else -> "PC 와 통신하지 못했습니다 · 인터넷 연결을 확인하세요"
        }
        return HttpError(0, msg, unreachable = true)
    }

    private fun obj(path: String): JSONObject = JSONObject(call(path))
    private fun post(path: String, body: JSONObject = JSONObject()): JSONObject {
        val t = call(path, "POST", body.toString())
        return if (t.isBlank()) JSONObject() else JSONObject(t)
    }

    private fun delete(path: String) { call(path, "DELETE") }

    // ── 숫자 읽기 ──
    // 서버는 값이 없으면 null 을 준다. 0 으로 바꿔 읽으면 "못 받았다"와 "0 이다"가 섞인다.

    private fun JSONObject.num(key: String): Double =
        if (isNull(key)) Double.NaN else optDouble(key, Double.NaN)

    private fun JSONObject.arrOf(key: String): DoubleArray {
        val a = optJSONArray(key) ?: return DoubleArray(0)
        return DoubleArray(a.length()) { if (a.isNull(it)) Double.NaN else a.optDouble(it, Double.NaN) }
    }

    private fun JSONObject.longsOf(key: String): LongArray {
        val a = optJSONArray(key) ?: return LongArray(0)
        return LongArray(a.length()) { a.optLong(it, 0L) }
    }

    // ── 비교 ──

    fun compare(market: String, force: Boolean = false): List<OverviewRepo.Row> {
        val o = obj("/api/compare?market=$market" + if (force) "&force=true" else "")
        val arr = o.optJSONArray("rows") ?: JSONArray()
        val out = ArrayList<OverviewRepo.Row>(arr.length())
        for (i in 0 until arr.length()) {
            val r = arr.optJSONObject(i) ?: continue
            val tk = r.optString("ticker")
            out.add(
                OverviewRepo.Row(
                    ticker = tk,
                    name = r.optString("name").ifBlank { tk },
                    price = r.num("price"),
                    prevClose = r.num("prevClose"),
                    day = r.num("day"),
                    week = Double.NaN,        // 서버가 안 준다 — 표에도 안 쓴다
                    fromHigh = Double.NaN,
                    zPct = r.num("zPct"),
                    mPct = r.num("mPct"),
                    signal = r.optString("signal", "hold"),
                    beta = r.num("beta"),
                    sigmaPct = r.num("sigmaPct"),
                    open = r.num("open"),
                    high = r.num("high"),
                    low = r.num("low"),
                    holding = r.optBoolean("holding"),
                    hasHistory = r.optBoolean("hasHistory"),
                ),
            )
            Tickers.learn(tk, r.optString("name").ifBlank { null }, r.optBoolean("krw"))
        }
        return out
    }

    // ── 분석 ──

    /** 한 종목 분석 한 벌 — 서버가 계산해 준 것을 그대로 담는다. */
    data class Analysis(
        val ticker: String,
        val name: String,
        val krw: Boolean,
        val candles: List<Candle>,
        val trades: List<Trade>,
        val result: Quant.Result?,
        val avgPrice: Double,
        val qty: Double,
    )

    fun analysis(ticker: String, force: Boolean = false): Analysis {
        val o = obj("/api/analysis?ticker=${enc(ticker)}" + if (force) "&force=true" else "")
        val bars = ArrayList<Candle>()
        val ca = o.optJSONArray("candles") ?: JSONArray()
        for (i in 0 until ca.length()) {
            val b = ca.optJSONObject(i) ?: continue
            bars.add(
                Candle(
                    b.optLong("t"), b.optDouble("open", Double.NaN), b.optDouble("high", Double.NaN),
                    b.optDouble("low", Double.NaN), b.optDouble("close", Double.NaN),
                ),
            )
        }
        val trades = ArrayList<Trade>()
        val ta = o.optJSONArray("trades") ?: JSONArray()
        for (i in 0 until ta.length()) {
            val t = ta.optJSONObject(i) ?: continue
            trades.add(
                Trade(
                    date = t.optString("date"),
                    type = t.optString("type", "buy"),
                    // 소수 그대로 — 0.5주 체결을 Int 로 깎으면 0주가 된다
                    qty = t.optDouble("qty", 0.0),
                    price = t.optDouble("price", Double.NaN),
                    // 서버가 쓰는 키는 orderId 다 (srcId 는 폰에만 있던 이름)
                    srcId = if (t.isNull("orderId")) null else t.optString("orderId"),
                ),
            )
        }
        val krw = o.optBoolean("krw")
        val name = o.optString("name").ifBlank { ticker }
        Tickers.learn(ticker, name, krw)
        return Analysis(
            ticker = ticker,
            name = name,
            krw = krw,
            candles = bars,
            trades = trades,
            result = resultOf(o.optJSONObject("result"), bars),
            avgPrice = o.num("avgPrice"),
            qty = o.num("qty"),
        )
    }

    /**
     * 서버 계산 결과 → `Quant.Result`.
     *
     * 서버는 `price`(종가 시계열)를 안 준다. 화면은 정규화 값을 금액으로 되돌릴 때
     * **첫날 종가**만 쓰므로 일봉에서 같은 날짜를 찾아 채운다(같은 원본이라 값이 같다).
     */
    private fun resultOf(r: JSONObject?, bars: List<Candle>): Quant.Result? {
        if (r == null) return null
        val dates = r.longsOf("dates")
        if (dates.isEmpty()) return null
        val byDay = HashMap<Long, Double>(bars.size * 2)
        for (b in bars) byDay[b.t / 86400L] = b.close
        val norm = r.arrOf("tickerNorm")
        val base = byDay[dates[0] / 86400L] ?: bars.firstOrNull()?.close ?: Double.NaN
        val price = DoubleArray(dates.size) {
            byDay[dates[it] / 86400L] ?: (norm.getOrElse(it) { Double.NaN } * base)
        }
        return Quant.Result(
            dates = dates,
            price = price,
            tickerNorm = norm,
            spyNorm = r.arrOf("spyNorm"),
            predicted = r.arrOf("predicted"),
            bandUpper = r.arrOf("bandUpper"),
            bandLower = r.arrOf("bandLower"),
            zPct = r.arrOf("zPct"),
            mPct = r.arrOf("mPct"),
            rsi = r.arrOf("rsi"),
            macd = r.arrOf("macd"),
            macdSignal = r.arrOf("macdSignal"),
            beta = r.num("beta"),
            sigmaPct = r.num("sigmaPct"),
            lastPrice = r.num("lastPrice"),
            lastZpct = r.num("lastZpct"),
            lastMpct = r.num("lastMpct"),
            signal = r.optString("signal", "hold"),
        )
    }

    fun minutes(ticker: String): List<Candle> {
        val o = obj("/api/minutes?ticker=${enc(ticker)}")
        val a = o.optJSONArray("candles") ?: JSONArray()
        return (0 until a.length()).mapNotNull { i ->
            val b = a.optJSONObject(i) ?: return@mapNotNull null
            Candle(
                b.optLong("t"), b.optDouble("open", Double.NaN), b.optDouble("high", Double.NaN),
                b.optDouble("low", Double.NaN), b.optDouble("close", Double.NaN),
            )
        }
    }

    // ── 현재가 ──

    /** 심볼 → (현재가, 체결시각, 이번 장 체결 없음). */
    fun prices(symbols: List<String>): Map<String, TossApi.Quote> {
        if (symbols.isEmpty()) return emptyMap()
        val out = LinkedHashMap<String, TossApi.Quote>()
        for (chunk in symbols.chunked(200)) {
            val o = obj("/api/prices?symbols=" + enc(chunk.joinToString(",")))
            val it = o.keys()
            while (it.hasNext()) {
                val k = it.next()
                val q = o.optJSONObject(k) ?: continue
                val px = q.num("price")
                if (px.isNaN()) continue
                // 숫자가 아니면 **모른다**로 둔다. optLong 은 문자열에 0 을 돌려주는데,
                // 그걸 체결 시각으로 믿으면 전 종목이 '이번 장 체결 없음'이 된다.
                val at = (q.opt("at") as? Number)?.toLong()
                out[k] = TossApi.Quote(px, at, q.optBoolean("stale"))
            }
        }
        return out
    }

    // ── 계좌 ──

    fun account(force: Boolean = false): TossSync.Account {
        val o = obj("/api/account" + if (force) "?force=true" else "")
        val items = ArrayList<TossApi.Holding>()
        val a = o.optJSONArray("items") ?: JSONArray()
        for (i in 0 until a.length()) {
            val h = a.optJSONObject(i) ?: continue
            items.add(
                TossApi.Holding(
                    symbol = h.optString("symbol"),
                    name = h.optString("name"),
                    marketCountry = h.optString("marketCountry"),
                    currency = h.optString("currency", "KRW"),
                    quantity = h.optDouble("quantity", 0.0),
                    lastPrice = h.optDouble("lastPrice", 0.0),
                    avgPrice = h.optDouble("avgPrice", 0.0),
                    purchaseAmount = h.optDouble("purchaseAmount", 0.0),
                    evalAmount = h.optDouble("evalAmount", 0.0),
                    pnlAmount = h.optDouble("pnlAmount", 0.0),
                    pnlRate = h.optDouble("pnlRate", 0.0),
                    pnlAmountAfterCost = h.optDouble("pnlAmount", 0.0),
                    pnlRateAfterCost = h.optDouble("pnlRate", 0.0),
                    dailyPnlAmount = h.optDouble("dailyPnlAmount", 0.0),
                    dailyPnlRate = h.optDouble("dailyPnlRate", 0.0),
                ),
            )
            Tickers.learn(h.optString("symbol"), h.optString("name"), h.optString("currency") == "KRW")
        }
        val holdings = TossApi.Holdings(
            krwPurchase = 0.0, usdPurchase = 0.0,
            krwEval = o.optDouble("krwEval", 0.0), usdEval = o.optDouble("usdEval", 0.0),
            krwPnl = o.optDouble("pnlKrw", 0.0), usdPnl = 0.0,   // 서버가 원화로 합쳐 준다
            pnlRate = o.optDouble("pnlRate", 0.0),
            items = items,
            krwDailyPnl = o.optDouble("dailyPnlKrw", 0.0), usdDailyPnl = 0.0,
            dailyPnlRate = o.optDouble("dailyPnlRate", 0.0),
        )
        ServerConfig.saveAccountNo(o.optString("accountNo"))
        return TossSync.Account(
            holdings = holdings,
            krwCash = o.optDouble("krwCash", 0.0),
            usdCash = o.optDouble("usdCash", 0.0),
            rate = o.optDouble("rate", 1400.0),
        )
    }

    // ── 장 운영시간 ──

    data class Market(val open: Boolean, val label: String, val sessions: List<TossApi.Session>)

    fun market(): Market {
        val o = obj("/api/market")
        val a = o.optJSONArray("sessions") ?: JSONArray()
        val ses = (0 until a.length()).mapNotNull { i ->
            val s = a.optJSONObject(i) ?: return@mapNotNull null
            TossApi.Session(
                s.optString("market"), s.optString("name"),
                s.optLong("start"), s.optLong("end"),
            )
        }
        return Market(o.optBoolean("open"), o.optString("label", ""), ses)
    }

    // ── 설정 ──

    data class Settings(
        val months: Int,
        val maxMonths: Int,
        val tickers: List<String>,
        val tickSeconds: Int,
        val deposits: List<Deposit>,
        val principal: Double,
        val trades: Int,
        val version: String,
    )

    fun settings(): Settings {
        val o = obj("/api/settings")
        val ta = o.optJSONArray("tickers") ?: JSONArray()
        val tickers = ArrayList<String>(ta.length())
        for (i in 0 until ta.length()) {
            val t = ta.optJSONObject(i) ?: continue
            val sym = t.optString("ticker")
            tickers.add(sym)
            Tickers.learn(sym, if (t.isNull("name")) null else t.optString("name"), t.optBoolean("krw"))
        }
        val da = o.optJSONArray("deposits") ?: JSONArray()
        val deps = (0 until da.length()).mapNotNull { i ->
            val d = da.optJSONObject(i) ?: return@mapNotNull null
            Deposit(d.optString("date"), d.optDouble("krw", 0.0))
        }
        return Settings(
            months = o.optInt("months", 24),
            maxMonths = o.optInt("maxMonths", 60),
            tickers = tickers,
            tickSeconds = o.optInt("tickSeconds", 10),
            deposits = deps,
            principal = o.optDouble("principal", 0.0),
            trades = o.optInt("trades", 0),
            version = o.optString("version", ""),
        )
    }

    fun addTicker(ticker: String) {
        post("/api/tickers", JSONObject().put("ticker", ticker))
    }

    fun removeTicker(ticker: String) {
        delete("/api/tickers/${enc(ticker)}")
    }

    fun setMonths(n: Int) {
        post("/api/settings/months", JSONObject().put("months", n))
    }

    fun setTickSeconds(sec: Int) {
        post("/api/settings/tick", JSONObject().put("seconds", sec))
    }

    fun addDeposit(date: String, krw: Double) {
        post("/api/deposits", JSONObject().put("date", date).put("krw", krw))
    }

    fun removeDeposit(index: Int) {
        delete("/api/deposits/$index")
    }

    fun fetchFills(): Int = post("/api/fills").optInt("total", 0)

    fun clearCandleCache() {
        post("/api/cache/clear")
    }

    // ── 자산 추이 ──

    /** 서버가 계산해 준 자산 추이 (원/달러 환산까지 끝난 값). */
    data class Series(
        val dates: List<String>,
        val eval: DoubleArray,
        val cash: DoubleArray,
        val total: DoubleArray,
        val pnl: DoubleArray,
        val principal: DoubleArray,
    )

    fun snapshots(usd: Boolean): Series {
        val o = obj("/api/snapshots" + if (usd) "?usd=true" else "")
        val da = o.optJSONArray("dates") ?: JSONArray()
        return Series(
            dates = (0 until da.length()).map { da.optString(it) },
            eval = o.arrOf("eval"),
            cash = o.arrOf("cash"),
            total = o.arrOf("total"),
            pnl = o.arrOf("pnl"),
            principal = o.arrOf("principal"),
        )
    }

    /** 매매 일지 — 종목별로 묶어 돌려준다 (화면이 그 모양을 쓴다). */
    fun journal(): LinkedHashMap<String, MutableList<Trade>> {
        val o = obj("/api/journal?limit=1000")
        val a = o.optJSONArray("trades") ?: JSONArray()
        val out = LinkedHashMap<String, MutableList<Trade>>()
        for (i in 0 until a.length()) {
            val t = a.optJSONObject(i) ?: continue
            val tk = t.optString("ticker")
            out.getOrPut(tk) { ArrayList() }.add(
                Trade(
                    date = t.optString("date"),
                    type = t.optString("type", "buy"),
                    qty = t.optDouble("qty", 0.0),
                    price = t.optDouble("price", Double.NaN),
                    srcId = if (t.isNull("orderId")) null else t.optString("orderId"),
                ),
            )
        }
        return out
    }

    /** 연결 확인 — 설정 화면에서 주소·암호를 넣고 눌러 본다. */
    fun health(): String {
        val o = obj("/api/health")
        return o.optString("status", "ok")
    }
}
