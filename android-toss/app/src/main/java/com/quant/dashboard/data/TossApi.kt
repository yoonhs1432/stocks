package com.quant.dashboard.data

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/** 토스 API 호출 실패. code 는 스펙의 flat 에러 코드(`invalid-token`, `access_denied` 등). */
class TossException(val code: String, val http: Int, message: String) : Exception(message)

/**
 * 토스증권 Open API 클라이언트 — **조회 전용**.
 *
 * 주문·정정·취소(`POST /api/v1/orders`, `/modify`, `/cancel`)와 조건주문은 **의도적으로 구현하지 않는다.**
 * 이 앱은 계좌를 읽기만 한다.
 *
 * - base: `https://openapi.tossinvest.com`
 * - 인증: OAuth2 Client Credentials — `POST /oauth2/token` (form-urlencoded), refresh token 없음.
 *   client 당 유효 토큰은 1개이며 재발급 시 이전 토큰이 즉시 무효화된다.
 * - 계좌 컨텍스트가 필요한 API 는 `X-Tossinvest-Account: {accountSeq}` 헤더 필요.
 * - 응답 envelope: 성공 `{"result": ...}` / 실패 `{"error": {requestId, code, message}}`.
 *   단 `/oauth2/token` 만 OAuth2 표준 형식(`access_token` / `error`).
 * - 모든 수치는 **문자열(decimal)** 로 내려온다.
 */
object TossApi {
    private const val BASE = "https://openapi.tossinvest.com"
    private const val TIMEOUT = 10_000

    // ── 토큰 (메모리에만 보관, 앱 재시작 시 재발급) ──
    @Volatile private var token: String? = null
    @Volatile private var tokenExpiresAt = 0L   // epochMillis
    @Volatile private var tokenKey = ""         // 자격증명이 바뀌면 무효화

    fun clearToken() { token = null; tokenExpiresAt = 0L; tokenKey = "" }

    /** 유효한 access token 반환. 만료 60초 전이면 재발급. IO 스레드에서 호출. */
    @Synchronized
    private fun accessToken(): String {
        val id = BrokerCreds.appKey()
        val secret = BrokerCreds.appSecret()
        if (id.isBlank() || secret.isBlank()) throw TossException("no-credentials", 0, "App Key/Secret이 설정되지 않았습니다")
        val key = "$id:${secret.hashCode()}"
        val cached = token
        if (cached != null && key == tokenKey && System.currentTimeMillis() < tokenExpiresAt - 60_000) return cached

        val body = "grant_type=client_credentials" +
            "&client_id=" + URLEncoder.encode(id, "UTF-8") +
            "&client_secret=" + URLEncoder.encode(secret, "UTF-8")
        val conn = (URL("$BASE/oauth2/token").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
            setRequestProperty("Accept", "application/json")
            connectTimeout = TIMEOUT; readTimeout = TIMEOUT
        }
        try {
            conn.outputStream.use { it.write(body.toByteArray()) }
            val status = conn.responseCode
            val text = (if (status in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()?.use { it.readText() } ?: ""
            if (status !in 200..299) {
                // 토큰 엔드포인트는 OAuth2 표준 에러 형식 (`error` / `error_description`)
                val o = runCatching { JSONObject(text) }.getOrNull()
                val code = o?.optString("error").orEmpty().ifBlank { "http-$status" }
                throw TossException(code, status, tokenErrorMessage(code, o?.optString("error_description")))
            }
            val o = JSONObject(text)
            val t = o.getString("access_token")
            val expiresIn = o.optLong("expires_in", 86_400L)
            token = t
            tokenExpiresAt = System.currentTimeMillis() + expiresIn * 1000L
            tokenKey = key
            return t
        } finally {
            conn.disconnect()
        }
    }

    private fun tokenErrorMessage(code: String, desc: String?): String = when (code) {
        "invalid_client" -> "App Key 또는 Secret이 올바르지 않습니다"
        // 허용 IP 목록 밖에서 호출한 경우. 모바일은 IP가 자주 바뀌므로 자주 마주칠 수 있다.
        "access_denied" -> "허용되지 않은 IP입니다. 토스증권 WTS → 설정 → Open API → 허용 IP 관리에서 현재 IP를 등록하세요"
        "unsupported_grant_type" -> "지원하지 않는 인증 방식입니다"
        else -> if (!desc.isNullOrBlank()) desc else "인증 실패 ($code)"
    }

    // ── 공통 GET ──

    /** 성공 시 envelope 의 `result` 를 문자열로 반환. 실패 시 TossException. */
    private fun get(path: String, query: List<Pair<String, String>> = emptyList(), accountSeq: Long? = null): String {
        val qs = if (query.isEmpty()) "" else "?" + query.joinToString("&") {
            "${it.first}=${URLEncoder.encode(it.second, "UTF-8")}"
        }
        val conn = (URL("$BASE$path$qs").openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            setRequestProperty("Authorization", "Bearer ${accessToken()}")
            setRequestProperty("Accept", "application/json")
            if (accountSeq != null) setRequestProperty("X-Tossinvest-Account", accountSeq.toString())
            connectTimeout = TIMEOUT; readTimeout = TIMEOUT
        }
        try {
            val status = conn.responseCode
            val text = (if (status in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()?.use { it.readText() } ?: ""
            if (status !in 200..299) {
                val err = runCatching { JSONObject(text).optJSONObject("error") }.getOrNull()
                val code = err?.optString("code").orEmpty().ifBlank { "http-$status" }
                // 토큰 만료/무효면 캐시를 버려 다음 호출에서 재발급되게 한다
                if (status == 401) clearToken()
                val msg = err?.optString("message").orEmpty().ifBlank {
                    if (status == 429) "요청 한도를 초과했습니다" else "요청 실패 ($code)"
                }
                throw TossException(code, status, msg)
            }
            return text
        } finally {
            conn.disconnect()
        }
    }

    private fun resultObject(text: String): JSONObject = JSONObject(text).getJSONObject("result")
    private fun resultArray(text: String): JSONArray = JSONObject(text).getJSONArray("result")

    /** 문자열 decimal → Double. null/빈값/파싱실패는 기본값. */
    private fun JSONObject.dec(name: String, def: Double = Double.NaN): Double {
        if (isNull(name)) return def
        return optString(name).toDoubleOrNull() ?: def
    }

    // ── 계좌 ──

    data class TossAccount(val accountNo: String, val accountSeq: Long, val accountType: String)

    /**
     * `GET /api/v1/accounts` — 정상 상태 계좌 목록. 현재는 종합매매(BROKERAGE)만 반환된다.
     * 여기서 얻은 `accountSeq` 가 다른 계좌 API 의 `X-Tossinvest-Account` 헤더 값이다.
     */
    fun accounts(): List<TossAccount> {
        val arr = resultArray(get("/api/v1/accounts"))
        return (0 until arr.length()).mapNotNull { i ->
            val o = arr.optJSONObject(i) ?: return@mapNotNull null
            TossAccount(
                accountNo = o.optString("accountNo"),
                accountSeq = o.optLong("accountSeq"),
                accountType = o.optString("accountType"),
            )
        }
    }

    // ── 보유 자산 ──

    /** 보유 종목 1건. 금액은 모두 **거래 통화 기준** (KR=KRW, US=USD). */
    data class Holding(
        val symbol: String,
        val name: String,
        val marketCountry: String,   // KR | US
        val currency: String,        // KRW | USD
        val quantity: Double,
        val lastPrice: Double,
        val avgPrice: Double,
        val purchaseAmount: Double,
        val evalAmount: Double,
        val pnlAmount: Double,
        val pnlRate: Double,             // 소수비율 (0.1077 = 10.77%). = lastPrice/avgPrice - 1
        val pnlAmountAfterCost: Double,  // 수수료·세금 공제 후
        val pnlRateAfterCost: Double,
        val dailyPnlAmount: Double,      // 당일 손익 (거래 통화 기준)
        val dailyPnlRate: Double,
    ) {
        /** 전일 기준가 — 당일 손익률에서 역산. 실시간 시세로 당일 등락을 다시 계산할 때 쓴다. */
        val basePrice: Double
            get() = if (dailyPnlRate > -1.0 && dailyPnlRate != 0.0) lastPrice / (1 + dailyPnlRate) else lastPrice
    }

    /** 계좌 전체 요약 + 종목 목록. 통화별 합계는 환산 없이 통화별로만 집계된다. */
    data class Holdings(
        val krwPurchase: Double, val usdPurchase: Double,
        val krwEval: Double, val usdEval: Double,
        val krwPnl: Double, val usdPnl: Double,
        val pnlRate: Double,          // 전체 원화 환산 기준 손익률
        val items: List<Holding>,
        // 아래는 토스 앱 화면과 대조하기 위해 API 값을 그대로 들고 있는 것들
        val krwPnlAfterCost: Double = 0.0, val usdPnlAfterCost: Double = 0.0,
        val pnlRateAfterCost: Double = 0.0,
        val krwDailyPnl: Double = 0.0, val usdDailyPnl: Double = 0.0,
        val dailyPnlRate: Double = 0.0,   // 전체 원화 환산 기준 당일 손익률
    )

    /** `GET /api/v1/holdings` — 국내(KR)·미국(US) 주식만 포함. */
    fun holdings(accountSeq: Long): Holdings {
        val r = resultObject(get("/api/v1/holdings", accountSeq = accountSeq))
        fun pair(o: JSONObject?): Pair<Double, Double> =
            Pair(o?.dec("krw", 0.0) ?: 0.0, o?.dec("usd", 0.0) ?: 0.0)

        val (kp, up) = pair(r.optJSONObject("totalPurchaseAmount"))
        val (ke, ue) = pair(r.optJSONObject("marketValue")?.optJSONObject("amount"))
        val pl = r.optJSONObject("profitLoss")
        val (kl, ul) = pair(pl?.optJSONObject("amount"))
        val (kla, ula) = pair(pl?.optJSONObject("amountAfterCost"))
        val dpl = r.optJSONObject("dailyProfitLoss")
        val (kd, ud) = pair(dpl?.optJSONObject("amount"))

        val arr = r.optJSONArray("items") ?: JSONArray()
        val items = (0 until arr.length()).mapNotNull { i ->
            val o = arr.optJSONObject(i) ?: return@mapNotNull null
            val mv = o.optJSONObject("marketValue")
            val p = o.optJSONObject("profitLoss")
            Holding(
                symbol = o.optString("symbol"),
                name = o.optString("name"),
                marketCountry = o.optString("marketCountry"),
                currency = o.optString("currency"),
                quantity = o.dec("quantity", 0.0),
                lastPrice = o.dec("lastPrice", 0.0),
                avgPrice = o.dec("averagePurchasePrice", 0.0),
                purchaseAmount = mv?.dec("purchaseAmount", 0.0) ?: 0.0,
                evalAmount = mv?.dec("amount", 0.0) ?: 0.0,
                pnlAmount = p?.dec("amount", 0.0) ?: 0.0,
                pnlRate = p?.dec("rate", 0.0) ?: 0.0,
                pnlAmountAfterCost = p?.dec("amountAfterCost", 0.0) ?: 0.0,
                pnlRateAfterCost = p?.dec("rateAfterCost", 0.0) ?: 0.0,
                dailyPnlAmount = o.optJSONObject("dailyProfitLoss")?.dec("amount", 0.0) ?: 0.0,
                dailyPnlRate = o.optJSONObject("dailyProfitLoss")?.dec("rate", 0.0) ?: 0.0,
            )
        }
        return Holdings(
            kp, up, ke, ue, kl, ul, pl?.dec("rate", 0.0) ?: 0.0, items,
            krwPnlAfterCost = kla, usdPnlAfterCost = ula,
            pnlRateAfterCost = pl?.dec("rateAfterCost", 0.0) ?: 0.0,
            krwDailyPnl = kd, usdDailyPnl = ud,
            dailyPnlRate = dpl?.dec("rate", 0.0) ?: 0.0,
        )
    }

    // ── 체결 내역 (종료된 주문) ──

    /** 실제로 체결된 주문 1건 (체결 수량 > 0). */
    data class Fill(
        val orderId: String,
        val symbol: String,
        val buy: Boolean,
        val date: String,        // YYYY-MM-DD (체결일, KST)
        val quantity: Double,
        val price: Double,       // 평균 체결가 (거래 통화)
        val currency: String,
    )

    /**
     * `GET /api/v1/orders?status=CLOSED` — 종료된 주문을 커서 페이징으로 모두 모아
     * **실제 체결분만**(execution.filledQuantity > 0) 반환한다.
     * 취소·거부 주문도 부분 체결이 있었다면 그 수량은 실제 매매이므로 포함한다.
     *
     * @param from 조회 시작일 (YYYY-MM-DD, 주문 생성일 기준). null 이면 전체 기간.
     */
    fun fills(accountSeq: Long, from: String? = null, maxPages: Int = 20): List<Fill> {
        val out = ArrayList<Fill>()
        var cursor: String? = null
        var page = 0
        while (page < maxPages) {
            val q = ArrayList<Pair<String, String>>()
            q.add("status" to "CLOSED")
            q.add("limit" to "100")
            if (from != null) q.add("from" to from)
            cursor?.let { q.add("cursor" to it) }
            val r = resultObject(get("/api/v1/orders", q, accountSeq))
            val arr = r.optJSONArray("orders") ?: JSONArray()
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val ex = o.optJSONObject("execution") ?: continue
                val qty = ex.dec("filledQuantity", 0.0)
                if (qty <= 0.0) continue
                val price = ex.dec("averageFilledPrice", Double.NaN)
                if (price.isNaN() || price <= 0.0) continue
                // 체결 시각이 없으면(이론상 없음) 주문 시각으로 대체
                val stamp = ex.optString("filledAt").ifBlank { o.optString("orderedAt") }
                val date = stamp.take(10)
                if (date.length != 10) continue
                out.add(
                    Fill(
                        orderId = o.optString("orderId"),
                        symbol = o.optString("symbol"),
                        buy = o.optString("side") == "BUY",
                        date = date,
                        quantity = qty,
                        price = price,
                        currency = o.optString("currency"),
                    )
                )
            }
            if (!r.optBoolean("hasNext", false)) break
            cursor = r.optString("nextCursor").let { if (r.isNull("nextCursor") || it.isBlank()) null else it }
            if (cursor == null) break
            page++
        }
        return out
    }

    // ── 시세 ──

    /**
     * `GET /api/v1/prices?symbols=…` — 현재가 다건 조회 (요청당 최대 200종목).
     * @return symbol → 현재가. 조회 실패 종목은 누락된다.
     */
    /**
     * 현재가 1건. `at` 은 **체결 시각**이며, 스펙상 "체결 미발생 등으로 시각이 없을 경우 null" 이다 —
     * 즉 `at == null` 이면 이번 세션에 아직 체결이 없어 `price` 가 직전 종가라는 뜻이다.
     */
    data class Quote(val price: Double, val at: Long?)

    fun prices(symbols: List<String>): Map<String, Quote> {
        if (symbols.isEmpty()) return emptyMap()
        val out = LinkedHashMap<String, Quote>()
        for (chunk in symbols.chunked(200)) {
            val arr = resultArray(get("/api/v1/prices", listOf("symbols" to chunk.joinToString(","))))
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val v = o.dec("lastPrice", Double.NaN)
                if (v.isNaN()) continue
                val at = if (o.isNull("timestamp")) null else epochSec(o.optString("timestamp"))
                out[o.optString("symbol")] = Quote(v, at)
            }
        }
        return out
    }

    /**
     * `GET /api/v1/candles` — 일봉. 요청당 최대 200봉이므로 `nextBefore` 로 페이징해 이어붙인다.
     * 응답은 최신순이므로 마지막에 **오래된→최신** 순으로 뒤집어 반환한다.
     *
     * @param count 필요한 봉 수 (2년 ≈ 500 거래일 → 3페이지)
     */
    fun dailyOhlc(symbol: String, count: Int = 520, adjusted: Boolean = true): List<Candle> {
        val out = ArrayList<Candle>(count)
        var before: String? = null
        var guard = 0
        while (out.size < count && guard < 10) {
            val q = ArrayList<Pair<String, String>>()
            q.add("symbol" to symbol)
            q.add("interval" to "1d")
            q.add("count" to minOf(200, count - out.size).coerceAtLeast(1).toString())
            q.add("adjusted" to adjusted.toString())
            // '+' 는 URLEncoder 가 %2B 로 인코딩해 주므로 그대로 전달한다
            before?.let { q.add("before" to it) }
            val r = resultObject(get("/api/v1/candles", q))
            val arr = r.optJSONArray("candles") ?: JSONArray()
            if (arr.length() == 0) break
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val t = epochSec(o.optString("timestamp")) ?: continue
                val c = o.dec("closePrice", Double.NaN)
                if (c.isNaN()) continue
                out.add(
                    Candle(
                        t = t,
                        open = o.dec("openPrice", c),
                        high = o.dec("highPrice", c),
                        low = o.dec("lowPrice", c),
                        close = c,
                    )
                )
            }
            before = r.optString("nextBefore").let { if (r.isNull("nextBefore") || it.isBlank()) null else it }
            if (before == null) break
            guard++
        }
        // 최신순 → 오래된순, 중복 timestamp 제거 (페이지 경계가 inclusive 라 겹칠 수 있음)
        return out.asReversed().distinctBy { it.t }
    }

    /** ISO 8601 offset 문자열 → epoch 초. 실패 시 null. */
    private fun epochSec(iso: String): Long? =
        runCatching { java.time.OffsetDateTime.parse(iso).toEpochSecond() }.getOrNull()

    /** `GET /api/v1/stocks/all` 항목. 미국 종목도 `name` 은 한글로 온다 (AAPL → "애플"). */
    data class ListedStock(
        val symbol: String,
        val name: String,
        val securityType: String,
        val isCommonShare: Boolean,
    )

    /**
     * 마켓별 전체 종목 (토스에서 거래 가능한 것만, 페이지네이션 없음).
     * 마켓당 수천 건이라 **하루 1회 받아 캐시**하는 용도다 (`Universe`).
     * market: KOSPI · KOSDAQ · NYSE · NASDAQ · AMEX · KR_ETC · US_ETC
     */
    fun listStocks(market: String, status: String = "ACTIVE"): List<ListedStock> {
        val arr = resultArray(get("/api/v1/stocks/all", listOf("market" to market, "status" to status)))
        val out = ArrayList<ListedStock>(arr.length())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val sym = o.optString("symbol")
            if (sym.isBlank()) continue
            out.add(
                ListedStock(
                    symbol = sym,
                    name = o.optString("name"),
                    securityType = o.optString("securityType"),
                    isCommonShare = o.optBoolean("isCommonShare", true),
                )
            )
        }
        return out
    }

    // ── 장 운영 시간 ──

    /** 거래 세션 1구간 (epoch 초). */
    data class Session(val market: String, val name: String, val start: Long, val end: Long)

    private fun sessionOf(market: String, name: String, o: JSONObject?): Session? {
        if (o == null) return null
        val a = epochSec(o.optString("startTime")) ?: return null
        val b = epochSec(o.optString("endTime")) ?: return null
        return Session(market, name, a, b)
    }

    /**
     * `GET /api/v1/market-calendar/{KR|US}` — 전일·당일·익일 3영업일의 세션 시간.
     *
     * 미국 정규장은 22:30(KST) 시작해 **다음 날 05:00 에 끝나므로**, 새벽에는 "당일"이 아니라
     * 전 영업일의 세션이 열려 있다. 그래서 3일치를 모두 펼쳐 반환한다.
     */
    fun marketSessions(country: String): List<Session> {
        val r = resultObject(get("/api/v1/market-calendar/$country"))
        val out = ArrayList<Session>()
        for (dayKey in listOf("previousBusinessDay", "today", "nextBusinessDay")) {
            val d = r.optJSONObject(dayKey) ?: continue
            if (country == "KR") {
                val g = d.optJSONObject("integrated") ?: continue
                sessionOf("KR", "프리마켓", g.optJSONObject("preMarket"))?.let { out.add(it) }
                sessionOf("KR", "정규장", g.optJSONObject("regularMarket"))?.let { out.add(it) }
                sessionOf("KR", "애프터마켓", g.optJSONObject("afterMarket"))?.let { out.add(it) }
            } else {
                sessionOf("US", "데이마켓", d.optJSONObject("dayMarket"))?.let { out.add(it) }
                sessionOf("US", "프리마켓", d.optJSONObject("preMarket"))?.let { out.add(it) }
                sessionOf("US", "정규장", d.optJSONObject("regularMarket"))?.let { out.add(it) }
                sessionOf("US", "애프터마켓", d.optJSONObject("afterMarket"))?.let { out.add(it) }
            }
        }
        return out
    }

    // ── 예수금 (매수 가능 금액) ──

    /**
     * `GET /api/v1/buying-power` — 통화별 현금 기반 매수 가능 금액(미수 미발생 기준).
     *
     * ⚠️ 엄밀한 "예수금"이 아니라 **매수 가능 금액**이다. 미결제 대금 등이 반영되면
     *    실제 예수금과 다를 수 있다. 토스가 주는 유일한 현금 지표라 총자산 계산에 이 값을 쓴다.
     */
    fun buyingPower(accountSeq: Long, currency: String): Double {
        val r = resultObject(get("/api/v1/buying-power", listOf("currency" to currency), accountSeq))
        return r.dec("cashBuyingPower", 0.0)
    }

    // ── 환율 ──

    /** `GET /api/v1/exchange-rate` — USD→KRW. 1분 주기 갱신되는 참고용 표시 환율. */
    fun usdKrw(): Double {
        val r = resultObject(
            get("/api/v1/exchange-rate", listOf("baseCurrency" to "USD", "quoteCurrency" to "KRW"))
        )
        return r.dec("rate", Double.NaN)
    }
}
