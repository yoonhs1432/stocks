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
        val pnlRate: Double,         // 소수비율 (0.1077 = 10.77%)
        val dailyPnlRate: Double,
    )

    /** 계좌 전체 요약 + 종목 목록. 통화별 합계는 환산 없이 통화별로만 집계된다. */
    data class Holdings(
        val krwPurchase: Double, val usdPurchase: Double,
        val krwEval: Double, val usdEval: Double,
        val krwPnl: Double, val usdPnl: Double,
        val pnlRate: Double,          // 전체 원화 환산 기준 손익률
        val items: List<Holding>,
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
                dailyPnlRate = o.optJSONObject("dailyProfitLoss")?.dec("rate", 0.0) ?: 0.0,
            )
        }
        return Holdings(kp, up, ke, ue, kl, ul, pl?.dec("rate", 0.0) ?: 0.0, items)
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
    fun prices(symbols: List<String>): Map<String, Double> {
        if (symbols.isEmpty()) return emptyMap()
        val out = LinkedHashMap<String, Double>()
        for (chunk in symbols.chunked(200)) {
            val arr = resultArray(get("/api/v1/prices", listOf("symbols" to chunk.joinToString(","))))
            for (i in 0 until arr.length()) {
                val o = arr.optJSONObject(i) ?: continue
                val v = o.dec("lastPrice", Double.NaN)
                if (!v.isNaN()) out[o.optString("symbol")] = v
            }
        }
        return out
    }

    /**
     * `GET /api/v1/candles` — 일봉. 요청당 최대 200봉이므로 `nextBefore` 로 페이징해 이어붙인다.
     * 응답은 최신순이므로 마지막에 **오래된→최신** 순으로 뒤집어 반환한다 (Yahoo.ohlc 와 동일 규약).
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

    /** (epochSec, close) — Yahoo.closes 와 동일 규약. */
    fun dailyCloses(symbol: String, count: Int = 520): List<Pair<Long, Double>> =
        dailyOhlc(symbol, count).map { Pair(it.t, it.close) }

    /** ISO 8601 offset 문자열 → epoch 초. 실패 시 null. */
    private fun epochSec(iso: String): Long? =
        runCatching { java.time.OffsetDateTime.parse(iso).toEpochSecond() }.getOrNull()

    // ── 종목 기본 정보 (커버리지 확인) ──

    data class StockInfo(
        val symbol: String,
        val name: String,
        val market: String,        // KOSPI | KOSDAQ | NYSE | NASDAQ | AMEX | KR_ETC | US_ETC
        val securityType: String,  // STOCK | ETF | ETN | FOREIGN_ETF …
        val status: String,        // SCHEDULED | ACTIVE | DELISTED
        val currency: String,
    )

    /**
     * `GET /api/v1/stocks?symbols=…` — 토스가 취급하는 종목의 기본 정보 (요청당 최대 200).
     * **결과에 없는 심볼 = 토스에서 조회·거래되지 않는 종목**이므로 커버리지 확인에 쓴다.
     *
     * 알 수 없는 심볼이 섞이면 배치 전체가 404 로 실패할 수 있어, 실패 시 한 종목씩 다시 물어
     * 어떤 것이 빠지는지 가려낸다.
     */
    fun stocks(symbols: List<String>): Map<String, StockInfo> {
        if (symbols.isEmpty()) return emptyMap()
        val out = LinkedHashMap<String, StockInfo>()
        for (chunk in symbols.chunked(200)) {
            try {
                parseStocks(get("/api/v1/stocks", listOf("symbols" to chunk.joinToString(",")))) { out[it.symbol] = it }
            } catch (e: Exception) {
                // 배치 실패 → 개별 조회로 가려내기
                for (sym in chunk) {
                    try {
                        parseStocks(get("/api/v1/stocks", listOf("symbols" to sym))) { out[it.symbol] = it }
                    } catch (e2: Exception) {
                        // 이 심볼은 토스에 없음 — 결과에서 빠진 채로 둔다
                    }
                }
            }
        }
        return out
    }

    private inline fun parseStocks(text: String, add: (StockInfo) -> Unit) {
        val arr = resultArray(text)
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            add(
                StockInfo(
                    symbol = o.optString("symbol"),
                    name = o.optString("name"),
                    market = o.optString("market"),
                    securityType = o.optString("securityType"),
                    status = o.optString("status"),
                    currency = o.optString("currency"),
                )
            )
        }
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
