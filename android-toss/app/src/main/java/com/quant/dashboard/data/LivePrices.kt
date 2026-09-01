package com.quant.dashboard.data

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * 실시간 시세 틱 — `/api/v1/prices` 로 **요청 1번에 전 종목**(최대 200) 현재가를 받아온다.
 *
 * 일봉·분석(Z·M·회귀)은 무거워서 캐시 그대로 두고, 화면에 보이는 **현재가·등락률만**
 * 짧은 주기로 갱신한다. Compose 가 관찰하는 상태라 값이 바뀌면 해당 화면만 다시 그려진다.
 */
object LivePrices {
    /** symbol → 현재가 + 체결 시각. 조회되지 않은 종목은 없음. */
    var quotes by mutableStateOf<Map<String, TossApi.Quote>>(emptyMap())
        private set

    /** 마지막 **성공** 시각(epochMillis). 0이면 아직 없음. */
    var updatedAt by mutableStateOf(0L)
        private set

    /** 성공한 틱 횟수 — 화면에서 갱신 여부를 눈으로 확인하는 용도(값이 그대로여도 증가). */
    var tickSeq by mutableStateOf(0)
        private set

    /** 직전 틱에서 **가격이 실제로 바뀐** 심볼 (행 깜빡임용). */
    var changed by mutableStateOf<Set<String>>(emptySet())
        private set

    /** 지금 틱이 안 도는 이유. 정상이면 null. */
    var note by mutableStateOf<String?>(null)
        private set

    @Volatile private var throttleUntil = 0L

    fun price(symbol: String): Double? = quotes[symbol]?.price

    /** 체결 시각(epoch 초). null = 이번 세션 체결 없음 또는 미조회. */
    fun tradedAt(symbol: String): Long? = quotes[symbol]?.at

    /**
     * 이 종목의 현재가가 **직전 종가**인가 (이번 세션에 아직 체결이 없음).
     * 시세가 없으면 false — 모르는 것을 낡았다고 표시하지 않는다.
     */
    fun isStale(symbol: String, market: String): Boolean {
        val q = quotes[symbol] ?: return false
        val start = MarketHours.sessionStart(market) ?: return false
        return q.at == null || q.at < start
    }

    fun setNote(v: String?) { if (note != v) note = v }

    fun clear() {
        quotes = emptyMap(); updatedAt = 0L; changed = emptySet(); throttleUntil = 0L
    }

    /**
     * 한 번 갱신. IO 디스패처에서 호출.
     * 429 를 받으면 잠시 쉬었다 재개한다(서버 한도를 계속 두드리지 않도록).
     */
    fun tick(symbols: List<String>) {
        if (symbols.isEmpty() || !BrokerCreds.isLinked()) return
        val now = System.currentTimeMillis()
        if (now < throttleUntil) {
            note = "한도 초과 · ${((throttleUntil - now) / 1000).coerceAtLeast(1)}초 후 재개"
            return
        }
        try {
            val m = TossApi.prices(symbols)
            if (m.isNotEmpty()) {
                val prev = quotes
                changed = m.filter { (k, v) -> prev[k]?.price != v.price }.keys
                // 조회 실패한 종목의 이전 값은 유지 (깜빡임 방지)
                quotes = prev + m
                updatedAt = now
                tickSeq++
                note = null
            }
        } catch (e: TossException) {
            if (e.http == 429) {
                throttleUntil = now + 30_000
                note = "한도 초과 · 30초 후 재개"
            } else {
                note = "시세 오류(${e.code})"
            }
        } catch (e: Exception) {
            // 일시적 네트워크 오류는 조용히 무시 — 다음 틱에서 재시도
            note = "네트워크 오류"
        }
    }
}
