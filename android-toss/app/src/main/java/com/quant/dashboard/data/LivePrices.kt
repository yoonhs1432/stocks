package com.quant.dashboard.data

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * 실시간 시세 틱 — `/api/v1/prices` 로 **요청 1번에 전 종목**(최대 200) 현재가를 받아온다.
 *
 * 일봉·분석(Z·M·회귀)은 무거워서 5분 캐시 그대로 두고, 화면에 보이는 **현재가·등락률만**
 * 짧은 주기로 갱신한다. Compose 가 관찰하는 상태라 값이 바뀌면 해당 화면만 다시 그려진다.
 */
object LivePrices {
    /** symbol → 마지막 체결가. 조회되지 않은 종목은 없음. */
    var prices by mutableStateOf<Map<String, Double>>(emptyMap())
        private set

    /** 마지막 갱신 시각(epochMillis). 0이면 아직 없음. */
    var updatedAt by mutableStateOf(0L)
        private set

    /** 연속 429(한도 초과) 횟수 — 백오프에 사용. */
    @Volatile private var throttleUntil = 0L

    fun price(symbol: String): Double? = prices[symbol]

    fun clear() { prices = emptyMap(); updatedAt = 0L; throttleUntil = 0L }

    /**
     * 한 번 갱신. IO 디스패처에서 호출.
     * 429 를 받으면 잠시 쉬었다 재개한다(서버 한도를 계속 두드리지 않도록).
     */
    fun tick(symbols: List<String>) {
        if (symbols.isEmpty() || !BrokerCreds.isLinked()) return
        val now = System.currentTimeMillis()
        if (now < throttleUntil) return
        try {
            val m = TossApi.prices(symbols)
            if (m.isNotEmpty()) {
                // 조회 실패한 종목의 이전 값은 유지 (깜빡임 방지)
                prices = prices + m
                updatedAt = now
            }
        } catch (e: TossException) {
            if (e.http == 429) throttleUntil = now + 30_000
        } catch (e: Exception) {
            // 일시적 네트워크 오류는 조용히 무시 — 다음 틱에서 재시도
        }
    }
}
