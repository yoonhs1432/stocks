package com.quant.dashboard.data

/**
 * 화면이 쓰는 **자료 모양**만 남은 파일.
 *
 * 예전에는 여기서 토스 Open API 를 직접 불렀다. 지금은 집 PC 서버(`Server.kt`)가 그 일을
 * 하고, 폰은 받아 그리기만 한다. 다음 두 가지 때문이다.
 *  1. 토스 API 는 **허용 IP** 안에서만 열린다 — 폰 IP 는 계속 바뀐다.
 *  2. 앱키·시크릿이 폰에 없어야 한다.
 *
 * 이름을 그대로 둔 것은 화면 코드가 `TossApi.Holding` 같은 타입을 그대로 쓰기 때문이다.
 */
object TossApi {

    /** 현재가 한 건. `at` 은 **체결 시각**(없으면 이번 세션에 체결이 없었다는 뜻). */
    data class Quote(
        val price: Double,
        val at: Long?,
        /** 이번(또는 마지막) 세션에 체결이 없었다 — 서버가 판정해 준다. */
        val stale: Boolean = false,
    )

    /** 보유 종목 한 줄. 금액은 **거래 통화 기준**(국내 ₩ / 해외 $). */
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
        val pnlRate: Double,             // 소수비율 (0.1077 = 10.77%)
        val pnlAmountAfterCost: Double,  // 수수료·세금 공제 후
        val pnlRateAfterCost: Double,
        val dailyPnlAmount: Double,      // 당일 손익 (거래 통화 기준)
        val dailyPnlRate: Double,
    ) {
        /** 전일 기준가 — 당일 손익률에서 역산. 실시간 시세로 당일 등락을 다시 계산할 때 쓴다. */
        val basePrice: Double
            get() = if (dailyPnlRate > -1.0 && dailyPnlRate != 0.0) lastPrice / (1 + dailyPnlRate) else lastPrice
    }

    /** 계좌 요약 + 보유 목록. */
    data class Holdings(
        val krwPurchase: Double, val usdPurchase: Double,
        val krwEval: Double, val usdEval: Double,
        val krwPnl: Double, val usdPnl: Double,
        val pnlRate: Double,          // 전체 원화 환산 기준 손익률
        val items: List<Holding>,
        val krwPnlAfterCost: Double = 0.0, val usdPnlAfterCost: Double = 0.0,
        val pnlRateAfterCost: Double = 0.0,
        val krwDailyPnl: Double = 0.0, val usdDailyPnl: Double = 0.0,
        val dailyPnlRate: Double = 0.0,
    )

    /** 거래 세션 1구간 (epoch 초). */
    data class Session(val market: String, val name: String, val start: Long, val end: Long)
}
