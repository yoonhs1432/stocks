package com.quant.dashboard.data

/**
 * 표시명·통화 규칙.
 *
 * 이름과 "원화 종목인가"는 **서버가 알려 준다**(비교 행·설정·보유 목록에 같이 온다).
 * 예전에는 폰이 토스 종목 목록 수천 건을 하루 한 번 받아 두고 직접 찾았는데,
 * 서버가 이미 그 일을 하고 있으므로 받은 값을 기억만 해 두면 된다.
 */
object Tickers {
    const val BASE = "SPY"  // 회귀 기준 자산

    val DEFAULT = listOf(
        "FNGU", "TQQQ", "SOXL", "HIBL", "QPUX", "LABU", "DFEN", "DPST",
        "GDXU", "KORU", "005930", "AVXX", "SPYU", "TARK", "URTY", "TNA",
        "BNKU", "GLD",
    )

    @Volatile private var names: Map<String, String> = emptyMap()
    @Volatile private var krw: Set<String> = emptySet()

    /** 서버에서 받은 이름·통화를 기억해 둔다. 화면 그리는 중에 불리므로 가볍게. */
    @Synchronized
    fun learn(symbol: String, name: String?, isKrw: Boolean) {
        if (symbol.isBlank()) return
        if (!name.isNullOrBlank() && name != symbol && names[symbol] != name) {
            names = names + (symbol to name)
        }
        if (isKrw && symbol !in krw) krw = krw + symbol
    }

    fun displayName(ticker: String): String = names[ticker] ?: ticker

    /**
     * 원화로 표시할 종목인가. 6자리 숫자 코드가 기본이고, 그 규칙을 벗어나는 국내 종목
     * (예: SOL 시리즈)은 서버가 알려 준 값으로 잡는다.
     */
    fun isKrw(ticker: String): Boolean {
        val t = ticker.substringBefore('.')
        if (t.length == 6 && t.all { it.isDigit() }) return true
        return t in krw
    }

    fun currencySymbol(ticker: String): String = if (isKrw(ticker)) "₩" else "$"

    /** 통화 기호 + 천단위 가격 문자열. 원화는 정수, 달러는 소수 2자리. */
    fun priceLabel(ticker: String, value: Double): String =
        if (isKrw(ticker)) "₩${"%,.0f".format(value)}" else "$${"%,.2f".format(value)}"
}
