package com.quant.dashboard.data

import android.content.Context
import android.content.Intent
import android.net.Uri

/**
 * 토스증권의 **그 종목 주문 화면**으로 건너뛴다.
 *
 * ⚠️ 이 앱은 주문을 내지 않는다. 토스 앱(또는 웹 WTS)의 주문 화면을 열어 줄 뿐이고,
 * 수량·가격 입력과 체결은 전부 거기서 사용자가 한다. 이 앱의 토스 연동은 읽기 전용이다.
 *
 * 주소 형식은 토스증권 웹에서 확인한 것이다.
 *   국내  https://www.tossinvest.com/stocks/A005930/order   (단축코드 앞에 A)
 *   해외  https://www.tossinvest.com/stocks/TSLA/order      (티커 경로)
 *
 * 토스 앱이 이 도메인을 자기 것으로 등록해 뒀으면(앱 링크) **앱이 바로 뜨고**,
 * 아니면 브라우저가 같은 주문 화면(WTS)을 연다. 둘 다 안 되면 false 를 돌려준다.
 */
object TossLink {
    /** 토스 안드로이드 앱 패키지. */
    const val PKG = "viva.republica.toss"

    fun orderUrl(ticker: String): String = "https://www.tossinvest.com/stocks/${code(ticker)}/order"

    /** 국내 6자리 코드는 앞에 `A` 가 붙는다. 거래소 접미사(.KS/.KQ)는 뗀다. */
    fun code(ticker: String): String {
        val t = ticker.trim().uppercase().substringBefore('.')
        return if (t.length == 6 && t.all { it.isDigit() }) "A$t" else t
    }

    /**
     * 주문 화면을 연다. **앱 → 브라우저** 순으로 시도한다.
     * @return 어느 쪽으로든 열었으면 true
     */
    fun openOrder(ctx: Context, ticker: String): Boolean {
        val uri = Uri.parse(orderUrl(ticker))
        // ① 토스 앱에게 직접 건넨다 (앱이 안 깔렸거나 이 주소를 안 받으면 예외가 난다)
        if (start(ctx, Intent(Intent.ACTION_VIEW, uri).setPackage(PKG))) return true
        // ② 아무나 — 보통 브라우저가 토스증권 WTS 주문 화면을 연다
        return start(ctx, Intent(Intent.ACTION_VIEW, uri))
    }

    private fun start(ctx: Context, intent: Intent): Boolean = runCatching {
        ctx.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        true
    }.getOrDefault(false)
}
