package com.quant.dashboard.data

import android.content.Context
import android.content.Intent
import android.net.Uri
import com.quant.dashboard.TossOpenActivity

/**
 * 토스증권의 **그 종목 주문 화면**으로 건너뛴다.
 *
 * ⚠️ 이 앱은 주문을 내지 않는다. 주문 화면을 열어 줄 뿐이고 수량·가격·체결은 전부
 * 거기서 사용자가 한다. 이 앱의 토스 연동은 읽기 전용이다.
 *
 * ── 왜 이렇게 복잡한가 ──
 * 토스 앱은 `tossinvest.com` 을 자기 주소로 등록해 두지 않았다(앱 링크 없음). 그래서
 * 주소를 그냥 던지면 **브라우저**가 열리고, 거기서 [앱 열기]를 한 번 더 눌러야 한다.
 * 그 버튼이 쏘는 게 `supertoss://…` 같은 앱 전용 주소인데 공개된 문서가 없다.
 *
 * 그래서 **한 번은 배우고, 그 뒤로는 바로 간다.**
 *  ① 기억해 둔 주소가 있으면 → 토스 앱으로 직행 (브라우저를 안 거친다)
 *  ② 없으면 → 우리 앱 안의 작은 웹 화면으로 주문 페이지를 열고, 그 페이지가 앱으로
 *     넘어가려는 순간을 가로채 **그 주소를 기억한 뒤** 토스를 띄운다
 *  ③ 웹 화면조차 못 쓰면 → 예전처럼 브라우저로
 *
 * ⚠️ 국내와 해외는 다르다. 국내는 주소에 단축코드(A005930)가 그대로 들어 있어 한 번
 * 배우면 전 종목에 쓰지만, **해외는 티커가 아니라 토스 내부 상품코드**(US20100629001
 * 같은)를 쓴다. 그 코드는 티커로 만들 수 없고 토스 Open API 도 주지 않는다. 그래서
 * 해외는 종목마다 한 번씩 배운다. 국내 틀에 해외 티커를 끼워 넣으면 토스가
 * "지원하지 않는 상품"이라고 한다 — 실제로 그랬다.
 */
object TossLink {
    /** 토스 안드로이드 앱 패키지. */
    const val PKG = "viva.republica.toss"

    private const val PREFS = "toss_link"
    /** 기억한 주소에서 종목 코드가 있던 자리. */
    const val SLOT = "{CODE}"

    fun orderUrl(ticker: String): String = "https://www.tossinvest.com/stocks/${code(ticker)}/order"

    /** 국내 6자리 코드는 앞에 `A` 가 붙는다. 거래소 접미사(.KS/.KQ)는 뗀다. */
    fun code(ticker: String): String {
        val t = ticker.trim().uppercase().substringBefore('.')
        return if (t.length == 6 && t.all { it.isDigit() }) "A$t" else t
    }

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    // 국내와 해외는 **주소 만드는 법이 다르다.**
    //  · 국내: 주소 안에 단축코드(A005930)가 그대로 있다 → 한 번 배우면 전 종목에 쓴다
    //  · 해외: 티커가 아니라 토스 내부 상품코드(US20100629001 같은)를 쓴다. 티커로는
    //    만들어 낼 수 없어서(토스 API 도 안 준다) **종목마다 한 번씩** 배운다.
    //    국내에서 배운 틀에 해외 티커를 끼워 넣으면 "지원하지 않는 상품"이 뜬다.
    private fun keyTemplate(krw: Boolean) = if (krw) "tpl_kr" else "tpl_us"
    private fun keyLink(ticker: String) = "link_" + code(ticker)

    /** 그 시장에서 배워 둔 주소 틀. 없으면 null. */
    fun template(ctx: Context, ticker: String): String? =
        prefs(ctx).getString(keyTemplate(Tickers.isKrw(ticker)), null)?.takeIf { it.contains(SLOT) }

    /** 이 종목으로 기억해 둔 주소. 없으면 null. */
    fun link(ctx: Context, ticker: String): String? = prefs(ctx).getString(keyLink(ticker), null)

    /**
     * 웹 화면이 앱으로 넘어가려는 순간 잡은 주소를 기억한다.
     *
     * 종목 코드가 그 주소 안에 **그대로 있으면** 같은 시장 전체에 쓸 수 있는 틀로,
     * 없으면(해외 내부코드) **그 종목 전용**으로 저장한다.
     */
    fun learn(ctx: Context, ticker: String, deepLink: String) {
        val c = code(ticker)
        if (c.isBlank() || deepLink.isBlank()) return
        val e = prefs(ctx).edit()
        if (deepLink.contains(c)) e.putString(keyTemplate(Tickers.isKrw(ticker)), deepLink.replace(c, SLOT))
        else e.putString(keyLink(ticker), deepLink)
        e.apply()
    }

    private fun forget(ctx: Context, key: String) = prefs(ctx).edit().remove(key).apply()

    /** 배운 것을 전부 지운다 (설정에서 손으로 되돌릴 때). */
    fun forgetAll(ctx: Context) = prefs(ctx).edit().clear().apply()

    /**
     * 주문 화면을 연다.
     * @return 어떻게든 열었으면 true
     */
    fun openOrder(ctx: Context, ticker: String): Boolean {
        // ① 이 종목으로 기억해 둔 주소 (해외는 이 길로 간다)
        link(ctx, ticker)?.let {
            if (fire(ctx, it)) return true
            forget(ctx, keyLink(ticker))       // 이제 안 먹는다 — 버리고 다시 배운다
        }
        // ② 같은 시장에서 배운 틀 (국내는 이 길로 간다)
        template(ctx, ticker)?.let {
            if (fire(ctx, it.replace(SLOT, code(ticker)))) return true
            forget(ctx, keyTemplate(Tickers.isKrw(ticker)))
        }
        // ③ 우리 앱 안 웹 화면에서 열고, 앱으로 넘어가는 순간을 가로챈다
        if (start(ctx, Intent(ctx, TossOpenActivity::class.java).putExtra("ticker", ticker))) return true
        // ④ 마지막 수단 — 그냥 브라우저
        return start(ctx, Intent(Intent.ACTION_VIEW, Uri.parse(orderUrl(ticker))))
    }

    /** 앱 전용 주소(`supertoss://` · `intent://`) 를 실제로 띄운다. */
    fun fire(ctx: Context, url: String): Boolean {
        val intent = runCatching {
            if (url.startsWith("intent:")) Intent.parseUri(url, Intent.URI_INTENT_SCHEME)
            else Intent(Intent.ACTION_VIEW, Uri.parse(url))
        }.getOrNull() ?: return false
        // intent:// 에 들어 있던 브라우저 폴백은 쓰지 않는다 — 그러면 또 브라우저가 뜬다
        intent.selector = null
        return start(ctx, intent)
    }

    private fun start(ctx: Context, intent: Intent): Boolean = runCatching {
        ctx.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        true
    }.getOrDefault(false)
}
