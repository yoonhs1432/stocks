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
 */
object TossLink {
    /** 토스 안드로이드 앱 패키지. */
    const val PKG = "viva.republica.toss"

    private const val PREFS = "toss_link"
    private const val K_TEMPLATE = "deeplink_template"
    /** 기억한 주소에서 종목 코드가 있던 자리. */
    const val SLOT = "{CODE}"

    fun orderUrl(ticker: String): String = "https://www.tossinvest.com/stocks/${code(ticker)}/order"

    /** 국내 6자리 코드는 앞에 `A` 가 붙는다. 거래소 접미사(.KS/.KQ)는 뗀다. */
    fun code(ticker: String): String {
        val t = ticker.trim().uppercase().substringBefore('.')
        return if (t.length == 6 && t.all { it.isDigit() }) "A$t" else t
    }

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** 기억해 둔 앱 주소 틀 (`supertoss://…{CODE}…`). 없으면 null. */
    fun template(ctx: Context): String? =
        prefs(ctx).getString(K_TEMPLATE, null)?.takeIf { it.contains(SLOT) }

    /**
     * 웹 화면이 앱으로 넘어가려는 순간 잡은 주소를 기억한다.
     *
     * 종목 코드가 그 주소 안에 **그대로 들어 있을 때만** 기억한다. 안 들어 있으면
     * (예: 토스 내부 상품코드) 다른 종목에 갖다 쓸 수 없어 오히려 엉뚱한 화면이 뜬다.
     */
    fun learn(ctx: Context, ticker: String, deepLink: String) {
        val c = code(ticker)
        if (c.isBlank() || !deepLink.contains(c)) return
        prefs(ctx).edit().putString(K_TEMPLATE, deepLink.replace(c, SLOT)).apply()
    }

    fun forget(ctx: Context) = prefs(ctx).edit().remove(K_TEMPLATE).apply()

    /**
     * 주문 화면을 연다.
     * @return 어떻게든 열었으면 true
     */
    fun openOrder(ctx: Context, ticker: String): Boolean {
        // ① 배워 둔 앱 주소로 직행
        template(ctx)?.let { t ->
            if (fire(ctx, t.replace(SLOT, code(ticker)))) return true
            forget(ctx)      // 이제 안 먹는 주소다 — 버리고 다시 배운다
        }
        // ② 우리 앱 안 웹 화면에서 열고, 앱으로 넘어가는 순간을 가로챈다
        if (start(ctx, Intent(ctx, TossOpenActivity::class.java).putExtra("ticker", ticker))) return true
        // ③ 마지막 수단 — 그냥 브라우저
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
