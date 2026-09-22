package com.quant.dashboard

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.os.Bundle
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import com.quant.dashboard.data.TossLink

/**
 * 토스 주문 화면으로 넘어가기 위한 **배우는 징검다리**.
 *
 * 토스 앱은 `tossinvest.com` 을 자기 주소로 등록해 두지 않아서, 주소를 던지면 브라우저가
 * 열리고 거기서 [앱 열기]를 또 눌러야 한다. 여기서 두 가지를 주워 둔다.
 *
 *  ① **앱으로 넘어가는 주소** — [앱 열기] 가 쏘는 `supertoss://…`. 국내는 이 주소에
 *     단축코드가 그대로 들어 있어 한 번 배우면 전 종목에 쓴다.
 *  ② **정식 상품코드** — 해외는 티커(GDXU)가 아니라 내부코드(US…)를 쓴다. 토스가
 *     주소를 정식 코드로 바꿔 주는 순간 그 코드를 줍는다. ①에서 배운 모양에 이 코드를
 *     끼우면 **[앱 열기]를 누르지 않아도** 앱으로 바로 넘어간다.
 *
 * 로그인도, 주문도 여기서 하지 않는다.
 */
class TossOpenActivity : ComponentActivity() {

    private var ticker = ""
    private var jumped = false

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ticker = intent.getStringExtra("ticker").orEmpty()
        if (ticker.isBlank()) { finish(); return }

        val web = WebView(this)
        web.settings.javaScriptEnabled = true        // 없으면 [앱 열기] 가 동작하지 않는다
        web.settings.domStorageEnabled = true
        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, req: WebResourceRequest): Boolean {
                val u = req.url
                if (u.scheme == "http" || u.scheme == "https") {
                    note(u.toString())
                    return false
                }
                jump(u.toString(), remember = true)
                return true
            }

            override fun onPageStarted(view: WebView, url: String, favicon: Bitmap?) = note(url)

            // 토스 웹은 화면 안에서 주소만 바꾸는 경우가 많다 — 그때는 이쪽만 불린다
            override fun doUpdateVisitedHistory(view: WebView, url: String, isReload: Boolean) = note(url)
        }
        setContentView(web)
        web.loadUrl(TossLink.orderUrl(ticker))
    }

    /** 주소에서 `/stocks/<코드>` 를 본다. 내가 넣은 것과 다르면 그게 정식 코드다. */
    private fun note(url: String) {
        if (jumped) return
        val found = Regex("/stocks/([A-Za-z0-9]+)").find(url)?.groupValues?.get(1) ?: return
        if (found.equals(TossLink.code(ticker), ignoreCase = true)) return
        TossLink.learnCode(this, ticker, found)
        // 앱으로 가는 모양을 이미 안다면 여기서 바로 넘어간다 ([앱 열기] 를 안 눌러도 된다)
        val shape = TossLink.anyTemplate(this) ?: return
        jump(shape.replace(TossLink.SLOT, found), remember = false)
    }

    private fun jump(deepLink: String, remember: Boolean) {
        if (jumped) return
        jumped = true
        if (remember) TossLink.learn(this, ticker, deepLink)
        TossLink.fire(this, deepLink)
        finish()
    }
}
