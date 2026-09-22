package com.quant.dashboard

import android.annotation.SuppressLint
import android.os.Bundle
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import com.quant.dashboard.data.TossLink

/**
 * 토스 주문 화면으로 넘어가기 위한 **한 번뿐인 징검다리**.
 *
 * 토스 앱은 `tossinvest.com` 을 자기 주소로 등록해 두지 않아서, 주소를 던지면 브라우저가
 * 열리고 거기서 [앱 열기]를 또 눌러야 한다. 그 버튼이 쏘는 앱 전용 주소를 여기서
 * **가로채 기억해 두면**, 다음부터는 이 화면조차 거치지 않고 토스로 바로 간다
 * (`TossLink.openOrder`).
 *
 * 하는 일은 그게 전부다. 로그인도, 주문도 여기서 하지 않는다.
 */
class TossOpenActivity : ComponentActivity() {

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val ticker = intent.getStringExtra("ticker").orEmpty()
        if (ticker.isBlank()) { finish(); return }

        val web = WebView(this)
        web.settings.javaScriptEnabled = true        // 없으면 [앱 열기] 가 동작하지 않는다
        web.settings.domStorageEnabled = true
        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, req: WebResourceRequest): Boolean {
                val u = req.url
                // 웹 주소는 그대로 웹뷰에서 연다
                if (u.scheme == "http" || u.scheme == "https") return false
                // 앱으로 넘어가려는 순간 — 이 주소를 기억하고 토스를 띄운다
                val link = u.toString()
                TossLink.learn(this@TossOpenActivity, ticker, link)
                TossLink.fire(this@TossOpenActivity, link)
                finish()
                return true
            }
        }
        setContentView(web)
        web.loadUrl(TossLink.orderUrl(ticker))
    }
}
