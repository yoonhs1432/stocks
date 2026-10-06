package com.quant.dashboard

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.ServerConfig
import com.quant.dashboard.data.Store
import com.quant.dashboard.ui.AppScaffold
import com.quant.dashboard.ui.theme.QuantTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Store.init(applicationContext)
        ServerConfig.init(applicationContext)
        // 지난번 비교 표를 꺼내 둔다 — 첫 화면이 비어 있지 않게. 작은 파일 하나라 즉시 끝난다.
        OverviewRepo.restore()
        setContent {
            QuantTheme {
                AppScaffold()
            }
        }
    }
}
