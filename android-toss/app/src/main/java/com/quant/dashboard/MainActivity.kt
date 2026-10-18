package com.quant.dashboard

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.quant.dashboard.data.AppUpdate
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import com.quant.dashboard.data.OverviewRepo
import com.quant.dashboard.data.Snapshots
import com.quant.dashboard.data.TossSync
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
        // 계좌·자산 추이도 지난 값을 먼저 — 포트폴리오가 "불러오는 중…" 으로 시작하지 않게.
        TossSync.restore()
        Snapshots.restore()
        setContent {
            QuantTheme {
                AppScaffold()
            }
        }
        // 스스로 업데이트 — 알림을 띄우려면 13+ 에서 허락이 필요하다
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            registerForActivityResult(ActivityResultContracts.RequestPermission()) {}
                .launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        AppUpdate.schedule(applicationContext)
        // 켤 때도 한 번 — 한 시간 주기를 기다리지 않게
        lifecycleScope.launch(Dispatchers.IO) { runCatching { AppUpdate.run(applicationContext, user = false) } }
    }

    override fun onResume() {
        super.onResume()
        AppUpdate.foreground = true
    }

    override fun onPause() {
        AppUpdate.foreground = false
        super.onPause()
    }
}
