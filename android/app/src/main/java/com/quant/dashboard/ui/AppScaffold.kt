package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.sp
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.TextSecondary

private data class Tab(val emoji: String, val label: String)

private val TABS = listOf(
    Tab("📊", "분석"), Tab("🗺️", "비교"), Tab("💼", "포트폴리오"), Tab("⚙️", "설정"),
)

@Composable
fun AppScaffold() {
    var tab by remember { mutableStateOf(0) }
    Scaffold(
        bottomBar = {
            NavigationBar {
                TABS.forEachIndexed { i, t ->
                    NavigationBarItem(
                        selected = tab == i,
                        onClick = { tab = i },
                        icon = { Text(t.emoji, fontSize = 18.sp) },
                        label = { Text(t.label, fontSize = 11.sp) },
                    )
                }
            }
        }
    ) { pad ->
        Box(Modifier.fillMaxSize().padding(pad)) {
            when (tab) {
                0 -> AnalysisScreen()
                1 -> CompareScreen()
                2 -> Stub("💼 포트폴리오", "매매 기록·보유·자산추이 — 다음 업데이트")
                else -> Stub("⚙️ 설정", "종목 추가/삭제·매매기록 입력 — 다음 업데이트")
            }
        }
    }
}

@Composable
private fun Stub(title: String, sub: String) {
    Box(Modifier.fillMaxSize().background(BgApp), contentAlignment = Alignment.Center) {
        Text("$title\n\n$sub", color = TextSecondary, fontSize = 14.sp)
    }
}
