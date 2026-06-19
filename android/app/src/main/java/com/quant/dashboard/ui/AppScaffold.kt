package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.MarketRepo
import com.quant.dashboard.data.Store
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

private data class Tab(val emoji: String, val label: String)

private val TABS = listOf(
    Tab("📊", "분석"), Tab("🗺️", "비교"), Tab("💼", "포트폴리오"), Tab("⚙️", "설정"),
)

/**
 * 앱 전역 상태 — 기준일(As-of)·설정 변경 시 모든 탭이 자동으로 데이터를 다시 로드하도록
 * dataVersion을 관찰. (Streamlit의 st.rerun + 슬라이싱 동작 미러)
 */
object AppState {
    var asof by mutableStateOf(Store.asofDate())
        private set
    var dataVersion by mutableStateOf(0)
        private set

    /** 기준일 설정/해제 (null=해제) 후 전 탭 리로드 트리거. */
    fun applyAsof(d: String?) {
        Store.setAsofDate(d)
        asof = Store.asofDate()
        bump()
    }

    /** 설정(종목·시드·기간·봉기준 등) 변경 후 전 탭 리로드 트리거. */
    fun bump() { dataVersion++ }
}

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
        Column(Modifier.fillMaxSize().padding(pad)) {
            MarketHeader()
            Box(Modifier.fillMaxWidth().weight(1f)) {
                when (tab) {
                    0 -> AnalysisScreen()
                    1 -> CompareScreen()
                    2 -> PortfolioScreen()
                    else -> SettingsScreen()
                }
            }
        }
    }
}

@Composable
private fun MarketHeader() {
    var info by remember { mutableStateOf<MarketRepo.Info?>(null) }
    LaunchedEffect(Unit) {
        info = withContext(Dispatchers.IO) { MarketRepo.load() }
    }
    val i = info ?: return
    val (regimeText, regimeColor) = when (i.regime) {
        "bull" -> "🟢 강세" to Color(0xFF16A34A)
        "bear" -> "🔴 약세" to Color(0xFFDC2626)
        "correction" -> "🟠 조정" to Color(0xFFFB923C)
        "neutral" -> "⚪ 중립" to Color(0xFF6B7280)
        else -> "⚫ —" to Color(0xFF4B5563)
    }
    Row(
        Modifier.fillMaxWidth().horizontalScroll(rememberScrollState())
            .padding(horizontal = 10.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        // 기준일(As-of) 활성 배지 — 탭하면 해제
        AppState.asof?.let { d ->
            Box(
                Modifier.background(Color(0xFFCA8A04), RoundedCornerShape(8.dp))
                    .clickable { AppState.applyAsof(null) }
                    .padding(horizontal = 7.dp, vertical = 2.dp),
            ) {
                Text("📅 $d ✕", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
        val spy = i.spyRet6m?.let { "SPY(6M) ${if (it >= 0) "+" else ""}${"%.1f".format(it * 100)}%" } ?: "SPY —"
        Badge("$regimeText  $spy", regimeColor)
        i.vix?.let {
            val c = when { it < 15 -> Color(0xFF16A34A); it < 20 -> Color(0xFF6B7280); it < 30 -> Color(0xFFFB923C); else -> Color(0xFFDC2626) }
            Badge("VIX ${"%.1f".format(it)}", c)
        }
        i.us10y?.let {
            val c = when { it < 3 -> Color(0xFF16A34A); it < 4 -> Color(0xFF6B7280); it < 5 -> Color(0xFFFB923C); else -> Color(0xFFDC2626) }
            Badge("10Y ${"%.2f".format(it)}%", c)
        }
        i.usdkrw?.let { Badge("₩ ${"%,.0f".format(it)}", Color(0xFF374151)) }
    }
}

@Composable
private fun Badge(text: String, bg: Color) {
    Box(Modifier.background(bg, RoundedCornerShape(8.dp)).padding(horizontal = 7.dp, vertical = 2.dp)) {
        Text(text, color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
    }
}
