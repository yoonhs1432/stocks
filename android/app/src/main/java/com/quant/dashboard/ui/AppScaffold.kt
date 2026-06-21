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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.MarketRepo
import com.quant.dashboard.data.Store
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgElevated
import com.quant.dashboard.ui.theme.DividerColor
import com.quant.dashboard.ui.theme.Gold
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TabActive
import com.quant.dashboard.ui.theme.TabInactive
import com.quant.dashboard.ui.theme.TextSecondary
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
        containerColor = BgApp,
        bottomBar = { TabBar(tab) { tab = it } },
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

/** 하단 고정 탭바 — 활성 빨강 / 비활성 회색, 배경 #0A0C0F. */
@Composable
private fun TabBar(current: Int, onSelect: (Int) -> Unit) {
    Column {
        Box(Modifier.fillMaxWidth().height(1.dp).background(DividerColor))
        Row(
            Modifier.fillMaxWidth().background(BgElevated).padding(vertical = 7.dp),
        ) {
            TABS.forEachIndexed { i, t ->
                val on = current == i
                val c = if (on) TabActive else TabInactive
                Column(
                    Modifier.weight(1f).clickable { onSelect(i) },
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(2.dp),
                ) {
                    Text(t.emoji, fontSize = 17.sp)
                    Text(t.label, fontSize = 11.sp, color = c,
                        fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
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
    // 일간 등락 → 한국식 색 (상승 빨강 / 하락 파랑 / 보합 회색)
    fun pctColorOf(v: Double) = when { v > 0 -> Profit; v < 0 -> Loss; else -> TextSecondary }
    Row(
        Modifier.fillMaxWidth().background(BgApp).horizontalScroll(rememberScrollState())
            .padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        // 기준일(As-of) 활성 배지 — 탭하면 해제
        AppState.asof?.let { d ->
            Box(
                Modifier.background(Gold, RoundedCornerShape(8.dp))
                    .clickable { AppState.applyAsof(null) }
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            ) {
                Text("📅 $d ✕", color = Color(0xFF0C0E11), fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
        i.spy?.let { Chip("SPY", "${if (it >= 0) "+" else ""}${"%.1f".format(it)}%", pctColorOf(it)) }
        i.nasdaq?.let { Chip("NASDAQ", "${if (it >= 0) "+" else ""}${"%.1f".format(it)}%", pctColorOf(it)) }
        i.kospi?.let { Chip("KOSPI", "${if (it >= 0) "+" else ""}${"%.1f".format(it)}%", pctColorOf(it)) }
        i.us10y?.let { Chip("10Y", "${"%.2f".format(it)}%", Gold) }
        i.usdkrw?.let { Chip("₩", "%,.0f".format(it), TextSecondary) }
    }
}

/** 시장 지표 칩 — 라벨 + 모노 값, 컬러 틴트 배경. */
@Composable
private fun Chip(label: String, value: String, color: Color) {
    Row(
        Modifier.background(color.copy(alpha = 0.16f), RoundedCornerShape(8.dp))
            .padding(horizontal = 8.dp, vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, color = TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
        Text(value, color = color, fontSize = 10.sp, fontWeight = FontWeight.Bold, fontFamily = Mono)
    }
}
