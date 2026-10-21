package com.quant.dashboard.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.ServerConfig
import com.quant.dashboard.data.LivePrices
import com.quant.dashboard.data.MarketHours
import com.quant.dashboard.data.Snapshots
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.TossSync
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgElevated
import com.quant.dashboard.ui.theme.DividerColor
import com.quant.dashboard.ui.theme.TabActive
import com.quant.dashboard.ui.theme.TabInactive
import kotlin.math.cos
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// 2026-10-21 리밸런싱이 주력이 되면서 구조를 바꿨다 — 홈·바구니·더보기 3탭.
// 계좌 탭은 홈과 겹쳐 없앴다(총자산 추이 = 홈, 보유 금액·손익 = 바구니 타일, 평단·세금 = 더보기 › 손익·세금).
// 예전 포트폴리오·비교·분석·설정은 더보기 안쪽 화면(page). 목업 research/mockup_v4_I.png
private val TAB_LABELS = listOf("홈", "바구니", "더보기")

/**
 * 앱 전역 상태 — 설정이 바뀌면 모든 탭이 데이터를 다시 로드하도록 dataVersion 을 관찰.
 */
object AppState {
    var dataVersion by mutableStateOf(0)
        private set

    /** 비교·포트폴리오 탭에서 종목 클릭 시 분석 탭으로 넘길 종목(처리 후 null). */
    var pendingTicker by mutableStateOf<String?>(null)

    /** 설정(종목·기간 등) 변경 후 전 탭 리로드 트리거. */
    fun bump() { dataVersion++ }
}

@Composable
fun AppScaffold() {
    var tab by remember { mutableStateOf(0) }
    // 탭 위에 겹치는 안쪽 화면 — "pnl" · "account" · "compare"(종목 분석) · "analysis" · "settings". null 이면 탭 화면.
    var page by remember { mutableStateOf<String?>(null) }
    // 분석에서 뒤로 → 들어온 곳(종목 분석 표면 compare, 계좌에서 왔으면 null)
    var pageBack by remember { mutableStateOf<String?>(null) }

    // ── 실시간 시세 틱 ──
    // 장이 열려 있을 때만, 설정한 주기로 `/prices` 를 한 번 호출해 전 종목 현재가를 갱신한다.
    // 일봉·분석은 무거워서 기존 5분 캐시 그대로 두고, 화면의 현재가·등락률만 이 값으로 덮어쓴다.
    // 종목 유니버스(이름 검색·국내 표시명)는 앱 시작 시 한 번 메모리에 올려 둔다 —
    // 표시명은 화면 그리는 중에 불리므로 그때 파일을 읽으면 안 된다.
    // 서버에서 설정·종목 목록·입금·매매기록을 받아 둔다. 화면이 그리는 중에 읽는 값이라
    // 여기서 미리 채워 놓아야 한다(그리는 중에 네트워크를 타면 앱이 죽는다).
    // 한 번 실패하면 앱을 껐다 켜기 전까지 종목 목록·기간·매매기록이 비어 있었다.
    // 받을 때까지 8초마다 다시 묻는다 — 집 PC 와의 끊김은 대개 저절로 돌아온다.
    // 탭들은 **이게 끝나야** 시세를 부른다(`Store.synced()`). 종목 목록을 모르는 채로 먼저
    // 부르면 기본 목록으로 한 번, 목록이 와서 또 한 번 — 20~30초짜리 조회가 두 번이 된다.
    LaunchedEffect(Unit) {
        while (true) {
            val changed = withContext(Dispatchers.IO) { Store.syncFromServer(force = true) }
            if (Store.synced()) {
                // 첫 동기화는 목록을 처음 채우는 것이라 거의 항상 '달라졌다'로 나온다.
                // 그래도 여기서 한 번은 깨워야 탭들이 첫 로드를 시작한다.
                AppState.bump()
                // 아직 안 연 탭의 값도 **지금** 받아 둔다 — 포트폴리오를 처음 누를 때 기다리지 않게.
                // 계좌가 오면 보유 종목이 틱 대상에도 바로 들어간다. 실패해도 탭이 알아서 다시 묻는다.
                launch(Dispatchers.IO) { runCatching { TossSync.account() } }
                launch(Dispatchers.IO) { runCatching { Snapshots.ensure() } }
                break
            }
            if (changed) AppState.bump()
            kotlinx.coroutines.delay(8_000)
        }
    }

    LaunchedEffect(AppState.dataVersion) {
        while (true) {
            val sec = Store.tickSeconds()
            if (sec <= 0 || !ServerConfig.isSet()) {
                LivePrices.clear()
                LivePrices.noteIdle(if (sec <= 0) "실시간 갱신 꺼짐" else "PC 미연결")
                kotlinx.coroutines.delay(30_000)
                continue
            }
            withContext(Dispatchers.IO) {
                MarketHours.ensure()
                if (MarketHours.anyOpen()) {
                    val held = TossSync.cachedAccount()?.holdings?.items?.map { it.symbol } ?: emptyList()
                    LivePrices.tick((Store.loadTickers() + Tickers.BASE + held).distinct())
                } else {
                    // 왜 안 도는지 화면에 남긴다 — 조용히 멈추면 고장과 구분이 안 된다
                    LivePrices.noteIdle(MarketHours.label().ifBlank { "장 마감" })
                }
            }
            kotlinx.coroutines.delay(sec * 1000L)
        }
    }
    Scaffold(
        containerColor = BgApp,
        bottomBar = { TabBar(tab) { tab = it; page = null; pageBack = null } },
    ) { pad ->
        Column(Modifier.fillMaxSize().padding(pad)) {
            Box(Modifier.fillMaxWidth().weight(1f)) {
                val openAnalysis: (String) -> Unit = {
                    AppState.pendingTicker = it
                    pageBack = page
                    page = "analysis"
                }
                // 탭을 오가도 **보던 자리**가 남게 한다. 그냥 갈아 끼우면 스크롤이 매번 맨 위로
                // 돌아가서, 분석을 보고 돌아오면 비교 표에서 보던 종목을 다시 찾아 내려가야 했다.
                val keep = rememberSaveableStateHolder()
                keep.SaveableStateProvider(page ?: "tab$tab") {
                    when (page) {
                        "compare" -> {
                            androidx.activity.compose.BackHandler { page = null }
                            CompareScreen(onOpenAnalysis = openAnalysis)
                        }
                        "analysis" -> AnalysisScreen(onBack = { page = pageBack; pageBack = null })
                        "settings" -> {
                            androidx.activity.compose.BackHandler { page = null }
                            SettingsScreen()
                        }
                        "pnl" -> PnlScreen(onBack = { page = null })
                        "account" -> {
                            androidx.activity.compose.BackHandler { page = null }
                            PortfolioScreen(onOpenAnalysis = openAnalysis)
                        }
                        else -> when (tab) {
                            0 -> HomeScreen()
                            1 -> BasketScreen()
                            else -> MoreScreen(onOpen = { page = it })
                        }
                    }
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
            TAB_LABELS.forEachIndexed { i, label ->
                val on = current == i
                val c = if (on) TabActive else TabInactive
                Column(
                    Modifier.weight(1f).clickable { onSelect(i) },
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(3.dp),
                ) {
                    TabIcon(i, c)
                    Text(label, fontSize = 11.sp, color = c,
                        fontWeight = if (on) FontWeight.Bold else FontWeight.Normal)
                }
            }
        }
    }
}

/** 단색 라인 탭 아이콘 (0=집(홈) / 1=과녁(바구니) / 그 밖=점 셋(더보기)). */
@Composable
private fun TabIcon(index: Int, color: Color) {
    Canvas(Modifier.size(22.dp)) {
        val w = size.width; val h = size.height
        when (index) {
            0 -> {  // 집 (홈)
                val sw = w * 0.085f
                val p = androidx.compose.ui.graphics.Path().apply {
                    moveTo(w * 0.16f, h * 0.48f); lineTo(w * 0.5f, h * 0.18f); lineTo(w * 0.84f, h * 0.48f)
                }
                drawPath(p, color, style = Stroke(sw))
                drawRect(color, topLeft = Offset(w * 0.26f, h * 0.46f), size = Size(w * 0.48f, h * 0.38f), style = Stroke(sw))
            }
            1 -> {  // 과녁 (바구니)
                val sw = w * 0.085f
                drawCircle(color, w * 0.34f, Offset(w / 2, h / 2), style = Stroke(sw))
                drawCircle(color, w * 0.18f, Offset(w / 2, h / 2), style = Stroke(sw))
                drawCircle(color, w * 0.06f, Offset(w / 2, h / 2))
            }
            else -> {  // 점 셋 (더보기)
                for (px in listOf(0.24f, 0.5f, 0.76f)) drawCircle(color, w * 0.075f, Offset(w * px, h * 0.5f))
            }
        }
    }
}

