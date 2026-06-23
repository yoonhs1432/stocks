package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.Gold
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import com.quant.dashboard.ui.theme.pctColor

// 한국식: 상승=빨강 / 하락=파랑 / 0=회색
private val TableGray = Color(0xFFA4ADB8)
private fun pnColor(v: Double) = when {
    v > 0 -> Profit; v < 0 -> Loss; else -> TableGray
}
// Z 셀: ≥80 파랑(매도) / ≤20 빨강(매수) / 그 외 회색
private fun zCellColor(pct: Double) = when {
    pct >= 80 -> Loss; pct <= 20 -> Profit; else -> TableGray
}

@Composable
fun CompareScreen(vm: CompareViewModel = viewModel()) {
    val s = vm.state
    LaunchedEffect(AppState.dataVersion) { vm.sync(AppState.dataVersion) }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("종목 비교", color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold)

        when {
            s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                CircularProgressIndicator()
            }
            s.error != null -> Text("⚠️ ${s.error}", color = Loss)
            else -> {
                Column(Modifier.fillMaxWidth()) {
                    Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                        .padding(vertical = 5.dp, horizontal = 2.dp)) {
                        HCell(vm, "종목", SortKey.NAME, 2.2f, TextAlign.Start)
                        HCell(vm, "현재가", SortKey.PRICE, 2f)
                        HCell(vm, "일", SortKey.DAY, 1.4f)
                        HCell(vm, "주", SortKey.WEEK, 1.4f)
                        HCell(vm, "Z", SortKey.Z, 1f)
                        HCell(vm, "M▲", SortKey.M, 1f)
                        HCell(vm, "전고", SortKey.FROM_HIGH, 1.6f)
                    }
                    vm.sorted().forEachIndexed { idx, r ->
                        Row(Modifier.fillMaxWidth().padding(horizontal = 2.dp, vertical = 4.dp)) {
                            DotName((if (r.holding) 2 else if (r.hasHistory) 1 else 0), r.name, 2.2f)
                            Cell(Tickers.priceLabel(r.ticker, r.price), 2f, pnColor(r.day))
                            Cell(signed(r.day), 1.4f, pnColor(r.day))
                            Cell(signed(r.week), 1.4f, pnColor(r.week))
                            Cell("%.0f".format(r.zPct), 1f, zCellColor(r.zPct))
                            Cell("%.0f".format(r.mPct), 1f, Gold, fw = FontWeight.Bold)
                            Cell("%.1f%%".format(r.fromHigh), 1.6f,
                                if (r.fromHigh >= -3) Profit else if (r.fromHigh >= -50) TableGray else Loss)
                        }
                        if (idx < vm.sorted().lastIndex)
                            Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                    }
                }
                Text("● 보유 / ○ 이력 · 헤더 탭=정렬 · 색: 매수 빨강·매도 파랑",
                    color = TextSecondary, fontSize = 11.sp)

                // ── Z·M 사분면 (전 종목 현재 위치) ──
                Text("🎯 종목별 Z·M 현재 위치", color = TextPrimary,
                    fontSize = 13.sp, fontWeight = FontWeight.Bold)
                // 임계선 10/50/90 (강매수 빨강 / 중립 회 / 강매도 파랑)
                val gMid = Color(0x998B949E); val gBuy = Color(0x66DC2626); val gSell = Color(0x661D4ED8)
                val zmLines = listOf(GridLine(50.0, gMid, 1.0f), GridLine(10.0, gBuy, 0.8f), GridLine(90.0, gSell, 0.8f))
                ScatterChart(
                    points = s.rows.map { ScatterPt(it.zPct, it.mPct, it.name, pctColor(it.mPct)) },
                    xMin = -5.0, xMax = 105.0, yMin = -5.0, yMax = 105.0,
                    vLines = zmLines, hLines = zmLines,
                    xAxisLabel = "Z (가격 위치 0~100)", yAxisLabel = "M (모멘텀 0~100)", labelTopCenter = true, height = 360.dp,
                )
                Text("X=Z(가격 위치), Y=M(모멘텀) · Q1↑↑=강세 / Q3↓↓=약세 · 임계선 10/50/90",
                    color = TextSecondary, fontSize = 11.sp)

                // ── σ·β 산점도 (변동성·시장민감도) ──
                Text("📊 변동성(σ) · 시장민감도(β)", color = TextPrimary,
                    fontSize = 13.sp, fontWeight = FontWeight.Bold)
                val finite = s.rows.filter { it.beta.isFinite() && it.sigmaPct.isFinite() && it.sigmaPct > 0 }
                val betas = finite.map { it.beta }
                val sigmas = finite.map { it.sigmaPct }
                val medB = if (betas.isNotEmpty()) betas.sorted()[betas.size / 2] else 0.0
                val medS = if (sigmas.isNotEmpty()) sigmas.sorted()[sigmas.size / 2] else 1.0
                ScatterChart(
                    points = finite.map { ScatterPt(it.beta, it.sigmaPct, it.name, pctColor(it.mPct)) },
                    xMin = (betas.minOrNull() ?: -1.0) - 0.8,
                    xMax = (betas.maxOrNull() ?: 1.0) + 1.0,
                    yMin = maxOf((sigmas.minOrNull() ?: 1.0) * 0.8, 0.5),
                    yMax = (sigmas.maxOrNull() ?: 10.0) * 1.25,
                    yLog = true,
                    vLines = listOf(GridLine(medB, Color(0x88E5E7EB), 1.2f), GridLine(0.0, Color(0x99768390), 1.2f)),
                    hLines = listOf(GridLine(medS, Color(0x88E5E7EB), 1.2f)),
                    xAxisLabel = "β", yAxisLabel = "σ% (변동성)", height = 360.dp,
                )
                Text("X=β·SPY · Y=σ%(로그) · 색=모멘텀 · 점선=중앙값",
                    color = TextSecondary, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun RowScope.HCell(vm: CompareViewModel, text: String, key: SortKey, weight: Float, align: TextAlign = TextAlign.End) {
    val s = vm.state
    val on = s.sortKey == key
    val mark = if (on) (if (s.sortDesc) " ▼" else " ▲") else ""
    Text(
        text + mark, color = if (on) TextPrimary else TextSecondary, fontSize = 12.5.sp,
        fontWeight = FontWeight.SemiBold, textAlign = align,
        modifier = Modifier.weight(weight).clickable { vm.setSort(key) },
    )
}

@Composable
private fun RowScope.Cell(text: String, weight: Float, color: Color, align: TextAlign = TextAlign.End, fw: FontWeight = FontWeight.Normal) {
    Text(text, color = color, fontSize = 13.5.sp, textAlign = align, fontWeight = fw,
        fontFamily = Mono, modifier = Modifier.weight(weight))
}

/** 종목 셀 — 상태 점(보유=금채움/이력=금링) + 티커(모노). */
@Composable
private fun RowScope.DotName(state: Int, name: String, weight: Float) {
    Row(Modifier.weight(weight), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        if (state == 0) Spacer(Modifier.size(6.dp)) else Box(
            Modifier.size(6.dp).clip(RoundedCornerShape(50))
                .then(if (state == 2) Modifier.background(Gold)
                else Modifier.border(1.3.dp, Gold, RoundedCornerShape(50))),
        )
        Text(name, color = TextPrimary, fontSize = 13.5.sp, fontWeight = FontWeight.SemiBold, fontFamily = Mono)
    }
}

private fun signed(v: Double) = (if (v >= 0) "+" else "") + "%.1f%%".format(v)
