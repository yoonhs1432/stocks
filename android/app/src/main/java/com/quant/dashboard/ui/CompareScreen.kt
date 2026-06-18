package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Neutral
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import com.quant.dashboard.ui.theme.pctColor

private fun pnColor(v: Double) = when {
    v > 0 -> Profit; v < 0 -> Loss; else -> Neutral
}

@Composable
fun CompareScreen(vm: CompareViewModel = viewModel()) {
    val s = vm.state
    LaunchedEffect(Unit) { vm.loadIfEmpty() }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("🗺️ 종목 비교", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Bold)

        when {
            s.loading -> Row(Modifier.fillMaxWidth().padding(24.dp), Arrangement.Center) {
                CircularProgressIndicator()
            }
            s.error != null -> Text("⚠️ ${s.error}", color = Loss)
            else -> {
                Row(Modifier.fillMaxWidth()) {
                    HCell(vm, "종목", SortKey.NAME, 2.2f, TextAlign.Start)
                    HCell(vm, "현재가", SortKey.PRICE, 2f)
                    HCell(vm, "일", SortKey.DAY, 1.4f)
                    HCell(vm, "주", SortKey.WEEK, 1.4f)
                    HCell(vm, "Z", SortKey.Z, 1f)
                    HCell(vm, "M", SortKey.M, 1f)
                    HCell(vm, "전고", SortKey.FROM_HIGH, 1.6f)
                }
                vm.sorted().forEach { r ->
                    Row(Modifier.fillMaxWidth()) {
                        Cell(r.name, 2.2f, TextPrimary, TextAlign.Start, FontWeight.SemiBold)
                        Cell("$%,.2f".format(r.price), 2f, TextPrimary)
                        Cell(signed(r.day), 1.4f, pnColor(r.day))
                        Cell(signed(r.week), 1.4f, pnColor(r.week))
                        Cell("%.0f".format(r.zPct), 1f, pctColor(r.zPct))
                        Cell("%.0f".format(r.mPct), 1f, pctColor(r.mPct))
                        Cell("%.1f%%".format(r.fromHigh), 1.6f,
                            if (r.fromHigh >= -3) Profit else if (r.fromHigh >= -15) Neutral else Loss)
                    }
                }
                Text("헤더를 눌러 정렬 · 색: 매수=빨강 / 매도=파랑",
                    color = TextSecondary, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun RowScope.HCell(vm: CompareViewModel, text: String, key: SortKey, weight: Float, align: TextAlign = TextAlign.End) {
    val s = vm.state
    val mark = if (s.sortKey == key) (if (s.sortDesc) " ▼" else " ▲") else ""
    Text(
        text + mark, color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
        textAlign = align,
        modifier = Modifier.weight(weight).clickable { vm.setSort(key) },
    )
}

@Composable
private fun RowScope.Cell(text: String, weight: Float, color: Color, align: TextAlign = TextAlign.End, fw: FontWeight = FontWeight.Normal) {
    Text(text, color = color, fontSize = 12.sp, textAlign = align, fontWeight = fw,
        modifier = Modifier.weight(weight))
}

private fun signed(v: Double) = (if (v >= 0) "+" else "") + "%.1f%%".format(v)
