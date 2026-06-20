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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.Gist
import com.quant.dashboard.data.Store
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.time.LocalDate

private val RANGES = listOf("6개월" to "6mo", "1년" to "1y", "2년" to "2y")

@Composable
private fun SectionHeader(title: String) {
    Text(title, color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Bold,
        modifier = Modifier.padding(top = 12.dp, bottom = 1.dp))
    Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
}

@Composable
private fun Label(text: String) = Text(text, color = TextSecondary, fontSize = 12.sp)

@Composable
fun SettingsScreen() {
    var tickers by remember { mutableStateOf(Store.loadTickers().toList()) }
    var input by remember { mutableStateOf("") }
    var seed by remember { mutableStateOf(Store.seedUsd().toInt().toString()) }
    var range by remember { mutableStateOf(Store.lookbackRange()) }
    var interval by remember { mutableStateOf(Store.candleInterval()) }
    val nameEdits = remember { mutableStateMapOf<String, String>().apply { putAll(Store.nameOverrides()) } }
    var indivVer by remember { mutableStateOf(0) }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Text("⚙️ 설정", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Bold)

        // ══════════ 분석 ══════════
        SectionHeader("📈 분석")
        Label("시드 ($)")
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(seed, { seed = it }, singleLine = true, modifier = Modifier.weight(1f))
            Button(onClick = { seed.toDoubleOrNull()?.let { if (it > 0) { Store.setSeedUsd(it); AppState.bump() } } }) { Text("저장") }
        }
        Label("분석 기간 (조회)")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            RANGES.forEach { (label, r) ->
                FilterChip(selected = range == r, onClick = { range = r; Store.setLookbackRange(r); AppState.bump() },
                    label = { Text(label, fontSize = 12.sp) })
            }
        }
        Label("봉 기준")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("일봉" to "1d", "주봉" to "1wk").forEach { (label, iv) ->
                FilterChip(selected = interval == iv, onClick = { interval = iv; Store.setCandleInterval(iv); AppState.bump() },
                    label = { Text(label, fontSize = 12.sp) })
            }
        }
        var chartM by remember { mutableStateOf(Store.chartMonths()) }
        Label("차트 조회기간")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("1개월" to 1, "2개월" to 2, "4개월" to 4, "1년" to 12).forEach { (label, m) ->
                FilterChip(selected = chartM == m, onClick = { chartM = m; Store.setChartMonths(m); AppState.bump() },
                    label = { Text(label, fontSize = 12.sp) })
            }
        }
        // 기준일 시뮬레이션
        var asofEnabled by remember { mutableStateOf(Store.asofDate() != null) }
        var asofText by remember { mutableStateOf(Store.asofDate() ?: LocalDate.now().toString()) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(checked = asofEnabled, onCheckedChange = { asofEnabled = it; if (!it) AppState.applyAsof(null) })
            Text("📅 기준일 시뮬레이션 (이 날짜까지 데이터만)", color = TextSecondary, fontSize = 12.sp)
        }
        if (asofEnabled) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(asofText, { asofText = it }, label = { Text("기준일 (YYYY-MM-DD)") },
                    singleLine = true, modifier = Modifier.weight(1f))
                Button(onClick = {
                    val ok = try { LocalDate.parse(asofText.trim()); true } catch (e: Exception) { false }
                    if (ok) AppState.applyAsof(asofText.trim())
                }) { Text("적용") }
            }
        }
        AppState.asof?.let { Text("현재 기준일: $it (헤더 ✕로 해제)", color = TextSecondary, fontSize = 11.sp) }

        // ══════════ 포트폴리오 ══════════
        SectionHeader("💼 포트폴리오")
        var eqUnit by remember { mutableStateOf(Store.equityUnit()) }
        Label("자산추이 기본 단위")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("일", "주", "월").forEach { u ->
                FilterChip(selected = eqUnit == u, onClick = { eqUnit = u; Store.setEquityUnit(u); AppState.bump() },
                    label = { Text(u, fontSize = 12.sp) })
            }
        }
        var eqM by remember { mutableStateOf(Store.equityMonths()) }
        Label("자산추이 기간")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("1개월" to 1, "2개월" to 2, "3개월" to 3, "6개월" to 6, "1년" to 12, "전체" to 600).forEach { (label, m) ->
                FilterChip(selected = eqM == m, onClick = { eqM = m; Store.setEquityMonths(m); AppState.bump() },
                    label = { Text(label, fontSize = 12.sp) })
            }
        }

        // ══════════ 데이터 (Gist) — 접힘 ══════════
        SectionHeader("☁️ 데이터 (Gist 연동)")
        var token by remember { mutableStateOf(Store.gistToken()) }
        var gistId by remember { mutableStateOf(Store.gistId()) }
        var gistMsg by remember { mutableStateOf<String?>(null) }
        var busy by remember { mutableStateOf(false) }
        var gistOpen by remember { mutableStateOf(false) }
        val scope = rememberCoroutineScope()
        val status = gistMsg ?: if (token.isNotBlank() && gistId.isNotBlank()) "연동됨 (탭하여 불러오기)" else "미설정"
        Text("${if (gistOpen) "▲" else "▼"}  $status", color = TextSecondary, fontSize = 12.sp,
            modifier = Modifier.fillMaxWidth().clickable { gistOpen = !gistOpen })
        if (gistOpen) {
            OutlinedTextField(token, { token = it }, label = { Text("GitHub Token (ghp_…)") },
                singleLine = true, modifier = Modifier.fillMaxWidth())
            OutlinedTextField(gistId, { gistId = it }, label = { Text("Gist ID") },
                singleLine = true, modifier = Modifier.fillMaxWidth())
            Button(
                enabled = !busy,
                onClick = {
                    Store.setGist(token, gistId)
                    busy = true; gistMsg = "불러오는 중…"
                    scope.launch {
                        val msg = withContext(Dispatchers.IO) {
                            try {
                                val tradesTxt = Gist.fetchFile(token.trim(), gistId.trim(), Gist.FILE_TRADES)
                                val tickersTxt = Gist.fetchFile(token.trim(), gistId.trim(), Gist.FILE_TICKERS)
                                val settingsTxt = Gist.fetchFile(token.trim(), gistId.trim(), Gist.FILE_SETTINGS)
                                if (tradesTxt == null && tickersTxt == null && settingsTxt == null) "실패: 토큰/Gist ID 확인 또는 파일 없음"
                                else {
                                    val nt = tradesTxt?.let { Store.saveTradesFromJson(it) } ?: 0
                                    val nk = tickersTxt?.let { Store.saveTickersFromJson(it) } ?: 0
                                    val ni = settingsTxt?.let { Store.saveSettingsFromJson(it) } ?: 0
                                    "✓ 불러옴: 매매 ${nt}종목 · 종목 ${nk}개 · 개별 ${ni}개"
                                }
                            } catch (e: Exception) { "오류: ${e.message}" }
                        }
                        gistMsg = msg; busy = false
                        tickers = Store.loadTickers().toList()
                        nameEdits.clear(); nameEdits.putAll(Store.nameOverrides())
                        indivVer++; AppState.bump()
                    }
                },
                modifier = Modifier.fillMaxWidth(),
            ) { Text(if (busy) "불러오는 중…" else "Gist에서 불러오기") }
            Text("데스크톱과 같은 Gist 사용. 불러오면 로컬 데이터를 덮어씁니다.",
                color = TextSecondary, fontSize = 11.sp)
        }

        // ══════════ 종목 관리 (2열 컴팩트) ══════════
        SectionHeader("📋 종목 관리")
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(input, { input = it }, placeholder = { Text("티커 (NVDA·005930)") },
                singleLine = true, modifier = Modifier.weight(1f))
            Button(onClick = {
                if (input.isNotBlank()) { Store.addTicker(input); tickers = Store.loadTickers().toList(); input = ""; AppState.bump() }
            }) { Text("추가") }
        }
        Text("최소 ${Store.MIN_TICKERS}개 · 한국=6자리(.KS/.KQ 자동) · 개별/ETF·이름 셀에서 편집",
            color = TextSecondary, fontSize = 11.sp)

        indivVer.let {
            tickers.chunked(2).forEach { pair ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.Top) {
                    pair.forEach { tk ->
                        TickerCell(
                            tk = tk, weight = 1f,
                            indiv = Store.isIndividual(tk),   // indivVer.let 안이라 토글 시 재평가→셀 재구성
                            nameValue = nameEdits[tk] ?: "",
                            onName = { nameEdits[tk] = it },
                            onToggle = { Store.setIndividual(tk, !Store.isIndividual(tk)); indivVer++; AppState.bump() },
                            onDelete = { Store.removeTicker(tk); tickers = Store.loadTickers().toList(); AppState.bump() },
                        )
                    }
                    if (pair.size == 1) Spacer(Modifier.weight(1f))
                }
            }
        }
        Button(onClick = {
            nameEdits.forEach { (tk, nm) -> Store.setNameOverride(tk, nm) }
            AppState.bump()
        }, modifier = Modifier.fillMaxWidth()) { Text("이름 저장") }
        Text("이름 칸을 비우고 저장하면 기본 표시명으로 복귀.", color = TextSecondary, fontSize = 11.sp)
    }
}

/** 종목 관리 컴팩트 셀 — 티커 + 개별/ETF 토글(상단) / 별칭 + 삭제(하단). */
@Composable
private fun RowScope.TickerCell(
    tk: String, weight: Float, indiv: Boolean, nameValue: String,
    onName: (String) -> Unit, onToggle: () -> Unit, onDelete: () -> Unit,
) {
    Column(
        Modifier.weight(weight).border(1.dp, BorderColor, RoundedCornerShape(8.dp)).padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        // 상단: 티커 + 개별/ETF 토글(큰 버튼)
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(tk, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold,
                maxLines = 1, modifier = Modifier.weight(1f))
            Box(
                Modifier.clip(RoundedCornerShape(6.dp))
                    .background(if (indiv) Profit else Color(0xFF30363D))
                    .clickable { onToggle() }
                    .padding(horizontal = 12.dp, vertical = 6.dp),
            ) { Text(if (indiv) "개별" else "ETF", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold) }
        }
        // 하단: 별칭 입력 + 삭제 (서로 떨어뜨림)
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            BasicTextField(
                value = nameValue, onValueChange = onName, singleLine = true,
                textStyle = TextStyle(color = TextPrimary, fontSize = 11.sp),
                cursorBrush = SolidColor(TextPrimary),
                modifier = Modifier.weight(1f).background(Color(0xFF0D1117), RoundedCornerShape(4.dp))
                    .padding(horizontal = 6.dp, vertical = 6.dp),
                decorationBox = { inner ->
                    Box(Modifier.fillMaxWidth()) {
                        if (nameValue.isEmpty()) Text("별칭(선택)", color = TextSecondary, fontSize = 11.sp)
                        inner()
                    }
                },
            )
            Text("삭제", color = Loss, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.clickable { onDelete() }.padding(horizontal = 6.dp, vertical = 4.dp))
        }
    }
}
