package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.Gist
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import java.time.LocalDate

private val RANGES = listOf("6개월" to "6mo", "1년" to "1y", "2년" to "2y")

@Composable
fun SettingsScreen() {
    var tickers by remember { mutableStateOf(Store.loadTickers().toList()) }
    var input by remember { mutableStateOf("") }
    var seed by remember { mutableStateOf(Store.seedUsd().toInt().toString()) }
    var range by remember { mutableStateOf(Store.lookbackRange()) }
    var interval by remember { mutableStateOf(Store.candleInterval()) }
    // 종목별 이름 override 편집 버퍼
    val nameEdits = remember { mutableStateMapOf<String, String>().apply { putAll(Store.nameOverrides()) } }
    // 개별/ETF 토글 갱신용 카운터
    var indivVer by remember { mutableStateOf(0) }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("⚙️ 설정", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Bold)

        // ── 시드 ($) ──
        Text("시드 ($)", color = TextSecondary, fontSize = 12.sp)
        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = seed, onValueChange = { seed = it },
                singleLine = true, modifier = Modifier.weight(1f),
            )
            Button(onClick = {
                seed.toDoubleOrNull()?.let { if (it > 0) { Store.setSeedUsd(it); AppState.bump() } }
            }) { Text("저장") }
        }

        // ── 분석 기간 ──
        Text("분석 시작일 (조회 기간)", color = TextSecondary, fontSize = 12.sp)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            RANGES.forEach { (label, r) ->
                FilterChip(selected = range == r, onClick = {
                    range = r; Store.setLookbackRange(r); AppState.bump()
                }, label = { Text(label, fontSize = 12.sp) })
            }
        }

        // ── 봉 기준 (일봉/주봉) ──
        Text("봉 기준", color = TextSecondary, fontSize = 12.sp)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("일봉" to "1d", "주봉" to "1wk").forEach { (label, iv) ->
                FilterChip(selected = interval == iv, onClick = {
                    interval = iv; Store.setCandleInterval(iv); AppState.bump()
                }, label = { Text(label, fontSize = 12.sp) })
            }
        }

        // ── 차트 조회기간 ──
        var chartM by remember { mutableStateOf(Store.chartMonths()) }
        Text("차트 조회기간", color = TextSecondary, fontSize = 12.sp)
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("1개월" to 1, "2개월" to 2, "4개월" to 4, "1년" to 12).forEach { (label, m) ->
                FilterChip(selected = chartM == m, onClick = {
                    chartM = m; Store.setChartMonths(m); AppState.bump()
                }, label = { Text(label, fontSize = 12.sp) })
            }
        }

        // ── 기준일(As-of) 시뮬레이션 ──
        Text("📅 기준일 시뮬레이션", color = TextPrimary, fontSize = 15.sp,
            fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 6.dp))
        var asofEnabled by remember { mutableStateOf(Store.asofDate() != null) }
        var asofText by remember { mutableStateOf(Store.asofDate() ?: LocalDate.now().toString()) }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(checked = asofEnabled, onCheckedChange = {
                asofEnabled = it
                if (!it) AppState.applyAsof(null)
            })
            Text("과거 시점 재현 (이 날짜까지 데이터만)", color = TextSecondary, fontSize = 12.sp)
        }
        if (asofEnabled) {
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(asofText, { asofText = it }, label = { Text("기준일 (YYYY-MM-DD)") },
                    singleLine = true, modifier = Modifier.weight(1f))
                Button(onClick = {
                    val ok = try { LocalDate.parse(asofText.trim()); true } catch (e: Exception) { false }
                    if (ok) AppState.applyAsof(asofText.trim())
                }) { Text("적용") }
            }
        }
        AppState.asof?.let { Text("현재 기준일: $it (헤더 배지 ✕ 또는 체크 해제로 해제)", color = TextSecondary, fontSize = 11.sp) }

        // ── Gist 연동 (기존 데이터 불러오기) ──
        Text("☁️ Gist 불러오기 (매매기록·종목)", color = TextPrimary, fontSize = 15.sp,
            fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 6.dp))
        var token by remember { mutableStateOf(Store.gistToken()) }
        var gistId by remember { mutableStateOf(Store.gistId()) }
        var gistMsg by remember { mutableStateOf<String?>(null) }
        var busy by remember { mutableStateOf(false) }
        val scope = rememberCoroutineScope()
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
                                "완료: 매매 ${nt}종목 · 종목 ${nk}개 · 개별 ${ni}개 불러옴"
                            }
                        } catch (e: Exception) {
                            "오류: ${e.message}"
                        }
                    }
                    gistMsg = msg
                    busy = false
                    tickers = Store.loadTickers().toList()
                    nameEdits.clear(); nameEdits.putAll(Store.nameOverrides())
                    indivVer++   // 개별/ETF 토글 즉시 갱신
                    AppState.bump()
                }
            },
            modifier = Modifier.fillMaxWidth(),
        ) { Text(if (busy) "불러오는 중…" else "Gist에서 불러오기") }
        gistMsg?.let { Text(it, color = TextSecondary, fontSize = 12.sp) }
        Text("데스크톱과 같은 Gist 사용 (quant_trade_history.json 등). 불러오면 로컬 데이터를 덮어씁니다.",
            color = TextSecondary, fontSize = 11.sp)

        // ── 종목 관리 ──
        Text("종목 관리", color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(top = 6.dp))

        Row(verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = input, onValueChange = { input = it },
                placeholder = { Text("티커 (예: NVDA, 005930)") },
                singleLine = true, modifier = Modifier.weight(1f),
            )
            Button(onClick = {
                if (input.isNotBlank()) {
                    Store.addTicker(input)
                    tickers = Store.loadTickers().toList()
                    input = ""
                    AppState.bump()
                }
            }) { Text("추가") }
        }
        Text("최소 ${Store.MIN_TICKERS}개 유지 · 한국=6자리 코드(.KS/.KQ 자동) · 이름/개별 편집은 행별",
            color = TextSecondary, fontSize = 11.sp)

        // indivVer를 읽어 토글 시 재구성 보장
        indivVer.let {
            tickers.forEach { tk ->
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    OutlinedTextField(
                        value = nameEdits[tk] ?: "",
                        onValueChange = { nameEdits[tk] = it },
                        placeholder = { Text(Tickers.displayName(tk), fontSize = 12.sp) },
                        label = { Text(tk, fontSize = 10.sp) },
                        singleLine = true, modifier = Modifier.weight(1.6f),
                    )
                    val indiv = Store.isIndividual(tk)
                    FilterChip(selected = indiv, onClick = {
                        Store.setIndividual(tk, !indiv); indivVer++; AppState.bump()
                    }, label = { Text(if (indiv) "개별" else "ETF", fontSize = 11.sp) })
                    TextButton(onClick = {
                        Store.removeTicker(tk)
                        tickers = Store.loadTickers().toList()
                        AppState.bump()
                    }) { Text("삭제", color = Loss, fontSize = 12.sp) }
                }
            }
        }
        Button(onClick = {
            nameEdits.forEach { (tk, nm) -> Store.setNameOverride(tk, nm) }
            AppState.bump()
        }, modifier = Modifier.fillMaxWidth()) { Text("이름 저장") }
        Text("이름 칸을 비우고 저장하면 기본 표시명으로 되돌아갑니다.",
            color = TextSecondary, fontSize = 11.sp)
    }
}
