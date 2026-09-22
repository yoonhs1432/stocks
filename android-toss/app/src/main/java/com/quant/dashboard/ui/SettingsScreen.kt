package com.quant.dashboard.ui

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.unit.Dp
import com.quant.dashboard.data.Deposits
import java.time.LocalDate
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Server
import com.quant.dashboard.data.ServerConfig
import com.quant.dashboard.data.MarketHours
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.TossSync
import com.quant.dashboard.ui.theme.Accent
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.TextMuted
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** 컴팩트 입력칸 — M3 OutlinedTextField 는 56dp 라 행이 커진다. */
@Composable
private fun NumBox(value: String, onValue: (String) -> Unit, width: Dp,
                   modifier: Modifier = Modifier, hint: String = "",
                   keyboard: KeyboardType = KeyboardType.Number) {
    Box(
        (if (width > 0.dp) modifier.width(width) else modifier)
            .height(40.dp).clip(RoundedCornerShape(10.dp)).background(SurfaceInput),
        contentAlignment = Alignment.Center,
    ) {
        if (value.isEmpty() && hint.isNotEmpty()) {
            Text(hint, color = TextMuted, fontSize = 14.sp)
        }
        BasicTextField(
            value, onValue, singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = keyboard),
            textStyle = TextStyle(color = TextPrimary, fontSize = 15.sp, fontFamily = Mono,
                fontWeight = FontWeight.Bold, textAlign = TextAlign.Center),
            cursorBrush = SolidColor(Accent),
            modifier = Modifier.fillMaxWidth().padding(horizontal = 6.dp),
        )
    }
}

/** "2026-09-09" 또는 "20260909" → "2026-09-09". 형식이 아니면 null. */
private fun normDate(s: String): String? {
    val d = s.filter { it.isDigit() }
    if (d.length != 8) return null
    val m = d.substring(4, 6).toInt(); val day = d.substring(6, 8).toInt()
    if (m !in 1..12 || day !in 1..31) return null
    return "${d.substring(0, 4)}-${d.substring(4, 6)}-${d.substring(6, 8)}"
}

private fun won(v: Double): String = "%,.0f원".format(v)

@Composable
private fun Label(text: String) = Text(text, color = TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)

@Composable
fun SettingsScreen() {
    var tickers by remember { mutableStateOf(Store.loadTickers().toList()) }
    var input by remember { mutableStateOf("") }

    Column(Modifier.fillMaxSize().background(BgApp)) {
    ScreenHeader("설정")
    Column(
        modifier = Modifier.fillMaxSize()
            .verticalScroll(rememberScrollState()).padding(horizontal = ScreenPad),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        // ══════════ 분석 ══════════
        SectionLabel("분석")
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            var rangeText by remember { mutableStateOf(Store.lookbackMonths().toString()) }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Label("분석 기간")
                Spacer(Modifier.weight(1f))
                NumBox(rangeText, { rangeText = it.filter { c -> c.isDigit() }.take(2) }, 60.dp)
                Text("개월", color = TextSecondary, fontSize = 13.sp)
                val mScope = rememberCoroutineScope()
                GhostButton("적용", color = Accent) {
                    val m = rangeText.toIntOrNull()?.coerceIn(3, Store.MAX_MONTHS) ?: Store.lookbackMonths()
                    rangeText = m.toString()
                    // 기간은 서버 설정이다 — 바꾸면 PC 가 일봉을 다시 받는다
                    mScope.launch {
                        withContext(Dispatchers.IO) { runCatching { Store.setLookbackMonths(m) } }
                        AppState.bump()
                    }
                }
            }
            HDivider(Modifier.padding(top = 4.dp))
        }

        // ══════════ 토스증권 ══════════
        SectionLabel("집 PC 연결")
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            // 이 앱은 증권사를 직접 부르지 않는다. 집 PC 가 부르고, 폰은 거기서 받는다.
            // 그래서 폰에는 **주소와 접속 암호**만 있으면 된다(앱키·시크릿은 PC 에만).
            var url by remember { mutableStateOf(ServerConfig.url()) }
            var token by remember { mutableStateOf(ServerConfig.token()) }
            var msg by remember { mutableStateOf<String?>(null) }
            var busy by remember { mutableStateOf(false) }
            var ver by remember { mutableStateOf(0) }
            val scope = rememberCoroutineScope()
            val linked = ver.let { ServerConfig.isSet() }
            var edit by remember(linked) { mutableStateOf(!linked) }

            if (!ServerConfig.available()) {
                Text("기기 보안 저장소를 열 수 없어 연결 정보를 저장할 수 없습니다.",
                    color = Loss, fontSize = 12.sp)
                return@Column
            }

            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    if (linked) "연결됨 · ${ServerConfig.url()}" else "미연결",
                    color = if (linked) TextPrimary else TextSecondary, fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f), maxLines = 2,
                )
                if (linked && !edit) GhostButton("변경") { edit = true }
            }
            if (linked && ServerConfig.accountNo().isNotBlank()) {
                Text("계좌 ${ServerConfig.maskedAccount()}", color = TextMuted,
                    fontSize = 11.sp, fontFamily = Mono)
            }

            if (edit) {
                OutlinedTextField(url, { url = it },
                    label = { Text("PC 주소") },
                    placeholder = { Text("https://hsyunpc.tailXXXX.ts.net", fontSize = 11.sp) },
                    singleLine = true, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(token, { token = it }, label = { Text("접속 암호") },
                    singleLine = true, visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth())
                Text("PC 화면(설정 → 접속)에 있는 암호입니다.", color = TextMuted, fontSize = 11.sp)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    PrimaryButton(
                        if (busy) "확인 중…" else "연결",
                        enabled = !busy && url.isNotBlank() && token.isNotBlank(),
                        modifier = Modifier.weight(1f),
                        onClick = {
                            ServerConfig.save(url, token)
                            busy = true; msg = "확인 중…"
                            scope.launch {
                                val out = withContext(Dispatchers.IO) {
                                    try {
                                        Server.health()
                                        Store.syncFromServer(force = true)
                                        MarketHours.ensure(force = true)
                                        null
                                    } catch (e: Server.HttpError) {
                                        e.message
                                    } catch (e: Exception) {
                                        "PC 에 연결되지 않습니다 (주소를 확인하세요)"
                                    }
                                }
                                msg = out ?: "연결됨"
                                busy = false; ver++; edit = out != null; AppState.bump()
                            }
                        },
                    )
                    if (linked) GhostButton("삭제", color = Loss, enabled = !busy) {
                        ServerConfig.clear(); url = ""; token = ""; ver++; msg = null; AppState.bump()
                    }
                }
            }
            msg?.let {
                Text(it, color = if (it == "연결됨") TextSecondary else Loss, fontSize = 12.sp)
            }

            if (linked) {
                HDivider()
                PrimaryButton("체결내역 가져오기", enabled = !busy, modifier = Modifier.padding(vertical = 4.dp)) {
                    busy = true; msg = "가져오는 중…"
                    scope.launch {
                        val out = withContext(Dispatchers.IO) {
                            try { "매매기록 ${TossSync.importFills()}건 저장됨" }
                            catch (e: Exception) { "실패: ${e.message}" }
                        }
                        msg = out; busy = false; AppState.bump()
                    }
                }

                HDivider()
                var tick by remember { mutableStateOf(Store.tickSeconds()) }
                Label("실시간 갱신 주기")
                UnderlineSegments(
                    listOf("0" to "끔", "1" to "1초", "3" to "3초", "5" to "5초", "10" to "10초", "30" to "30초"),
                    selected = tick.toString(),
                    onSelect = { v ->
                        tick = v.toInt()
                        scope.launch {
                            withContext(Dispatchers.IO) { Store.setTickSeconds(tick) }
                            AppState.bump()
                        }
                    },
                )

                HDivider()
                ListRow(Modifier.clickable {
                    scope.launch {
                        withContext(Dispatchers.IO) { runCatching { Server.clearCandleCache() } }
                        AppState.bump()
                    }
                }) {
                    Text("일봉 다시 받기", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f))
                    Text("›", color = TextSecondary, fontSize = 16.sp)
                }
            }
        }

        // ══════════ 원금 ══════════
        //
        // 토스 API 에 입출금 내역이 없어 앱이 원금을 알 방법이 없다. 여기 적어 둔 값이
        // 포트폴리오 탭 자산 그래프의 원금선과 수익률 기준이 된다.
        SectionLabel("원금")
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            var deps by remember { mutableStateOf(Deposits.load()) }
            var dpDate by remember { mutableStateOf(LocalDate.now().toString()) }
            var dpAmt by remember { mutableStateOf("") }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Label("입금 합계")
                Spacer(Modifier.weight(1f))
                Text(won(deps.sumOf { it.krw }), color = TextPrimary, fontSize = 15.sp,
                    fontWeight = FontWeight.Bold, fontFamily = Mono)
            }
            Row(verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                // 숫자 키패드에는 '-' 가 없는 기기가 있어 전화 키패드를 쓴다(출금 = 음수 입력)
                NumBox(dpDate, { dpDate = it.filter { c -> c.isDigit() || c == '-' }.take(10) },
                    108.dp, keyboard = KeyboardType.Phone)
                NumBox(dpAmt, { dpAmt = it.filter { c -> c.isDigit() || c == '-' }.take(12) },
                    0.dp, Modifier.weight(1f), "금액", KeyboardType.Phone)
                val dScope = rememberCoroutineScope()
                GhostButton("추가", color = Accent) {
                    val d = normDate(dpDate)
                    val v = dpAmt.toDoubleOrNull()
                    if (d != null && v != null && v != 0.0) {
                        dScope.launch {
                            withContext(Dispatchers.IO) { runCatching { Deposits.add(d, v) } }
                            deps = Deposits.load(); dpAmt = ""; AppState.bump()
                        }
                    }
                }
            }
            // 출금은 금액에 - 를 붙인다
            deps.forEachIndexed { i, d ->
                ListRow(minHeight = 38.dp) {
                    Text(d.date, color = TextSecondary, fontSize = 13.sp, fontFamily = Mono)
                    Text((if (d.krw >= 0) "+" else "") + won(d.krw),
                        color = if (d.krw >= 0) TextPrimary else Loss, fontSize = 13.sp,
                        fontWeight = FontWeight.SemiBold, fontFamily = Mono,
                        modifier = Modifier.weight(1f), textAlign = TextAlign.End)
                    val rScope = rememberCoroutineScope()
                    Text("삭제", color = TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.clickable {
                            rScope.launch {
                                withContext(Dispatchers.IO) { runCatching { Deposits.removeAt(i) } }
                                deps = Deposits.load(); AppState.bump()
                            }
                        }.padding(horizontal = 4.dp, vertical = 6.dp))
                }
            }
        }

        // ══════════ 종목 관리 ══════════
        SectionLabel("종목 관리")

        val tScope = rememberCoroutineScope()
        var addMsg by remember { mutableStateOf<String?>(null) }
        // 구분자가 있으면 일괄 추가 (증권사 앱 관심종목을 통째로 붙여넣는 용도)
        val bulk = remember(input) {
            val t = input.trim()
            t.any { it == ',' || it == '\n' || it == ';' } || t.any { it == ' ' }
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(input, { input = it },
                placeholder = { Text("티커 또는 6자리 코드") },
                singleLine = false, maxLines = 4, modifier = Modifier.weight(1f))
            GhostButton(if (bulk) "모두 추가" else "추가", color = Accent) {
                val text = input
                if (text.isNotBlank()) {
                    input = ""
                    tScope.launch {
                        val (added, dup) = withContext(Dispatchers.IO) {
                            runCatching {
                                if (bulk) Store.addTickers(text)
                                else Store.addTickers(text.trim().replace(" ", ""))
                            }.getOrDefault(0 to 0)
                        }
                        tickers = Store.loadTickers().toList()
                        addMsg = when {
                            added == 0 && dup > 0 -> "이미 있습니다"
                            added == 0 -> "추가하지 못했습니다"
                            added > 1 -> "${added}개 추가"
                            else -> null
                        }
                        AppState.bump()
                    }
                }
            }
        }
        addMsg?.let { Text(it, color = TextSecondary, fontSize = 11.sp) }
        Text("국내는 6자리 코드로 넣으세요. 이름은 PC 가 붙여 줍니다.",
            color = TextMuted, fontSize = 11.sp)

        // ── 종목 리스트 (1열) ──
        HDivider(Modifier.padding(top = 6.dp))
        tickers.forEach { tk ->
            ListRow {
                Text(tk, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold,
                    fontFamily = Mono, maxLines = 1, modifier = Modifier.width(88.dp))
                Text(Tickers.displayName(tk).takeIf { it != tk } ?: "",
                    color = TextSecondary, fontSize = 13.sp, maxLines = 1, modifier = Modifier.weight(1f))
                Text("삭제", color = TextSecondary, fontSize = 13.sp, fontWeight = FontWeight.Bold,
                    modifier = Modifier.clickable {
                        tScope.launch {
                            withContext(Dispatchers.IO) { runCatching { Store.removeTicker(tk) } }
                            tickers = Store.loadTickers().toList(); AppState.bump()
                        }
                    }.padding(horizontal = 4.dp, vertical = 6.dp))
            }
        }
        Spacer(Modifier.height(16.dp))
    }
    }
}

