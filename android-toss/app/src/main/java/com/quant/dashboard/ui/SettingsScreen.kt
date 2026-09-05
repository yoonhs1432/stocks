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
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.BrokerCreds
import com.quant.dashboard.data.NetInfo
import com.quant.dashboard.data.Quotes
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.data.TossApi
import com.quant.dashboard.data.TossSync
import com.quant.dashboard.data.Universe
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
                // M3 OutlinedTextField 는 56dp 라 행이 커진다 → 40dp 컴팩트 입력칸
                Box(
                    Modifier.width(60.dp).height(40.dp).clip(RoundedCornerShape(10.dp)).background(SurfaceInput),
                    contentAlignment = Alignment.Center,
                ) {
                    BasicTextField(
                        rangeText, { rangeText = it.filter { c -> c.isDigit() }.take(2) },
                        singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        textStyle = TextStyle(color = TextPrimary, fontSize = 15.sp, fontFamily = Mono,
                            fontWeight = FontWeight.Bold, textAlign = TextAlign.Center),
                        cursorBrush = SolidColor(Accent),
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 6.dp),
                    )
                }
                Text("개월", color = TextSecondary, fontSize = 13.sp)
                GhostButton("적용", color = Accent) {
                    val m = rangeText.toIntOrNull()?.coerceIn(3, Store.MAX_MONTHS) ?: Store.lookbackMonths()
                    rangeText = m.toString()
                    Store.setLookbackMonths(m); Quotes.clearCache(); AppState.bump()
                }
            }
            HDivider(Modifier.padding(top = 4.dp))
        }

        // ══════════ 토스증권 ══════════
        SectionLabel("토스증권")
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            var bkKey by remember { mutableStateOf(BrokerCreds.appKey()) }
            var bkSecret by remember { mutableStateOf(BrokerCreds.appSecret()) }
            var bkMsg by remember { mutableStateOf<String?>(null) }
            var bkBusy by remember { mutableStateOf(false) }
            var bkVer by remember { mutableStateOf(0) }
            val scope = rememberCoroutineScope()
            val linked = bkVer.let { BrokerCreds.isLinked() }
            // 키 입력칸은 연결 전엔 펼쳐 두고, 연결된 뒤엔 [키 변경] 을 눌러야 나온다
            var editKeys by remember(linked) { mutableStateOf(!linked) }

            if (!BrokerCreds.available()) {
                Text("기기 보안 저장소를 열 수 없어 연동을 쓸 수 없습니다.", color = Loss, fontSize = 12.sp)
                return@Column
            }

            // ── 상태 줄 ──
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(
                    when {
                        linked -> "연결됨 · 계좌 ${BrokerCreds.maskedAccount()}"
                        BrokerCreds.hasKeys() -> "키만 저장됨"
                        else -> "미연결"
                    },
                    color = if (linked) TextPrimary else TextSecondary, fontSize = 13.sp,
                    fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f),
                )
                if (linked && !editKeys) GhostButton("키 변경") { editKeys = true }
            }

            // ── 키 입력 (미연결이거나 [키 변경]) ──
            if (editKeys) {
                OutlinedTextField(bkKey, { bkKey = it }, label = { Text("App Key") },
                    singleLine = true, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(bkSecret, { bkSecret = it }, label = { Text("App Secret") },
                    singleLine = true, visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth())
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    PrimaryButton(
                        if (bkBusy) "연결 중…" else "연결",
                        enabled = !bkBusy && bkKey.isNotBlank() && bkSecret.isNotBlank(),
                        modifier = Modifier.weight(1f),
                        onClick = {
                            BrokerCreds.saveKeys(bkKey, bkSecret)
                            bkBusy = true; bkMsg = "연결 중…"
                            scope.launch {
                                val msg = withContext(Dispatchers.IO) {
                                    try {
                                        // 계좌 목록을 받아 accountSeq 확정 (종합매매 계좌 우선)
                                        val accts = TossApi.accounts()
                                        val a = accts.firstOrNull { it.accountType == "BROKERAGE" } ?: accts.firstOrNull()
                                        if (a == null) "조회된 계좌가 없습니다"
                                        else { BrokerCreds.saveAccount(a.accountSeq, a.accountNo); null }
                                    } catch (e: Exception) {
                                        // 허용 IP 문제면 지금 IP 를 같이 알려줘야 바로 등록할 수 있다
                                        if (e is com.quant.dashboard.data.TossException && e.code == "access_denied")
                                            "허용 IP 밖입니다 — 아래 IP 를 WTS 에 등록하세요"
                                        else "실패: ${e.message}"
                                    }
                                }
                                bkMsg = msg; bkBusy = false; bkVer++; AppState.bump()
                            }
                        },
                    )
                    if (BrokerCreds.hasKeys()) GhostButton("삭제", color = Loss, enabled = !bkBusy) {
                        BrokerCreds.clear(); bkKey = ""; bkSecret = ""; bkVer++; bkMsg = null; AppState.bump()
                    }
                }
            }
            bkMsg?.let { Text(it, color = if (it.startsWith("실패") || it.contains("등록")) Loss else TextSecondary, fontSize = 12.sp) }

            // ── 현재 IP — 열면 자동 확인. 마지막으로 등록한 IP 와 다르면 빨갛게 ──
            IpRow()

            if (linked) {
                HDivider()
                PrimaryButton("체결내역 가져오기", enabled = !bkBusy, modifier = Modifier.padding(vertical = 4.dp)) {
                    bkBusy = true; bkMsg = "체결내역 가져오는 중…"
                    scope.launch {
                        val msg = withContext(Dispatchers.IO) {
                            try { TossSync.importFills().summary() } catch (e: Exception) { "실패: ${e.message}" }
                        }
                        bkMsg = msg; bkBusy = false; AppState.bump()
                    }
                }

                HDivider()
                var tick by remember { mutableStateOf(Store.tickSeconds()) }
                Label("실시간 갱신 주기")
                UnderlineSegments(
                    listOf("0" to "끔", "1" to "1초", "3" to "3초", "5" to "5초", "10" to "10초", "30" to "30초"),
                    selected = tick.toString(),
                    onSelect = { tick = it.toInt(); Store.setTickSeconds(tick); AppState.bump() },
                )

                HDivider()
                ListRow(Modifier.clickable { Quotes.clearCache(); AppState.bump() }) {
                    Text("일봉 다시 받기", color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f))
                    Text("›", color = TextSecondary, fontSize = 16.sp)
                }
            }
        }

        // ══════════ 종목 관리 ══════════
        SectionLabel("종목 관리")

        // 토스 종목 유니버스 — 이름으로 티커를 찾기 위한 캐시. 하루 1회 갱신.
        val uniScope = rememberCoroutineScope()
        var uniCount by remember { mutableStateOf(0) }
        var uniBusy by remember { mutableStateOf(false) }
        LaunchedEffect(Unit) {
            val linked = BrokerCreds.isLinked()
            if (linked) uniBusy = true
            uniCount = withContext(Dispatchers.IO) { if (linked) Universe.ensure() else Universe.count() }
            uniBusy = false
        }

        var addMsg by remember { mutableStateOf<String?>(null) }
        // 구분자가 있으면 일괄 추가. 공백은 "NVDA AAPL" 같은 티커 나열일 때만 구분자로 보고,
        // 한글이 섞여 있으면(예: "버크셔 해서웨이") 이름 검색어이므로 쪼개지 않는다.
        val bulk = remember(input) {
            val t = input.trim()
            t.any { it == ',' || it == '\n' || it == ';' } ||
                (t.any { it == ' ' } && t.none { it.code > 127 })
        }
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(input, { input = it },
                placeholder = { Text(if (uniCount > 0) "티커 또는 이름" else "티커") },
                singleLine = false, maxLines = 4, modifier = Modifier.weight(1f))
            GhostButton(if (bulk) "모두 추가" else "추가", color = Accent) {
                if (input.isNotBlank()) {
                    val (added, dup) =
                        if (bulk) Store.addTickers(input) else Store.addTickers(input.trim().replace(" ", ""))
                    tickers = Store.loadTickers().toList()
                    addMsg = when {
                        added == 0 && dup > 0 -> "이미 있습니다"
                        added > 1 -> "${added}개 추가"
                        else -> null
                    }
                    input = ""; AppState.bump()
                }
            }
        }
        addMsg?.let { Text(it, color = TextSecondary, fontSize = 11.sp) }

        // 이름 검색 결과 — 탭하면 바로 추가
        val found = remember(input, uniCount, bulk) {
            if (!bulk && input.length >= 1 && uniCount > 0) Universe.search(input) else emptyList()
        }
        found.forEach { it2 ->
            val already = it2.symbol in tickers
            Row(
                Modifier.fillMaxWidth().padding(vertical = 1.dp)
                    .clip(RoundedCornerShape(6.dp)).background(SurfaceInput)
                    .clickable(enabled = !already) {
                        Store.addTicker(it2.symbol); tickers = Store.loadTickers().toList()
                        input = ""; AppState.bump()
                    }
                    .padding(horizontal = 10.dp, vertical = 7.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(it2.symbol, color = if (already) TextMuted else TextPrimary,
                    fontSize = 13.sp, fontWeight = FontWeight.Bold)
                Text(it2.name, color = TextSecondary, fontSize = 12.sp, modifier = Modifier.weight(1f))
                Text(if (already) "추가됨" else "${it2.market} · ${it2.type}", color = TextMuted, fontSize = 10.sp)
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 6.dp)) {
            Text(
                when {
                    uniBusy -> "종목 목록 받는 중…"
                    uniCount > 0 -> "${"%,d".format(uniCount)}종목 (${Universe.cachedDate().ifEmpty { "부분" }})"
                    else -> ""
                },
                color = TextMuted, fontSize = 11.sp, modifier = Modifier.weight(1f),
            )
            if (BrokerCreds.isLinked()) GhostButton("목록 갱신") {
                uniScope.launch {
                    uniBusy = true
                    uniCount = withContext(Dispatchers.IO) { Universe.ensure(force = true) }
                    uniBusy = false
                }
            }
        }

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
                        Store.removeTicker(tk); tickers = Store.loadTickers().toList(); AppState.bump()
                    }.padding(horizontal = 4.dp, vertical = 6.dp))
            }
        }
        Spacer(Modifier.height(16.dp))
    }
    }
}

/**
 * 현재 공인 IP 한 줄 — 설정을 열면 자동으로 확인한다.
 *
 * 허용 IP 등록은 API 가 없어 앱에서 못 하므로, 할 수 있는 건 "바뀌었는지 바로 보이게"와
 * "두 탭이면 끝나게"까지다. [등록함] 을 누르면 그 IP 를 기억해 두고, 다음에 다르면 빨갛게 표시.
 */
@Composable
private fun IpRow() {
    val scope = rememberCoroutineScope()
    val clipboard = LocalClipboardManager.current
    val ctx = LocalContext.current
    var ip by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(true) }
    var registered by remember { mutableStateOf(Store.registeredIp()) }
    var copied by remember { mutableStateOf(false) }

    fun check() {
        scope.launch {
            busy = true; copied = false
            ip = withContext(Dispatchers.IO) { NetInfo.publicIp() }
            busy = false
        }
    }
    LaunchedEffect(Unit) { check() }

    val cur = ip
    val changed = cur != null && registered.isNotEmpty() && cur != registered
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("IP", color = TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
        Text(
            when { busy -> "확인 중…"; cur == null -> "확인 실패"; else -> cur },
            color = when { changed -> Loss; cur == null -> TextMuted; else -> TextPrimary },
            fontSize = 13.sp, fontFamily = Mono, maxLines = 1,
            modifier = Modifier.weight(1f).clickable { check() },
        )
        if (cur != null) {
            GhostButton(if (copied) "복사됨" else "복사") { clipboard.setText(AnnotatedString(cur)); copied = true }
            GhostButton("WTS") {
                runCatching { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.tossinvest.com"))) }
            }
            // 등록해 둔 IP 와 같으면 버튼이 사라진다 = 등록 완료 표시
            if (registered != cur) GhostButton("등록함", color = Accent) { Store.setRegisteredIp(cur); registered = cur }
        }
    }
    // 설명 문장은 두지 않는다. 문제가 있을 때(등록 IP 와 다름)만 한 줄
    if (changed) Text("등록 IP $registered 와 다름", color = Loss, fontSize = 11.sp)
}
