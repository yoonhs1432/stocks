package com.quant.dashboard.ui

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material3.Button
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
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.BgCard
import com.quant.dashboard.ui.theme.BorderColor
import com.quant.dashboard.ui.theme.ChipOn
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.Mono
import com.quant.dashboard.ui.theme.Profit
import com.quant.dashboard.ui.theme.SurfaceInput
import com.quant.dashboard.ui.theme.TextMuted
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
private fun SectionHeader(title: String) {
    Text(title, color = Profit, fontSize = 14.sp, fontWeight = FontWeight.Bold,
        modifier = Modifier.padding(top = 16.dp, bottom = 6.dp))
}

/** 섹션 카드 — 컨트롤 묶음을 담는 컨테이너. */
@Composable
private fun SettingsCard(content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(BgCard)
            .border(1.dp, BorderColor, RoundedCornerShape(14.dp)).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) { content() }
}

@Composable
private fun Label(text: String) = Text(text, color = TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)

/** 커스텀 세그먼트 칩 (활성 = ChipOn). */
@Composable
private fun Seg(label: String, selected: Boolean, onClick: () -> Unit) {
    Box(
        Modifier.clip(RoundedCornerShape(8.dp)).background(if (selected) ChipOn else SurfaceInput)
            .clickable { onClick() }.padding(horizontal = 14.dp, vertical = 8.dp),
    ) {
        Text(label, color = if (selected) TextPrimary else TextMuted, fontSize = 12.sp,
            fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal)
    }
}

/** 작은 보조 버튼 (복사·열기·키 변경 등). */
@Composable
private fun SmallBtn(label: String, onClick: () -> Unit) {
    Box(
        Modifier.clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
            .clickable { onClick() }.padding(horizontal = 12.dp, vertical = 7.dp),
    ) { Text(label, color = TextPrimary, fontSize = 12.sp, fontWeight = FontWeight.Bold) }
}

@Composable
private fun Divider() = Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))

@Composable
fun SettingsScreen() {
    var tickers by remember { mutableStateOf(Store.loadTickers().toList()) }
    var input by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("설정", color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold)

        // ══════════ 분석 ══════════
        SectionHeader("분석")
        SettingsCard {
            var rangeText by remember { mutableStateOf(Store.lookbackMonths().toString()) }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Label("분석 기간")
                Spacer(Modifier.weight(1f))
                OutlinedTextField(
                    rangeText, { rangeText = it.filter { c -> c.isDigit() }.take(2) },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.width(84.dp),
                )
                Text("개월", color = TextSecondary, fontSize = 13.sp)
                Button(onClick = {
                    val m = rangeText.toIntOrNull()?.coerceIn(3, Store.MAX_MONTHS) ?: Store.lookbackMonths()
                    rangeText = m.toString()
                    Store.setLookbackMonths(m); Quotes.clearCache(); AppState.bump()
                }) { Text("적용") }
            }
            Text("3 ~ ${Store.MAX_MONTHS}개월", color = TextMuted, fontSize = 11.sp)
        }

        // ══════════ 토스증권 ══════════
        SectionHeader("토스증권 (조회 전용)")
        SettingsCard {
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
                return@SettingsCard
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
                if (linked && !editKeys) SmallBtn("키 변경") { editKeys = true }
            }

            // ── 키 입력 (미연결이거나 [키 변경]) ──
            if (editKeys) {
                OutlinedTextField(bkKey, { bkKey = it }, label = { Text("App Key") },
                    singleLine = true, modifier = Modifier.fillMaxWidth())
                OutlinedTextField(bkSecret, { bkSecret = it }, label = { Text("App Secret") },
                    singleLine = true, visualTransformation = PasswordVisualTransformation(),
                    modifier = Modifier.fillMaxWidth())
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    Button(
                        enabled = !bkBusy && bkKey.isNotBlank() && bkSecret.isNotBlank(),
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
                        modifier = Modifier.weight(1f),
                    ) { Text(if (bkBusy) "연결 중…" else "연결") }
                    if (BrokerCreds.hasKeys()) Button(
                        enabled = !bkBusy,
                        onClick = { BrokerCreds.clear(); bkKey = ""; bkSecret = ""; bkVer++; bkMsg = null; AppState.bump() },
                        modifier = Modifier.weight(1f),
                    ) { Text("삭제") }
                }
            }
            bkMsg?.let { Text(it, color = if (it.startsWith("실패") || it.contains("등록")) Loss else TextSecondary, fontSize = 12.sp) }

            // ── 현재 IP — 열면 자동 확인. 마지막으로 등록한 IP 와 다르면 빨갛게 ──
            IpRow()

            if (linked) {
                Divider()
                Button(
                    enabled = !bkBusy,
                    onClick = {
                        bkBusy = true; bkMsg = "체결내역 가져오는 중…"
                        scope.launch {
                            val msg = withContext(Dispatchers.IO) {
                                try { TossSync.importFills().summary() } catch (e: Exception) { "실패: ${e.message}" }
                            }
                            bkMsg = msg; bkBusy = false; AppState.bump()
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) { Text("체결내역 가져오기") }

                Divider()
                var tick by remember { mutableStateOf(Store.tickSeconds()) }
                Label("실시간 갱신 주기")
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf("끔" to 0, "1초" to 1, "3초" to 3, "5초" to 5, "10초" to 10, "30초" to 30).forEach { (lab, v) ->
                        Seg(lab, tick == v) { tick = v; Store.setTickSeconds(v); AppState.bump() }
                    }
                }

                Divider()
                Box(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                        .clickable { Quotes.clearCache(); AppState.bump() }
                        .padding(vertical = 9.dp),
                    contentAlignment = Alignment.Center,
                ) { Text("일봉 다시 받기", color = TextSecondary, fontSize = 12.sp) }
            }
        }

        // ══════════ 종목 관리 ══════════
        SectionHeader("종목 관리")

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
            Button(onClick = {
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
            }) { Text(if (bulk) "모두 추가" else "추가") }
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
            if (BrokerCreds.isLinked()) SmallBtn("목록 갱신") {
                uniScope.launch {
                    uniBusy = true
                    uniCount = withContext(Dispatchers.IO) { Universe.ensure(force = true) }
                    uniBusy = false
                }
            }
        }

        // ── 종목 리스트 (1열) ──
        Column(
            Modifier.fillMaxWidth().padding(top = 6.dp).clip(RoundedCornerShape(12.dp))
                .background(BgCard).border(1.dp, BorderColor, RoundedCornerShape(12.dp)),
        ) {
            tickers.forEachIndexed { i, tk ->
                Row(
                    Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 9.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(tk, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold,
                        fontFamily = Mono, maxLines = 1, modifier = Modifier.width(88.dp))
                    Text(Tickers.displayName(tk).takeIf { it != tk } ?: "",
                        color = TextSecondary, fontSize = 13.sp, maxLines = 1, modifier = Modifier.weight(1f))
                    Text("삭제", color = Loss, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                        modifier = Modifier.clickable {
                            Store.removeTicker(tk); tickers = Store.loadTickers().toList(); AppState.bump()
                        }.padding(horizontal = 4.dp, vertical = 2.dp))
                }
                if (i < tickers.lastIndex) Divider()
            }
        }
        Spacer(Modifier.height(8.dp))
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
            SmallBtn(if (copied) "복사됨" else "복사") { clipboard.setText(AnnotatedString(cur)); copied = true }
            SmallBtn("WTS") {
                runCatching { ctx.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.tossinvest.com"))) }
            }
        }
    }
    if (cur != null) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(
                when {
                    registered.isEmpty() -> "WTS 에 등록했으면 [등록함]"
                    changed -> "등록한 IP($registered)와 다름 — WTS 에 다시 등록"
                    else -> "등록한 IP 와 같음"
                },
                color = if (changed) Loss else TextMuted, fontSize = 11.sp,
                modifier = Modifier.weight(1f),
            )
            if (registered != cur) SmallBtn("등록함") { Store.setRegisteredIp(cur); registered = cur }
        }
    }
}
