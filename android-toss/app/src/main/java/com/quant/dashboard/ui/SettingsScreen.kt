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
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.PasswordVisualTransformation
import com.quant.dashboard.data.BrokerCreds
import com.quant.dashboard.data.TossApi
import com.quant.dashboard.data.NetInfo
import com.quant.dashboard.data.TossSync
import com.quant.dashboard.data.Quotes
import com.quant.dashboard.data.Store
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
import kotlin.math.roundToInt

/**
 * 기간 설정 공통 슬라이더 — 1개월 단위, 최대 Store.MAX_MONTHS(2년).
 * 12개월 이상은 "N년 M개월"로 읽기 쉽게 표기한다.
 */
@Composable
private fun MonthSlider(label: String, months: Int, hint: String? = null, onChange: (Int) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Label(label)
        Spacer(Modifier.weight(1f))
        Text(monthsLabel(months), color = TextPrimary, fontSize = 12.sp,
            fontWeight = FontWeight.Bold, fontFamily = Mono)
    }
    Slider(
        value = months.toFloat(),
        onValueChange = { onChange(it.roundToInt().coerceIn(1, Store.MAX_MONTHS)) },
        valueRange = 1f..Store.MAX_MONTHS.toFloat(),
        steps = Store.MAX_MONTHS - 2,   // 1~24 사이 눈금 = 1개월 간격
    )
    hint?.let { Text(it, color = TextMuted, fontSize = 11.sp) }
}

private fun monthsLabel(m: Int): String = when {
    m < 12 -> "${m}개월"
    m % 12 == 0 -> "${m / 12}년"
    else -> "${m / 12}년 ${m % 12}개월"
}

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

@Composable
fun SettingsScreen() {
    var tickers by remember { mutableStateOf(Store.loadTickers().toList()) }
    var input by remember { mutableStateOf("") }
    var rangeM by remember { mutableStateOf(Store.lookbackMonths()) }
    val nameEdits = remember { mutableStateMapOf<String, String>().apply { putAll(Store.nameOverrides()) } }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("설정", color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold)

        // ══════════ 분석 ══════════
        SectionHeader("분석")
        SettingsCard {
            MonthSlider(
                "분석 기간 (조회)", rangeM,
                hint = if (rangeM < 3)
                    "⚠️ 회귀·MACD·RSI는 최소 30 거래일이 필요합니다 — 3개월 미만은 분석이 실패할 수 있습니다"
                else null,
            ) { rangeM = it; Store.setLookbackMonths(it); AppState.bump() }
            var chartM by remember { mutableStateOf(Store.chartMonths()) }
            MonthSlider(
                "차트 표시기간", chartM,
                hint = "분석 탭 시계열을 **처음** 열 때 보여줄 구간입니다.\n한 번 두 손가락으로 조절하면 그 구간이 기억되어, 종목을 바꾸거나 앱을 다시 켜도 유지됩니다.",
            ) { chartM = it; Store.setChartMonths(it); AppState.bump() }
        }

        // ══════════ 토스증권 ══════════
        // 계정(키 입력)은 처음 한 번만 쓰므로 접어 두고, 매일 쓰는 것만 밖으로 낸다.
        SectionHeader("토스증권 (조회 전용)")
        SettingsCard {
            var bkOpen by remember { mutableStateOf(false) }
            var bkKey by remember { mutableStateOf(BrokerCreds.appKey()) }
            var bkSecret by remember { mutableStateOf(BrokerCreds.appSecret()) }
            var bkMsg by remember { mutableStateOf<String?>(null) }
            var bkBusy by remember { mutableStateOf(false) }
            var bkVer by remember { mutableStateOf(0) }
            val bkScope = rememberCoroutineScope()
            val linked = bkVer.let { BrokerCreds.isLinked() }
            val status = when {
                !BrokerCreds.available() -> "사용 불가 (기기 보안 저장소 오류)"
                linked -> "연결됨 · 계좌 ${BrokerCreds.maskedAccount()}"
                BrokerCreds.hasKeys() -> "키만 저장됨 (연결 테스트 필요)"
                else -> "미설정"
            }

            // ── 계정 (접힘) ──
            Text("${if (bkOpen) "▲" else "▼"}  계정 · $status", color = TextSecondary, fontSize = 12.sp,
                modifier = Modifier.fillMaxWidth().clickable { bkOpen = !bkOpen })
            if (bkOpen) {
                if (!BrokerCreds.available()) {
                    Text("이 기기에서 암호화 저장소를 열지 못했습니다. 자격증명을 평문으로 저장하지 않기 위해 연동을 비활성화합니다.",
                        color = Loss, fontSize = 11.sp)
                } else {
                    OutlinedTextField(bkKey, { bkKey = it }, label = { Text("App Key (client_id)") },
                        singleLine = true, modifier = Modifier.fillMaxWidth())
                    OutlinedTextField(bkSecret, { bkSecret = it }, label = { Text("App Secret (client_secret)") },
                        singleLine = true, visualTransformation = PasswordVisualTransformation(),
                        modifier = Modifier.fillMaxWidth())
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        Button(
                            enabled = !bkBusy && bkKey.isNotBlank() && bkSecret.isNotBlank(),
                            onClick = {
                                BrokerCreds.saveKeys(bkKey, bkSecret)
                                bkBusy = true; bkMsg = "연결 중…"
                                bkScope.launch {
                                    val msg = withContext(Dispatchers.IO) {
                                        try {
                                            // 계좌 목록을 받아 accountSeq 확정 (종합매매 계좌 우선)
                                            val accts = TossApi.accounts()
                                            val a = accts.firstOrNull { it.accountType == "BROKERAGE" } ?: accts.firstOrNull()
                                            if (a == null) "조회된 계좌가 없습니다"
                                            else {
                                                BrokerCreds.saveAccount(a.accountSeq, a.accountNo)
                                                "✓ 연결됨 · 계좌 ${BrokerCreds.maskedAccount()}"
                                            }
                                        } catch (e: Exception) {
                                            val base = "실패: ${e.message}"
                                            // 허용 IP 문제면 지금 IP를 같이 알려줘야 바로 등록할 수 있다
                                            if (e is com.quant.dashboard.data.TossException && e.code == "access_denied") {
                                                val ip = NetInfo.publicIp()
                                                if (ip != null) "$base\n현재 공인 IP: $ip" else base
                                            } else base
                                        }
                                    }
                                    bkMsg = msg; bkBusy = false; bkVer++; AppState.bump()
                                }
                            },
                            modifier = Modifier.weight(1f),
                        ) { Text(if (bkBusy) "연결 중…" else "연결 테스트") }
                        Button(
                            enabled = !bkBusy,
                            onClick = {
                                BrokerCreds.clear(); bkKey = ""; bkSecret = ""
                                bkVer++; bkMsg = "삭제했습니다"; AppState.bump()
                            },
                            modifier = Modifier.weight(1f),
                        ) { Text("삭제") }
                    }

                    // 허용 IP — 휴대폰 IP 는 자주 바뀌므로 여기서 바로 확인·복사한다
                    val clipboard = LocalClipboardManager.current
                    var myIp by remember { mutableStateOf<String?>(null) }
                    var ipCopied by remember { mutableStateOf(false) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()) {
                        Button(
                            enabled = !bkBusy,
                            onClick = {
                                bkScope.launch {
                                    myIp = "확인 중…"; ipCopied = false
                                    val ip = withContext(Dispatchers.IO) { NetInfo.publicIp() }
                                    myIp = ip ?: "확인 실패"
                                }
                            },
                        ) { Text("현재 IP 확인") }
                        myIp?.let { ip ->
                            Text(ip, color = TextPrimary, fontSize = 13.sp, fontFamily = Mono,
                                maxLines = 1, modifier = Modifier.weight(1f))
                        }
                        // 확인된 IP 가 있을 때만 복사 버튼 (확인 중·실패 상태에는 복사할 게 없다)
                        val ipOk = myIp != null && myIp != "확인 중…" && myIp != "확인 실패"
                        if (ipOk) {
                            Box(
                                Modifier.clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                                    .clickable {
                                        clipboard.setText(AnnotatedString(myIp!!))
                                        ipCopied = true
                                    }
                                    .padding(horizontal = 14.dp, vertical = 8.dp),
                            ) { Text(if (ipCopied) "복사됨" else "복사", color = TextPrimary, fontSize = 12.sp,
                                fontWeight = FontWeight.Bold) }
                        }
                    }
                    if (myIp != null && myIp != "확인 중…" && myIp != "확인 실패") {
                        Text("이 값을 토스증권 WTS 허용 IP 목록에 등록하세요.",
                            color = TextMuted, fontSize = 10.sp)
                    }

                    Text("🔒 자격증명은 이 기기에만 암호화 저장되며 외부로 전송되지 않습니다.\n" +
                        "조회 전용입니다 — 주문·정정·취소 기능은 구현되어 있지 않습니다.\n" +
                        "⚠️ 토스 API는 허용 IP 목록 밖에서는 차단됩니다. 휴대폰 IP는 자주 바뀌므로\n" +
                        "토스증권 WTS(tossinvest.com) → 설정 → Open API → 허용 IP 관리에서 현재 IP를 등록해야 합니다.",
                        color = TextMuted, fontSize = 11.sp)
                }
            }
            bkMsg?.let {
                Text(it, color = if (it.startsWith("실패")) Loss else TextSecondary, fontSize = 11.sp)
            }

            // ── 데이터 (연결됐을 때만, 펼친 채로) ──
            if (linked) {
                Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                Label("체결내역 가져오기")
                Text("토스 계좌의 체결분을 매매기록으로 합칩니다. 이미 가져온 건과 손으로 입력해 둔 같은 거래는 건너뜁니다.",
                    color = TextMuted, fontSize = 11.sp)
                Button(
                    enabled = !bkBusy,
                    onClick = {
                        bkBusy = true; bkMsg = "체결내역 가져오는 중…"
                        bkScope.launch {
                            val msg = withContext(Dispatchers.IO) {
                                try { TossSync.importFills().summary() }
                                catch (e: Exception) { "실패: ${e.message}" }
                            }
                            bkMsg = msg; bkBusy = false; AppState.bump()
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) { Text("체결내역 가져오기") }

                Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                var tick by remember { mutableStateOf(Store.tickSeconds()) }
                Label("실시간 시세 갱신 주기")
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf("끔" to 0, "1초" to 1, "3초" to 3, "5초" to 5, "10초" to 10, "30초" to 30).forEach { (lab, v) ->
                        Seg(lab, tick == v) { tick = v; Store.setTickSeconds(v); AppState.bump() }
                    }
                }
                Text("장이 열려 있을 때만 동작합니다. 전 종목 현재가를 요청 1번으로 받아오므로 " +
                    "주기를 짧게 해도 호출 수는 늘지 않지만, 배터리와 레이트리밋에는 영향이 있습니다.\n" +
                    "한도(429)에 걸리면 30초 쉬었다 자동 재개합니다. 1초는 배터리 소모가 큽니다.",
                    color = TextMuted, fontSize = 11.sp)

                Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                Label("일봉 다시 받기")
                Box(
                    Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(SurfaceInput)
                        .clickable { Quotes.clearCache(); AppState.bump() }
                        .padding(vertical = 8.dp),
                    contentAlignment = Alignment.Center,
                ) { Text("받아 둔 일봉 버리고 다시 받기", color = TextSecondary, fontSize = 12.sp) }
                Text("일봉은 하루 한 번만 바뀌므로 받아 두고 재사용합니다(6시간). " +
                    "현재가는 위의 갱신 주기로 따로 받아옵니다.\n" +
                    "토스 일봉은 200개/요청이라 2년치면 종목당 3회 호출이라, 동시 요청을 3건으로 제한합니다.",
                    color = TextMuted, fontSize = 11.sp)
            }
        }

        // ══════════ 종목 관리 ══════════
        SectionHeader("종목 관리")

        // 토스 종목 유니버스(미국 3개 거래소) — 이름으로 티커를 찾기 위한 캐시. 하루 1회 갱신.
        val uniScope = rememberCoroutineScope()
        var uniCount by remember { mutableStateOf(0) }
        var uniBusy by remember { mutableStateOf(false) }
        LaunchedEffect(Unit) {
            // 파일 캐시 로드(수천 건 파싱)도 IO 로 — 메인 스레드에서 하면 화면이 끊긴다
            val linked = BrokerCreds.isLinked()
            if (linked) uniBusy = true
            uniCount = withContext(Dispatchers.IO) { if (linked) Universe.ensure() else Universe.count() }
            uniBusy = false
        }

        // 여러 개를 한 번에 붙여넣을 수 있게 여러 줄 입력 (토스 앱 관심종목 옮겨오기)
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
                placeholder = { Text(if (uniCount > 0) "티커 또는 이름 (NVDA·애플·삼성전자)" else "티커 (NVDA, AAPL, 005930)") },
                singleLine = false, maxLines = 4, modifier = Modifier.weight(1f))
            Button(onClick = {
                if (input.isNotBlank()) {
                    // 단건은 예전처럼 통째로 추가 — 이름 검색어가 공백으로 쪼개지지 않게
                    val (added, dup) =
                        if (bulk) Store.addTickers(input) else Store.addTickers(input.trim().replace(" ", ""))
                    tickers = Store.loadTickers().toList()
                    addMsg = when {
                        added == 0 && dup > 0 -> "이미 목록에 있습니다 (${dup}개)"
                        added == 0 -> null
                        dup > 0 -> "${added}개 추가 · ${dup}개는 이미 있어 건너뜀"
                        added > 1 -> "${added}개 추가"
                        else -> null
                    }
                    input = ""; AppState.bump()
                }
            }) { Text(if (bulk) "모두 추가" else "추가") }
        }
        addMsg?.let { Text(it, color = TextSecondary, fontSize = 11.sp) }

        // 이름 검색 결과 — 탭하면 바로 추가 (토스 연동 시에만 동작)
        // 구분자가 섞인 붙여넣기는 검색이 아니라 일괄 추가로 처리하므로 후보를 띄우지 않는다
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

        Text("최소 ${Store.MIN_TICKERS}개 · 국내=6자리(.KS/.KQ 자동) · 셀에서 별칭 편집\n" +
            "여러 개를 콤마·줄바꿈으로 구분해 한 번에 붙여넣을 수 있습니다",
            color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(vertical = 2.dp))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                when {
                    uniBusy -> "종목 목록 받는 중…"
                    uniCount > 0 -> "🔎 이름 검색 가능 — 미국·국내 ${"%,d".format(uniCount)}종목 (${Universe.cachedDate().ifEmpty { "부분" }})"
                    BrokerCreds.isLinked() -> "종목 목록을 아직 받지 못했습니다"
                    else -> "토스 연동 시 이름으로 종목을 찾을 수 있습니다"
                },
                color = TextMuted, fontSize = 11.sp, modifier = Modifier.weight(1f),
            )
            if (BrokerCreds.isLinked()) {
                Box(
                    Modifier.clip(RoundedCornerShape(6.dp)).background(SurfaceInput)
                        .clickable(enabled = !uniBusy) {
                            uniScope.launch {
                                uniBusy = true
                                uniCount = withContext(Dispatchers.IO) { Universe.ensure(force = true) }
                                uniBusy = false
                            }
                        }
                        .padding(horizontal = 10.dp, vertical = 5.dp),
                ) { Text("목록 갱신", color = TextSecondary, fontSize = 11.sp) }
            }
        }

        tickers.chunked(2).forEach { pair ->
            Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.Top) {
                pair.forEach { tk ->
                    TickerCell(
                        tk = tk, weight = 1f,
                        nameValue = nameEdits[tk] ?: "",
                        onName = { nameEdits[tk] = it },
                        onDelete = { Store.removeTicker(tk); tickers = Store.loadTickers().toList(); AppState.bump() },
                    )
                }
                if (pair.size == 1) Spacer(Modifier.weight(1f))
            }
        }
        Button(onClick = {
            nameEdits.forEach { (tk, nm) -> Store.setNameOverride(tk, nm) }
            AppState.bump()
        }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) { Text("이름 저장") }
        Text("이름 칸을 비우고 저장하면 기본 표시명으로 복귀.", color = TextMuted, fontSize = 11.sp)
        Spacer(Modifier.height(8.dp))
    }
}

/** 종목 관리 카드 — 티커(상단) / 별칭 + 삭제(하단). */
@Composable
private fun RowScope.TickerCell(
    tk: String, weight: Float, nameValue: String,
    onName: (String) -> Unit, onDelete: () -> Unit,
) {
    Column(
        Modifier.weight(weight).clip(RoundedCornerShape(12.dp)).background(BgCard)
            .border(1.dp, BorderColor, RoundedCornerShape(12.dp)).padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(tk, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold, fontFamily = Mono,
            maxLines = 1, modifier = Modifier.fillMaxWidth())
        // 하단: 별칭 입력 + 삭제
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            BasicTextField(
                value = nameValue, onValueChange = onName, singleLine = true,
                textStyle = TextStyle(color = TextPrimary, fontSize = 11.sp),
                cursorBrush = SolidColor(TextPrimary),
                modifier = Modifier.weight(1f).background(SurfaceInput, RoundedCornerShape(4.dp))
                    .padding(horizontal = 6.dp, vertical = 6.dp),
                decorationBox = { inner ->
                    Box(Modifier.fillMaxWidth()) {
                        if (nameValue.isEmpty()) Text("별칭(선택)", color = TextMuted, fontSize = 11.sp)
                        inner()
                    }
                },
            )
            Text("삭제", color = Loss, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                modifier = Modifier.clickable { onDelete() }.padding(horizontal = 4.dp, vertical = 4.dp))
        }
    }
}
