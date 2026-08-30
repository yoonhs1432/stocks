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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.Gist
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.input.PasswordVisualTransformation
import com.quant.dashboard.data.BrokerCreds
import com.quant.dashboard.data.TossApi
import com.quant.dashboard.data.MarketHours
import com.quant.dashboard.data.NetInfo
import com.quant.dashboard.data.TossSync
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
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
import java.time.LocalDate
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
    var seed by remember { mutableStateOf(Store.seedUsd().toInt().toString()) }
    var rangeM by remember { mutableStateOf(Store.lookbackMonths()) }
    var interval by remember { mutableStateOf(Store.candleInterval()) }
    val nameEdits = remember { mutableStateMapOf<String, String>().apply { putAll(Store.nameOverrides()) } }
    var indivVer by remember { mutableStateOf(0) }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("설정", color = TextPrimary, fontSize = 19.sp, fontWeight = FontWeight.Bold)

        // ══════════ 분석 ══════════
        SectionHeader("분석")
        SettingsCard {
            Label("시드 ($)")
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(seed, { seed = it }, singleLine = true, modifier = Modifier.weight(1f))
                Button(onClick = { seed.toDoubleOrNull()?.let { if (it > 0) { Store.setSeedUsd(it); AppState.bump() } } }) { Text("저장") }
            }
            MonthSlider(
                "분석 기간 (조회)", rangeM,
                hint = if (rangeM < 3)
                    "⚠️ 회귀·MACD·RSI는 최소 30 거래일이 필요합니다 — 3개월 미만은 분석이 실패할 수 있습니다"
                else null,
            ) { rangeM = it; Store.setLookbackMonths(it); AppState.bump() }
            Label("봉 기준")
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf("일봉" to "1d", "주봉" to "1wk").forEach { (label, iv) ->
                    Seg(label, interval == iv) { interval = iv; Store.setCandleInterval(iv); AppState.bump() }
                }
            }
            var chartM by remember { mutableStateOf(Store.chartMonths()) }
            MonthSlider("차트 표시기간", chartM) { chartM = it; Store.setChartMonths(it); AppState.bump() }
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
            AppState.asof?.let { Text("현재 기준일: $it (헤더 ✕로 해제)", color = TextMuted, fontSize = 11.sp) }
        }

        // ══════════ 포트폴리오 ══════════
        SectionHeader("포트폴리오")
        SettingsCard {
            var eqM by remember { mutableStateOf(Store.equityMonths()) }
            MonthSlider(
                "자산추이 기간", eqM,
                hint = "그래프는 항상 일 단위로 그립니다 — 주/월로 묶으면 같은 기간이 성기게 보일 뿐이라 없앴습니다.\n" +
                    "차트에서 핀치로 확대하고, 아무 지점이나 누르면 그 시점의 금액이 표시됩니다.",
            ) { eqM = it; Store.setEquityMonths(it); AppState.bump() }
        }

        // ══════════ 데이터 (Gist) — 접힘 ══════════
        SectionHeader("데이터 (Gist 연동)")
        SettingsCard {
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
                    color = TextMuted, fontSize = 11.sp)
            }
        }

        // ══════════ 토스증권 연동 (조회 전용) ══════════
        SectionHeader("토스증권 연동 (조회 전용)")
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
            Text("${if (bkOpen) "▲" else "▼"}  $status", color = TextSecondary, fontSize = 12.sp,
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
                                    bkMsg = msg; bkBusy = false; bkVer++
                                }
                            },
                            modifier = Modifier.weight(1f),
                        ) { Text(if (bkBusy) "연결 중…" else "연결 테스트") }
                        Button(
                            enabled = !bkBusy,
                            onClick = {
                                BrokerCreds.clear(); bkKey = ""; bkSecret = ""
                                bkVer++; bkMsg = "삭제했습니다"
                            },
                            modifier = Modifier.weight(1f),
                        ) { Text("삭제") }
                    }
                    val clipboard = LocalClipboardManager.current
                    var myIp by remember { mutableStateOf<String?>(null) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()) {
                        Button(
                            enabled = !bkBusy,
                            onClick = {
                                bkScope.launch {
                                    myIp = "확인 중…"
                                    val ip = withContext(Dispatchers.IO) { NetInfo.publicIp() }
                                    myIp = ip ?: "확인 실패"
                                }
                            },
                        ) { Text("현재 IP 확인") }
                        myIp?.let { ip ->
                            Text(ip, color = TextPrimary, fontSize = 12.sp, fontFamily = Mono,
                                modifier = Modifier.weight(1f).clickable {
                                    clipboard.setText(AnnotatedString(ip))
                                })
                        }
                    }
                    if (myIp != null && myIp != "확인 중…" && myIp != "확인 실패") {
                        Text("탭하면 복사됩니다. 이 값을 WTS 허용 IP 목록에 등록하세요.",
                            color = TextMuted, fontSize = 10.sp)
                    }

                    bkMsg?.let {
                        Text(it, color = if (it.startsWith("실패")) Loss else TextSecondary, fontSize = 11.sp)
                    }

                    // ── 연결된 뒤에만: 체결내역 가져오기 · 시세 소스 ──
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

                        // ── 종목 커버리지 (이관·시세 전환 판단용) ──
                        Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                        Label("종목 커버리지 확인")
                        Text("워치리스트 + 회귀 기준(SPY)이 토스에서 취급되는지 한 번에 확인합니다. " +
                            "여기 없는 종목은 토스에서 사고팔 수 없고, 시세도 Yahoo로 받게 됩니다.",
                            color = TextMuted, fontSize = 11.sp)
                        var cov by remember { mutableStateOf<List<Triple<String, Boolean, String>>?>(null) }
                        var covBusy by remember { mutableStateOf(false) }
                        Button(
                            enabled = !covBusy,
                            onClick = {
                                covBusy = true
                                bkScope.launch {
                                    val rows = withContext(Dispatchers.IO) {
                                        val syms = (Store.loadTickers() + Tickers.BASE).distinct()
                                        try {
                                            val found = TossApi.stocks(syms)
                                            syms.map { sym ->
                                                val i = found[sym]
                                                if (i == null) Triple(sym, false, "토스에 없음 → Yahoo")
                                                else Triple(sym, i.status == "ACTIVE",
                                                    "${i.name} · ${i.market} · ${i.securityType}" +
                                                        if (i.status != "ACTIVE") " · ${i.status}" else "")
                                            }
                                        } catch (e: Exception) {
                                            listOf(Triple("오류", false, e.message ?: ""))
                                        }
                                    }
                                    cov = rows; covBusy = false
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                        ) { Text(if (covBusy) "확인 중…" else "종목 커버리지 확인") }
                        cov?.let { rows ->
                            val ok = rows.count { it.second }
                            Text("토스 취급 $ok / ${rows.size}", color = TextSecondary,
                                fontSize = 12.sp, fontWeight = FontWeight.Bold)
                            rows.forEach { (sym, okOne, desc) ->
                                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                    Text(if (okOne) "✓" else "✗", color = if (okOne) Profit else Loss,
                                        fontSize = 11.sp, fontWeight = FontWeight.Bold)
                                    Text(sym, color = TextPrimary, fontSize = 11.sp, fontFamily = Mono)
                                    Text(desc, color = TextMuted, fontSize = 11.sp)
                                }
                            }
                        }

                        // ── 실시간 시세 ──
                        Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                        var tick by remember { mutableStateOf(Store.tickSeconds()) }
                        Label("실시간 시세 갱신 주기")
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            listOf("끔" to 0, "3초" to 3, "5초" to 5, "10초" to 10, "30초" to 30).forEach { (lab, v) ->
                                Seg(lab, tick == v) { tick = v; Store.setTickSeconds(v); AppState.bump() }
                            }
                        }
                        Text("장이 열려 있을 때만 동작합니다. 전 종목 현재가를 요청 1번으로 받아오므로 " +
                            "주기를 짧게 해도 호출 수는 늘지 않지만, 배터리와 레이트리밋에는 영향이 있습니다.\n" +
                            "한도(429)에 걸리면 30초 쉬었다 자동 재개합니다.",
                            color = TextMuted, fontSize = 11.sp)

                        // ── 시간외 데이터 진단 ──
                        Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                        Label("시세 진단 (프리·애프터 확인용)")
                        Text("`/prices` 와 `/trades` 응답의 시각을 그대로 보여줍니다. 프리마켓 시간대(저녁)에 눌러 " +
                            "그 시각이 찍히면 시간외 데이터가 오는 것이고, 전일 마감 시각이면 오지 않는 것입니다.",
                            color = TextMuted, fontSize = 11.sp)
                        var diag by remember { mutableStateOf<List<String>?>(null) }
                        var diagBusy by remember { mutableStateOf(false) }
                        Button(
                            enabled = !diagBusy,
                            onClick = {
                                diagBusy = true
                                bkScope.launch {
                                    val lines = withContext(Dispatchers.IO) {
                                        try {
                                            val syms = (Store.loadTickers().take(3) + Tickers.BASE).distinct()
                                            val out = ArrayList<String>()
                                            MarketHours.ensure(force = true)
                                            out.add("개장: ${MarketHours.label() ?: "닫힘"}")
                                            out.add("기기 시각: ${java.time.OffsetDateTime.now()}")
                                            TossApi.pricesRaw(syms).forEach {
                                                out.add("prices ${it.symbol} = ${it.price} @ ${it.timestamp}")
                                            }
                                            syms.forEach { sym ->
                                                val t = runCatching { TossApi.lastTradeTime(sym) }.getOrNull()
                                                out.add("trades $sym 최근체결 @ ${t ?: "(없음)"}")
                                            }
                                            out
                                        } catch (e: Exception) {
                                            listOf("실패: ${e.message}")
                                        }
                                    }
                                    diag = lines; diagBusy = false
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                        ) { Text(if (diagBusy) "확인 중…" else "시세 진단") }
                        diag?.let { lines ->
                            lines.forEach {
                                Text(it, color = if (it.startsWith("실패")) Loss else TextSecondary,
                                    fontSize = 10.sp, fontFamily = Mono)
                            }
                        }

                        // ── 토스 기반 모드 ──
                        Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                        var tossM by remember { mutableStateOf(Store.tossMode()) }
                        Label("포트폴리오 기준")
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Seg("매매기록", !tossM) { tossM = false; Store.setTossMode(false); AppState.bump() }
                            Seg("토스 계좌", tossM) { tossM = true; Store.setTossMode(true); AppState.bump() }
                        }
                        Text("토스 계좌: 보유·평단·예수금·총자산을 증권사 실측으로 표시하고, " +
                            "이관 전 수기 매매기록은 화면에서 숨깁니다(파일은 지우지 않으므로 되돌릴 수 있음).\n" +
                            "자산추이는 토스에 과거 잔고 API가 없어 앱을 열 때마다 하루 1회 저장한 스냅샷으로 그립니다 — " +
                            "전환 시점부터 쌓입니다.",
                            color = TextMuted, fontSize = 11.sp)

                        Box(Modifier.fillMaxWidth().height(1.dp).background(BorderColor))
                        var tossQ by remember { mutableStateOf(Store.tossQuotes()) }
                        Label("시세 소스")
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Seg("Yahoo", !tossQ) { tossQ = false; Store.setTossQuotes(false); AppState.bump() }
                            Seg("토스", tossQ) { tossQ = true; Store.setTossQuotes(true); AppState.bump() }
                        }
                        Text("토스 시세는 일봉 200개/요청이라 2년치면 종목당 3회 호출이 필요합니다.\n" +
                            "조회되지 않는 종목과 코인(BTC·ETH)은 자동으로 Yahoo로 넘어갑니다.",
                            color = TextMuted, fontSize = 11.sp)
                    }
                    Text("🔒 자격증명은 이 기기에만 암호화 저장되며 Gist·서버로 전송되지 않습니다.\n" +
                        "조회 전용입니다 — 주문·정정·취소 기능은 구현되어 있지 않습니다.\n" +
                        "⚠️ 토스 API는 허용 IP 목록 밖에서는 차단됩니다. 휴대폰 IP는 자주 바뀌므로\n" +
                        "토스증권 WTS(tossinvest.com) → 설정 → Open API → 허용 IP 관리에서 현재 IP를 등록해야 합니다.",
                        color = TextMuted, fontSize = 11.sp)
                }
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
                placeholder = { Text(if (uniCount > 0) "티커 또는 이름 (NVDA·애플)" else "티커 (NVDA, AAPL, PLTR)") },
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

        Text("최소 ${Store.MIN_TICKERS}개 · 한국=6자리(.KS/.KQ 자동) · 개별/ETF·이름 셀에서 편집\n" +
            "여러 개를 콤마·줄바꿈으로 구분해 한 번에 붙여넣을 수 있습니다",
            color = TextMuted, fontSize = 11.sp, modifier = Modifier.padding(vertical = 2.dp))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                when {
                    uniBusy -> "종목 목록 받는 중…"
                    uniCount > 0 -> "🔎 이름 검색 가능 — 미국 ${"%,d".format(uniCount)}종목 (${Universe.cachedDate().ifEmpty { "부분" }})"
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

        indivVer.let {
            tickers.chunked(2).forEach { pair ->
                Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.Top) {
                    pair.forEach { tk ->
                        TickerCell(
                            tk = tk, weight = 1f,
                            indiv = Store.isIndividual(tk),
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
        }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) { Text("이름 저장") }
        Text("이름 칸을 비우고 저장하면 기본 표시명으로 복귀.", color = TextMuted, fontSize = 11.sp)
        Spacer(Modifier.height(8.dp))
    }
}

/** 종목 관리 카드 — 티커 + ETF/개별 뱃지(상단) / 별칭 + 삭제(하단). */
@Composable
private fun RowScope.TickerCell(
    tk: String, weight: Float, indiv: Boolean, nameValue: String,
    onName: (String) -> Unit, onToggle: () -> Unit, onDelete: () -> Unit,
) {
    Column(
        Modifier.weight(weight).clip(RoundedCornerShape(12.dp)).background(BgCard)
            .border(1.dp, BorderColor, RoundedCornerShape(12.dp)).padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        // 상단: 티커 + ETF/개별 뱃지
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(tk, color = TextPrimary, fontSize = 14.sp, fontWeight = FontWeight.Bold, fontFamily = Mono,
                maxLines = 1, modifier = Modifier.weight(1f))
            Box(
                Modifier.clip(RoundedCornerShape(6.dp))
                    .background(if (indiv) Profit.copy(alpha = 0.18f) else SurfaceInput)
                    .clickable { onToggle() }
                    .padding(horizontal = 8.dp, vertical = 3.dp),
            ) {
                Text(if (indiv) "개별" else "ETF",
                    color = if (indiv) Profit else TextMuted, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
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
