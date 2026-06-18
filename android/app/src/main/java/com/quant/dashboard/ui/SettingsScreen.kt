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
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.data.Store
import com.quant.dashboard.data.Tickers
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.Loss
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary

@Composable
fun SettingsScreen() {
    var tickers by remember { mutableStateOf(Store.loadTickers().toList()) }
    var input by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().background(BgApp)
            .verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text("⚙️ 설정 — 종목 관리", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Bold)

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
                }
            }) { Text("추가") }
        }
        Text("최소 1개 유지 · 잘못된 티커는 데이터 로드 실패 (한국=6자리 코드)",
            color = TextSecondary, fontSize = 11.sp)

        tickers.forEach { tk ->
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("${Tickers.displayName(tk)}  ($tk)", color = TextPrimary, fontSize = 14.sp)
                TextButton(onClick = {
                    Store.removeTicker(tk)
                    tickers = Store.loadTickers().toList()
                }) { Text("삭제", color = Loss) }
            }
        }

        Text("변경 후 분석·비교 탭은 새로고침(🔄) 시 반영됩니다.",
            color = TextSecondary, fontSize = 11.sp)
    }
}
