package com.quant.dashboard.ui

import android.app.Application
import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quant.dashboard.data.Server
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 홈(맞추기)·바구니 탭이 같이 쓰는 리밸런싱 상태. 액티비티에 묶여 두 탭이 같은 값을 본다.
 *
 * 체크(어디까지 주문했는지)는 **그 기간(periodEnd)** 에만 유효하게 파일에 남긴다 —
 * 앱을 껐다 켜도 이어서 할 수 있고, 다음 분기가 되면 저절로 비워진다.
 */
class RebalanceViewModel(app: Application) : AndroidViewModel(app) {
    var plan by mutableStateOf<Server.Rebalance?>(null)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var loading by mutableStateOf(false)
        private set
    var checked by mutableStateOf<Set<String>>(emptySet())
        private set

    private val prefs = app.getSharedPreferences("rebalance_ui", Context.MODE_PRIVATE)

    private fun apply(r: Result<Server.Rebalance>) {
        r.onSuccess {
            plan = it; error = null
            val key = it.status()
            if (prefs.getString("checkedFor", "") != key) {
                prefs.edit().putString("checkedFor", key).putStringSet("checked", emptySet()).apply()
            }
            checked = prefs.getStringSet("checked", emptySet())?.toSet().orEmpty()
        }.onFailure { error = it.message }
    }

    /** 체크를 묶는 열쇠 — 기간 끝 + 마지막으로 맞춘 날. '다 맞췄어요' 를 누르면 바뀐다. */
    private fun Server.Rebalance.status() = "$periodEnd|${lastDone.orEmpty()}"

    private var busy = false

    /**
     * @param quiet 자동 갱신 — 당겨서 새로고침 표시(loading)를 켜지 않는다
     * @param force 당겨서 새로고침 — PC 의 계좌 캐시를 건너뛰고, 자산 기록(홈 차트)도 새로 받는다
     */
    fun load(quiet: Boolean = false, force: Boolean = false) {
        if (busy && !force) return
        busy = true
        if (!quiet) loading = true
        viewModelScope.launch {
            if (force) launch(Dispatchers.IO) { runCatching { com.quant.dashboard.data.Snapshots.ensure(force = true) } }
            apply(withContext(Dispatchers.IO) { runCatching { Server.rebalance(force) } })
            busy = false
            loading = false
        }
    }

    /**
     * 화면이 보이는 동안 — 장중이면 1분마다, 아직 못 받았으면 8초마다 조용히 다시 받는다.
     * 예전 비교·포트폴리오 탭과 같은 주기다. PC 는 계좌를 20초 캐시하므로 토스를 더 부르지 않는다.
     */
    suspend fun autoRefresh() {
        while (true) {
            kotlinx.coroutines.delay(if (plan == null) 8_000 else 60_000)
            if (plan == null || com.quant.dashboard.data.MarketHours.anyOpen()) load(quiet = true)
        }
    }

    fun save(basket: List<String>? = null, cashPct: Int? = null, period: String? = null,
             fractional: Boolean? = null, sellOutside: Boolean? = null) {
        viewModelScope.launch {
            apply(withContext(Dispatchers.IO) {
                runCatching { Server.saveRebalance(basket, cashPct, period, fractional, sellOutside) }
            })
        }
    }

    fun done() {
        viewModelScope.launch { apply(withContext(Dispatchers.IO) { runCatching { Server.rebalanceDone() } }) }
    }

    fun toggle(ticker: String) {
        val next = if (ticker in checked) checked - ticker else checked + ticker
        checked = next
        prefs.edit().putStringSet("checked", next).apply()
    }

    fun check(ticker: String) {
        if (ticker !in checked) toggle(ticker)
    }
}
