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

    fun load() {
        if (loading) return
        loading = true
        viewModelScope.launch {
            apply(withContext(Dispatchers.IO) { runCatching { Server.rebalance() } })
            loading = false
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
