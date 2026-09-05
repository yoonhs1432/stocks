package com.quant.dashboard.ui

import com.quant.dashboard.data.Store

/**
 * 분석 탭 시계열의 x축 구간 — **종목을 바꿔도, 앱을 껐다 켜도 그대로 유지한다.**
 *
 * 예전에는 화면 안 `remember` 에 뒀다가 ViewModel 로 옮겼는데, 그래도 종목을 바꾸거나
 * 앱을 다시 켜면 "차트 표시기간" 기본값으로 돌아갔다. 사용자가 두 손가락으로 맞춘 구간은
 * 그 자체가 의도이므로 설정 파일에 남긴다.
 *
 * ⚠️ 배율(`sx`)을 그대로 저장하면 안 된다 — 배율은 **그 종목의 전체 기간에 대한 비율**이라
 * 데이터 길이가 다른 종목(신규 상장 등)에 옮기면 보이는 기간이 달라진다.
 * 그래서 **보이는 기간(개월)** 과 **오른쪽 끝이 최신에서 떨어진 정도(개월)** 로 저장하고,
 * 종목마다 그 종목의 전체 기간에 맞춰 배율을 다시 계산한다.
 */
object ChartRange {

    /** 보이는 기간(개월). 0 이하면 미설정 → 2개월. 한 번 조절하면 그 값이 남는다. */
    private var months: Double = -1.0

    /** 오른쪽 끝이 최신 봉에서 떨어진 정도(개월). 0 = 최신에 붙임. */
    private var endOffset: Double = 0.0

    private var loaded = false
    private var lastSaveAt = 0L
    private var dirty = false

    private fun ensureLoaded() {
        if (loaded) return
        months = Store.chartRangeMonths()
        endOffset = Store.chartRangeEnd()
        loaded = true
    }

    /** 지금 저장된 구간을 이 종목(전체 기간 = totalMonths)의 배율로 환산. */
    fun viewFor(totalMonths: Double): ChartView {
        ensureLoaded()
        val total = totalMonths.coerceAtLeast(0.1)
        val want = if (months > 0) months else 2.0
        // 보이는 기간이 전체보다 길면 전체를 보여준다(배율 1 미만은 없다)
        val span = (want / total).coerceIn(1.0 / ChartView.MAX_ZOOM, 1.0).toFloat()
        val sx = 1f / span
        // 오른쪽 끝 위치 — 최신에서 endOffset 개월 앞. 범위를 벗어나면 붙여 맞춘다
        val u1 = (1.0 - endOffset / total).coerceIn(span.toDouble(), 1.0)
        val u0 = (u1 - span).coerceIn(0.0, (1.0 - span).coerceAtLeast(0.0))
        return ChartView(sx = sx, nx = (-u0 * sx).toFloat())
    }

    /**
     * 사용자가 조절한 구간을 기억한다. 제스처마다 불리므로 파일 쓰기는 throttle 하고,
     * 메모리 값은 즉시 갱신한다(탭·종목을 바꾸면 그 값이 바로 쓰인다).
     */
    fun save(v: ChartView, totalMonths: Double) {
        ensureLoaded()
        val total = totalMonths.coerceAtLeast(0.1)
        val (u0, u1) = v.visibleX()
        months = (u1 - u0).toDouble() * total
        endOffset = (1.0 - u1).toDouble() * total
        dirty = true
        val now = System.currentTimeMillis()
        if (now - lastSaveAt > 700) { lastSaveAt = now; flush() }
    }

    /** 남은 변경분을 파일에 쓴다 (화면을 벗어날 때 호출). */
    fun flush() {
        if (!dirty) return
        dirty = false
        Store.setChartRange(months, endOffset)
    }
}
