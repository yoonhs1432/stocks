package com.quant.dashboard.data

import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * 토스 계좌 → 로컬 데이터 동기화. **읽기만 한다.**
 *
 * - 체결내역(`/api/v1/orders?status=CLOSED`)을 로컬 매매기록으로 가져온다.
 * - 이미 가져온 건은 `Trade.srcId`(=orderId)로, 손으로 입력해 둔 같은 거래는
 *   (날짜·매수매도·수량·단가) 근사 일치로 걸러 중복을 만들지 않는다.
 */
object TossSync {

    data class ImportResult(
        val added: Int,
        val alreadyHad: Int,
        val matchedManual: Int,   // 수동 입력분과 같은 거래로 판단해 건너뜀
        val fractionalSkipped: Int,
    ) {
        fun summary(): String = buildString {
            append("체결 ${added}건 추가")
            if (alreadyHad > 0) append(" · 기존 ${alreadyHad}건")
            if (matchedManual > 0) append(" · 수동기록과 동일 ${matchedManual}건")
            if (fractionalSkipped > 0) append(" · 소수점 수량 ${fractionalSkipped}건 제외")
        }
    }

    /** 같은 거래로 볼 수 있는지 — 날짜·방향이 같고 수량 동일, 단가가 0.5% 이내. */
    private fun sameTrade(t: Trade, f: TossApi.Fill, qty: Int): Boolean {
        if (t.date != f.date) return false
        if (t.type != (if (f.buy) "buy" else "sell")) return false
        if (t.qty != qty) return false
        if (t.price <= 0.0 || f.price <= 0.0) return false
        return abs(t.price - f.price) / f.price < 0.005
    }

    /**
     * 체결내역을 매매기록에 병합.
     *
     * @param from 조회 시작일 (YYYY-MM-DD). null 이면 전체 기간.
     *
     * ⚠️ 로컬 `Trade.qty` 가 Int 라 미국 주식 소수점 체결은 담을 수 없다.
     *    반올림해서 0이 되는 건(1주 미만)은 세어서 알리고 건너뛴다.
     */
    fun importFills(from: String? = null): ImportResult {
        val seq = BrokerCreds.accountSeq()
        require(seq >= 0) { "계좌가 연결되지 않았습니다" }

        val fills = TossApi.fills(seq, from)
        val trades = Store.loadTrades()
        var added = 0; var had = 0; var manual = 0; var frac = 0

        for (f in fills) {
            val list = trades.getOrPut(f.symbol) { ArrayList() }
            // 이미 같은 orderId 로 가져온 적이 있으면 skip
            if (list.any { it.srcId == f.orderId }) { had++; continue }

            val qty = f.quantity.roundToInt()
            if (qty <= 0) { frac++; continue }

            // 손으로 입력해 둔 같은 거래가 있으면 그 기록에 출처만 달아 준다 (중복 생성 방지)
            val idx = list.indexOfFirst { it.srcId == null && sameTrade(it, f, qty) }
            if (idx >= 0) {
                list[idx] = list[idx].copy(srcId = f.orderId)
                manual++
                continue
            }

            list.add(
                Trade(
                    date = f.date,
                    type = if (f.buy) "buy" else "sell",
                    qty = qty,
                    price = f.price,
                    memo = null,
                    srcId = f.orderId,
                )
            )
            added++
        }
        if (added > 0 || manual > 0) {
            // 종목별 날짜 오름차순 정렬 (사이클 계산이 시간순을 전제로 함)
            for ((k, v) in trades) trades[k] = v.sortedBy { it.date }.toMutableList()
            Store.saveTrades(trades)
        }
        return ImportResult(added, had, manual, frac)
    }

    // ── 보유 자산 스냅샷 (포트폴리오 탭 "실제 계좌" 카드용) ──

    @Volatile private var holdingsCache: TossApi.Holdings? = null
    @Volatile private var holdingsAt = 0L

    fun cachedHoldings(): TossApi.Holdings? = holdingsCache

    /** force=false면 5분 캐시 재사용. IO 디스패처에서 호출. */
    fun holdings(force: Boolean = false): TossApi.Holdings? {
        val seq = BrokerCreds.accountSeq()
        if (seq < 0) return null
        val now = System.currentTimeMillis()
        val c = holdingsCache
        if (!force && c != null && now - holdingsAt < 300_000) return c
        return try {
            val h = TossApi.holdings(seq)
            holdingsCache = h; holdingsAt = now
            h
        } catch (e: Exception) {
            c   // 실패 시 이전 스냅샷 유지
        }
    }

    fun clear() { holdingsCache = null; holdingsAt = 0L }
}
