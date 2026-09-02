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

    // ── 토스 기반 계좌 전체 (평가금액 + 예수금 = 총자산) ──

    /** 계좌 한 장 요약. 금액은 통화별 원본, `totalKrw` 만 환율로 환산. */
    data class Account(
        val holdings: TossApi.Holdings,
        val krwCash: Double,
        val usdCash: Double,
        val rate: Double,
    ) {
        val krwEval: Double get() = holdings.krwEval
        val usdEval: Double get() = holdings.usdEval
        /** 원화 환산 총자산 = 평가금액 + 매수가능금액. */
        fun totalKrw(): Double = krwEval + krwCash + (usdEval + usdCash) * rate
        /** 평가손익 합계(원화 환산). */
        fun pnlKrw(): Double = holdings.krwPnl + holdings.usdPnl * rate
        /** 수수료·세금 공제 후 평가손익(원화 환산) — 증권사 앱이 보통 이 값을 보여준다. */
        fun pnlAfterCostKrw(): Double = holdings.krwPnlAfterCost + holdings.usdPnlAfterCost * rate
        /** 당일 손익(원화 환산). */
        fun dailyPnlKrw(): Double = holdings.krwDailyPnl + holdings.usdDailyPnl * rate
    }

    @Volatile private var accountCache: Account? = null
    @Volatile private var accountAt = 0L

    fun cachedAccount(): Account? = accountCache

    /**
     * 보유 + 예수금 + 환율을 한 번에. force=false면 5분 캐시.
     * 성공하면 오늘 자 잔고 스냅샷을 남긴다(자산추이용, 하루 1회 덮어쓰기).
     */
    fun account(force: Boolean = false): Account? {
        val seq = BrokerCreds.accountSeq()
        if (seq < 0) return null
        val now = System.currentTimeMillis()
        val c = accountCache
        if (!force && c != null && now - accountAt < 300_000) return c
        return try {
            val h = TossApi.holdings(seq)
            // 예수금·환율은 실패해도 보유 현황은 보여준다
            val krw = runCatching { TossApi.buyingPower(seq, "KRW") }.getOrDefault(0.0)
            val usd = runCatching { TossApi.buyingPower(seq, "USD") }.getOrDefault(0.0)
            val rate = runCatching { TossApi.usdKrw() }.getOrNull()?.takeIf { it > 0 && !it.isNaN() } ?: 1400.0
            val a = Account(h, krw, usd, rate)
            accountCache = a; accountAt = now
            holdingsCache = h; holdingsAt = now
            // 손익·매입금액까지 같이 남긴다 — 없으면 입금과 수익을 구분할 수 없다
            Snapshots.recordToday(
                a.krwEval, a.usdEval, krw, usd, rate,
                krwPnl = h.krwPnl, usdPnl = h.usdPnl,
                krwPurchase = h.krwPurchase, usdPurchase = h.usdPurchase,
            )
            a
        } catch (e: Exception) {
            c   // 실패 시 이전 스냅샷 유지
        }
    }
}
