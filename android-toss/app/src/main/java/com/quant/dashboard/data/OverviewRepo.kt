package com.quant.dashboard.data

import com.quant.dashboard.quant.Portfolio
import com.quant.dashboard.quant.Quant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext

/**
 * 전 종목 요약(현재가·등락·Z·M·β·σ·신호·보유)을 한 번 받아 공유 캐시.
 * 비교 표·σ·β 산점도·Z·M 산점도·분석 탭 종목 버튼이 공통으로 사용.
 */
object OverviewRepo {
    data class Row(
        val ticker: String, val name: String, val price: Double,
        val prevClose: Double,     // 전일 종가 — 실시간 현재가로 등락률을 다시 계산할 때 사용
        val day: Double, val week: Double, val fromHigh: Double,
        val zPct: Double, val mPct: Double, val signal: String,
        val beta: Double, val sigmaPct: Double,
        // 미니 캔들용 당일 시/고/저 (없으면 NaN)
        val open: Double = Double.NaN,
        val high: Double = Double.NaN,
        val low: Double = Double.NaN,
        val holding: Boolean,      // 현재 보유 중 (★)
        val hasHistory: Boolean,   // 과거 매매 이력만 (☆)
    )

    /** 목록 캐시 슬롯. 조회기간·기준일이 바뀌면 무효화. */
    private class Slot {
        @Volatile var rows: List<Row> = emptyList()
        @Volatile var ts = 0L
        @Volatile var key = ""
    }
    private val watchSlot = Slot()

    fun cached(): List<Row> = watchSlot.rows

    /**
     * 워치리스트 + **토스 계좌 보유종목**. force=false면 5분 캐시 재사용. IO 디스패처에서 호출 권장.
     *
     * 보유종목을 워치리스트 파일에 넣지 않고 여기서 합치는 이유 — 계좌 상태에 따라 계속 바뀌므로
     * 파일에 박아 두면 전량 매도 후에 손으로 지워야 한다.
     */
    suspend fun load(force: Boolean = false): List<Row> {
        // 캐시만 읽으면 포트폴리오 탭을 아직 안 열었을 때 보유종목이 비어 목록에서 빠진다.
        // account() 는 5분 캐시라 여기서 불러도 추가 요청이 거의 없다.
        val acct = withContext(Dispatchers.IO) {
            if (force) runCatching { TossSync.account(force = true) }.getOrNull()
                ?: TossSync.cachedAccount()
            else TossSync.cachedAccount() ?: runCatching { TossSync.account() }.getOrNull()
        }
        val held = acct?.holdings?.items?.filter { it.quantity > 0 }?.map { it.symbol }.orEmpty()
        val tickers = (Store.loadTickers() + held).distinct()
        return loadInto(watchSlot, tickers, force, keyExtra = held.sorted().joinToString(","))
    }

    private suspend fun loadInto(
        slot: Slot, tickers: List<String>, force: Boolean, keyExtra: String = "",
    ): List<Row> {
        val now = System.currentTimeMillis()
        val months = Store.lookbackMonths()
        val curKey = "$months|$keyExtra"
        if (!force && slot.rows.isNotEmpty() && slot.key == curKey && now - slot.ts < 300_000) return slot.rows
        // force=사용자가 당겨서 새로고침 — 일봉 캐시(6시간)까지 무시하고 다시 받는다
        val spy = Quotes.closes(Tickers.BASE, months, force)
        if (spy.isEmpty()) return slot.rows
        val trades = Store.visibleTrades()
        val acct = TossSync.cachedAccount()
        // ★(보유)는 매매기록 계산이 아니라 실제 계좌 잔고로 판정한다
        val tossHeld: Set<String>? = acct?.holdings?.items
            ?.filter { it.quantity > 0 }?.map { it.symbol }?.toSet()
        // 시세를 못 받은 보유종목의 최후 수단 — 계좌가 알려주는 현재가
        val heldPrice: Map<String, Double> =
            acct?.holdings?.items?.associate { it.symbol to it.lastPrice }.orEmpty()
        val rows = coroutineScope {
            tickers.map { tk ->
                async(Dispatchers.IO) {
                    val bars = Quotes.ohlc(tk, months, force)
                    val p = DoubleArray(bars.size) { bars[it].close }
                    val closes = bars.map { it.t to it.close }
                    // 장중에는 분석용 일봉(6시간 캐시)의 오늘 봉이 낡아 있다 → 3봉만 따로 받아 덮는다.
                    // 장이 닫혀 있으면 마지막 봉이 이미 확정값이라 그대로 쓴다.
                    val mk = if (Tickers.isKrw(tk)) "KR" else "US"
                    val today = (if (MarketHours.labelFor(mk) != null) Quotes.todayCandle(tk) else null)
                        ?: bars.lastOrNull()
                    val m = p.size
                    val held = tossHeld?.contains(tk) ?: (Portfolio.currentHoldQty(trades[tk].orEmpty()) > 0)
                    val hasTrades = trades[tk].orEmpty().isNotEmpty()
                    val NA = Double.NaN

                    // 시세를 못 받았어도 보유 중이면 계좌가 알려주는 현재가로라도 목록에 남긴다.
                    // 조용히 빠지면 "내가 산 종목이 왜 안 보이지"가 된다.
                    if (m < 2) {
                        val hp = heldPrice[tk] ?: return@async null
                        return@async Row(
                            ticker = tk, name = Tickers.displayName(tk), price = hp, prevClose = hp,
                            day = 0.0, week = 0.0, fromHigh = 0.0,
                            zPct = NA, mPct = NA, signal = "", beta = NA, sigmaPct = NA,
                            holding = held, hasHistory = hasTrades,
                        )
                    }

                    // 상장한 지 얼마 안 된 종목은 회귀에 필요한 거래일이 모자라 analyze 가 null 이다.
                    // 그래도 가격·등락률은 보여줄 수 있으므로 행을 버리지 않고 Z·M 만 비운다.
                    val r = Quant.analyze(spy, closes)
                    // 등락률은 **원본 일봉**에서 계산한다. r.price 는 SPY 와 날짜 교집합을 낸 결과라
                    // 기준일이 직전 거래일이 아닐 수 있고, 그러면 "일" 열이 며칠치 변동으로 부풀려진다.
                    val prevD = p[m - 2]
                    val prevW = if (m > 5) p[m - 6] else prevD
                    val high = p.max()
                    Row(
                        ticker = tk, name = Tickers.displayName(tk), price = p[m - 1],
                        prevClose = prevD,
                        day = if (prevD > 0) (p[m - 1] / prevD - 1) * 100 else 0.0,
                        week = if (prevW > 0) (p[m - 1] / prevW - 1) * 100 else 0.0,
                        fromHigh = if (high > 0) (p[m - 1] / high - 1) * 100 else 0.0,
                        zPct = r?.lastZpct ?: NA, mPct = r?.lastMpct ?: NA, signal = r?.signal ?: "",
                        beta = r?.beta ?: NA, sigmaPct = r?.sigmaPct ?: NA, holding = held,
                        hasHistory = hasTrades,
                        open = today?.open ?: NA, high = today?.high ?: NA, low = today?.low ?: NA,
                    )
                }
            }.awaitAll().filterNotNull()
        }
        if (rows.isNotEmpty()) { slot.rows = rows; slot.ts = now; slot.key = curKey }
        return rows
    }
}
