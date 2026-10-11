package com.quant.dashboard.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext

/**
 * 종목별 분석을 **받아 둔다** — 종목을 바꿀 때 스피너 대신 차트가 먼저 보이게.
 *
 * 예전에는 종목을 누를 때마다 화면을 비우고 서버를 기다렸다. 서버가 계산을 미리 끝내 두어도
 * 왕복 시간(터널이면 0.3~1초)만큼은 빈 화면이었다. 지금은
 *  1. 메모리에 최근 본 종목들을 들고 있고(즉시),
 *  2. 파일에도 남겨 두며(앱을 새로 켠 직후에도 수십 ms),
 *  3. 비교 표가 뜨면 **누르기 전에** 보유·관심 종목을 조용히 받아 둔다([prefetch]).
 *
 * 그 종목 **자신의** 지난 결과만 보여 주므로, 예전에 겪은 "이전 종목 차트가 남아 보이는"
 * 문제와는 다르다. 보여 준 뒤에는 늘 새로 받아 조용히 갈아 끼운다.
 */
object AnalysisCache {

    data class Entry(val a: Server.Analysis, val at: Long)

    private const val MEM = 32          // 종목당 수십 KB — 서른 개 남짓이면 1~2MB
    private const val DISK = 60
    /** 이보다 새 것이면 미리 받기를 건너뛴다. 일봉 분석은 서버에서도 5분에 한 번 바뀐다. */
    private const val FRESH = 5 * 60 * 1000L

    // accessOrder=true — 꺼내 볼 때마다 뒤로 가서, 넘치면 가장 오래 안 본 것부터 빠진다
    private val mem = object : LinkedHashMap<String, Entry>(MEM, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Entry>?) = size > MEM
    }

    /** 메모리에 있는 것만 — 화면 그리는 중(메인 스레드)에 불러도 된다. */
    @Synchronized
    fun peek(ticker: String): Entry? = mem[ticker]

    @Synchronized
    private fun remember(ticker: String, e: Entry) { mem[ticker] = e }

    /** 메모리 → 파일 순으로 찾는다. 파일을 읽을 수 있으므로 **IO 에서** 부를 것. */
    fun get(ticker: String): Entry? {
        peek(ticker)?.let { return it }
        val (a, at) = Server.analysisCached(ticker) ?: return null
        if (a.result == null) return null
        return Entry(a, at).also { remember(ticker, it) }
    }

    /** 서버에서 새로 받아 캐시에 넣는다. IO 에서 부를 것. 실패하면 예외를 그대로 던진다. */
    fun fetch(ticker: String, force: Boolean = false): Server.Analysis {
        val a = Server.analysis(ticker, force)
        if (a.result != null) remember(ticker, Entry(a, System.currentTimeMillis()))
        return a
    }

    @Synchronized
    fun clear() { mem.clear() }

    /**
     * 누르기 전에 받아 둔다. **하나씩, 사이를 두고** — 한꺼번에 쏘면 터널과 PC 가 버겁고,
     * 그 사이 사용자가 누른 진짜 요청이 줄 뒤에 선다.
     *
     * 연달아 두 번 실패하면 그만둔다(끊긴 상태에서 스무 번 헛돌 이유가 없다).
     */
    suspend fun prefetch(tickers: List<String>) = withContext(Dispatchers.IO) {
        if (!ServerConfig.isSet()) return@withContext
        var fails = 0
        val now = System.currentTimeMillis()
        for (tk in tickers.distinct()) {
            val have = runCatching { get(tk) }.getOrNull()
            if (have != null && now - have.at < FRESH) continue
            try {
                fetch(tk)
                fails = 0
            } catch (e: Exception) {
                if (++fails >= 2) break
            }
            delay(250)
        }
        DiskCache.prune(Server.ANALYSIS_KEY, DISK)
    }
}
