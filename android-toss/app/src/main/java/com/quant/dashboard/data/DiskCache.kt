package com.quant.dashboard.data

/**
 * 서버 응답 **원문**을 파일로 남겨 두는 곳 — 앱을 껐다 켜도 첫 화면이 비지 않게.
 *
 * 비교 표는 이미 이렇게 하고 있었다(`OverviewRepo.restore`). 계좌·자산 추이·분석도 같은
 * 대접을 받게 하려고 공통으로 뺐다. 원문을 그대로 두므로 읽을 때는 평소와 같은 파서를 탄다
 * — 캐시용 직렬화 코드를 따로 두면 필드가 늘 때마다 두 군데를 고쳐야 한다.
 *
 * 값 자체는 곧 새로 받아 덮어쓴다. 여기 있는 건 "그동안 보여 줄 것"일 뿐이다.
 */
object DiskCache {

    private fun fileOf(key: String) =
        Store.fileIn("resp_" + key.replace(Regex("[^A-Za-z0-9_.-]"), "_") + ".json")

    fun put(key: String, text: String) {
        try { fileOf(key)?.writeText(text) } catch (e: Exception) { /* 못 남겨도 화면엔 영향 없다 */ }
    }

    /** (원문, 남긴 시각 ms). 없거나 못 읽으면 null. */
    fun get(key: String): Pair<String, Long>? {
        val f = fileOf(key) ?: return null
        return try {
            if (!f.exists()) null else f.readText() to f.lastModified()
        } catch (e: Exception) {
            null
        }
    }

    /** [prefix] 로 시작하는 파일을 최근 [keep] 개만 남긴다 — 종목별 분석처럼 계속 늘어나는 것용. */
    fun prune(prefix: String, keep: Int) {
        try {
            val dir = Store.fileIn("x")?.parentFile ?: return
            val mine = dir.listFiles { f -> f.name.startsWith("resp_$prefix") } ?: return
            if (mine.size <= keep) return
            mine.sortedByDescending { it.lastModified() }.drop(keep).forEach { it.delete() }
        } catch (e: Exception) {
            // 정리를 못 해도 동작에는 지장이 없다
        }
    }
}
