package com.quant.dashboard.data

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.quant.dashboard.BuildConfig
import com.quant.dashboard.MainActivity
import com.quant.dashboard.UpdateReceiver
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.TimeUnit

/**
 * **스스로 업데이트** — APK 를 손으로 받지 않아도 되게.
 *
 * 흐름: GitHub 릴리스(`android-latest`)의 `version.json` 을 보고 → 내 번호보다 크면 APK 를 받아
 * → `PackageInstaller` 로 설치를 건다. 저장소가 공개라 PC 가 꺼져 있어도 된다.
 *
 * 안드로이드 정책상 **처음 한 번은 사용자가 '설치'를 눌러야 한다**(알림으로 뜬다). 그 뒤로는
 * 이 앱이 '설치한 앱'으로 기록돼, 12 이상에서는 확인 없이 깔린다(`USER_ACTION_NOT_REQUIRED`).
 * 확인이 또 필요하면 시스템이 알려 주고, 그때도 알림을 띄운다.
 */
object AppUpdate {
    private const val BASE = "https://github.com/yoonhs1432/stocks/releases/download/android-latest/"
    private const val APK = "quant-toss-debug.apk"
    private const val CHANNEL = "update"
    private const val WORK = "app-update"
    const val NOTIFY_ID = 7301

    /** 화면이 보이는 중인가 — 이때는 설치 확인 창을 알림 없이 바로 띄운다. MainActivity 가 갱신. */
    @Volatile var foreground = false

    data class Remote(val code: Int, val stamp: String, val notes: String)

    /** 1시간마다 확인. 이미 걸려 있으면 그대로 둔다. */
    fun schedule(ctx: Context) {
        val req = PeriodicWorkRequestBuilder<Worker>(1, TimeUnit.HOURS)
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .build()
        WorkManager.getInstance(ctx).enqueueUniquePeriodicWork(WORK, ExistingPeriodicWorkPolicy.KEEP, req)
    }

    class Worker(ctx: Context, p: WorkerParameters) : CoroutineWorker(ctx, p) {
        override suspend fun doWork(): Result {
            runCatching { run(applicationContext, user = false) }
            return Result.success()
        }
    }

    private fun get(url: String): ByteArray {
        // github.com → objects.githubusercontent.com 으로 넘겨준다. 둘 다 https 라 자동으로 따라간다.
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 15_000
        c.readTimeout = 60_000
        c.useCaches = false
        try {
            if (c.responseCode != 200) throw IllegalStateException("HTTP ${c.responseCode}")
            return c.inputStream.use { it.readBytes() }
        } finally {
            c.disconnect()
        }
    }

    fun remote(): Remote? = runCatching {
        val o = JSONObject(String(get(BASE + "version.json" + "?t=" + System.currentTimeMillis())))
        Remote(o.optInt("versionCode"), o.optString("stamp"), o.optString("notes"))
    }.getOrNull()

    /**
     * 확인 → 받기 → 설치. 화면에 보여 줄 한 줄을 돌려준다.
     * @param user 사용자가 '업데이트 확인'을 눌렀다 — 최근에 시도한 번호라도 다시 한다.
     */
    fun run(ctx: Context, user: Boolean): String {
        val r = remote() ?: return "업데이트 확인 실패 — 인터넷 연결을 확인하세요"
        if (r.code <= BuildConfig.VERSION_CODE) return "최신 버전입니다 (${BuildConfig.BUILD_STAMP})"
        // 알림을 무시하고 있는데 한 시간마다 다시 받고 또 알리면 귀찮다 — 같은 번호는 6시간 쉰다
        val p = ctx.getSharedPreferences("update", Context.MODE_PRIVATE)
        val now = System.currentTimeMillis()
        if (!user && p.getInt("tried", 0) == r.code && now - p.getLong("at", 0) < 6 * 3600_000L) {
            return "새 버전 ${r.stamp} 설치 대기 중"
        }
        p.edit().putInt("tried", r.code).putLong("at", now).putString("notes", r.notes).apply()

        val f = File(ctx.cacheDir, "update.apk")
        val tmp = File(ctx.cacheDir, "update.apk.part")
        tmp.writeBytes(get(BASE + APK + "?t=" + now))
        // 릴리스가 version.json 만 먼저 올라가고 APK 는 아직 옛것일 수 있다 — 받은 APK 의 번호를 확인
        val info = ctx.packageManager.getPackageArchiveInfo(tmp.path, 0)
        val code = info?.let { if (Build.VERSION.SDK_INT >= 28) it.longVersionCode else @Suppress("DEPRECATION") it.versionCode.toLong() }
        if (info?.packageName != ctx.packageName || code == null || code <= BuildConfig.VERSION_CODE) {
            tmp.delete()
            p.edit().remove("tried").apply()
            return "새 APK 가 아직 올라가는 중입니다 — 잠시 뒤 다시"
        }
        f.delete()
        tmp.renameTo(f)
        install(ctx, f, r)
        return "새 버전 ${r.stamp} 설치 중…"
    }

    private fun install(ctx: Context, f: File, r: Remote) {
        val pi = ctx.packageManager.packageInstaller
        val params = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL).apply {
            setAppPackageName(ctx.packageName)
            if (Build.VERSION.SDK_INT >= 31) {
                setRequireUserAction(PackageInstaller.SessionParams.USER_ACTION_NOT_REQUIRED)
            }
        }
        val id = pi.createSession(params)
        pi.openSession(id).use { s ->
            s.openWrite("app.apk", 0, f.length()).use { out ->
                f.inputStream().use { it.copyTo(out) }
                s.fsync(out)
            }
            val i = Intent(ctx, UpdateReceiver::class.java)
                .setAction(UpdateReceiver.ACTION_STATUS)
                .putExtra("stamp", r.stamp)
            // 시스템이 결과를 엑스트라로 채워 넣으므로 MUTABLE 이어야 한다
            val flags = PendingIntent.FLAG_UPDATE_CURRENT or
                (if (Build.VERSION.SDK_INT >= 31) PendingIntent.FLAG_MUTABLE else 0)
            s.commit(PendingIntent.getBroadcast(ctx, id, i, flags).intentSender)
        }
    }

    /** 알림 한 장. 권한이 없으면(13+ 에서 거절) 조용히 넘어간다. */
    fun notify(ctx: Context, title: String, text: String, tap: Intent? = null) {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) return
        val nm = ctx.getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL, "앱 업데이트", NotificationManager.IMPORTANCE_HIGH))
        }
        val target = tap ?: Intent(ctx, MainActivity::class.java)
        val pi = PendingIntent.getActivity(ctx, 0, target.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val n = NotificationCompat.Builder(ctx, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_download_done)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(pi)
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .build()
        nm.notify(NOTIFY_ID, n)
    }

    /** 마지막으로 설치한 업데이트의 노트 — 완료 알림에 쓴다. */
    fun lastNotes(ctx: Context): String =
        ctx.getSharedPreferences("update", Context.MODE_PRIVATE).getString("notes", "") ?: ""
}
