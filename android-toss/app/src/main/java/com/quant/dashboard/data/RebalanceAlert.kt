package com.quant.dashboard.data

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
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
import com.quant.dashboard.MainActivity
import java.util.concurrent.TimeUnit

/**
 * 리밸런싱 알림 — 하루 몇 번 PC 에 물어보고
 *  ① 기간(분기 등)이 끝나 '맞출 때'가 되면 한 번
 *  ② 그 사이 보유 종목이 목표의 2배/절반을 넘으면 한 번(같은 종목 묶음은 7일에 한 번)
 * 알린다. PC 에 닿지 않으면 조용히 다음 차례를 기다린다.
 */
object RebalanceAlert {
    private const val WORK = "rebalance-alert"
    private const val CHANNEL = "rebalance"
    private const val ID = 7302

    fun schedule(ctx: Context) {
        val req = PeriodicWorkRequestBuilder<Worker>(6, TimeUnit.HOURS)
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .build()
        WorkManager.getInstance(ctx).enqueueUniquePeriodicWork(WORK, ExistingPeriodicWorkPolicy.KEEP, req)
    }

    class Worker(ctx: Context, p: WorkerParameters) : CoroutineWorker(ctx, p) {
        override suspend fun doWork(): Result {
            runCatching { check(applicationContext) }
            return Result.success()
        }
    }

    private fun check(ctx: Context) {
        if (!Server.isSet()) return
        Store.init(ctx); ServerConfig.init(ctx)
        val p = Server.rebalance()
        val pref = ctx.getSharedPreferences("rebalance", Context.MODE_PRIVATE)
        val label = mapOf("monthly" to "이번 달", "quarterly" to "이번 분기", "yearly" to "올해")[p.period] ?: "이번 기간"
        if (p.due && pref.getString("dueFor", "") != p.periodEnd) {
            val n = p.rows.count { it.action != "none" }
            notify(ctx, "리밸런싱 할 때", "${label} 첫 거래일 — 오늘 미국장에서 맞출 종목 ${n}개")
            pref.edit().putString("dueFor", p.periodEnd).apply()
            return
        }
        if (p.alerts.isNotEmpty()) {
            val key = p.alerts.sorted().joinToString(",")
            val now = System.currentTimeMillis()
            if (pref.getString("alertKey", "") != key || now - pref.getLong("alertAt", 0) > 7 * 86_400_000L) {
                notify(ctx, "목표에서 많이 벗어났습니다",
                    p.alerts.joinToString(", ") { Tickers.displayName(it) } + " — 목표의 2배 또는 절반을 넘었습니다")
                pref.edit().putString("alertKey", key).putLong("alertAt", now).apply()
            }
        }
    }

    private fun notify(ctx: Context, title: String, text: String) {
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) return
        val nm = ctx.getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "리밸런싱", NotificationManager.IMPORTANCE_DEFAULT))
        }
        val pi = PendingIntent.getActivity(ctx, 1, Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        nm.notify(ID, NotificationCompat.Builder(ctx, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_popup_reminder)
            .setContentTitle(title).setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setContentIntent(pi).setAutoCancel(true).build())
    }
}
