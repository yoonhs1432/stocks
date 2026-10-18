package com.quant.dashboard

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.os.Build
import com.quant.dashboard.data.AppUpdate

/**
 * 설치 결과 받기.
 *  - 확인이 필요하다(처음 한 번 등) → 화면이 보이면 바로 확인 창, 아니면 알림 → 누르면 확인 창
 *  - 내 앱이 새 버전으로 바뀌었다(MY_PACKAGE_REPLACED) → '업데이트 완료' 알림
 *    (자기 자신을 업데이트하면 프로세스가 죽어서 STATUS_SUCCESS 는 대개 못 받는다)
 */
class UpdateReceiver : BroadcastReceiver() {
    companion object {
        const val ACTION_STATUS = "com.quant.dashboard.UPDATE_STATUS"
    }

    override fun onReceive(ctx: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_MY_PACKAGE_REPLACED) {
            val notes = AppUpdate.lastNotes(ctx)
            AppUpdate.notify(ctx, "업데이트 완료 · ${BuildConfig.BUILD_STAMP}",
                notes.ifBlank { "새 버전으로 바뀌었습니다" })
            return
        }
        if (intent.action != ACTION_STATUS) return
        val stamp = intent.getStringExtra("stamp") ?: ""
        when (intent.getIntExtra(PackageInstaller.EXTRA_STATUS, PackageInstaller.STATUS_FAILURE)) {
            PackageInstaller.STATUS_PENDING_USER_ACTION -> {
                val confirm = if (Build.VERSION.SDK_INT >= 33) {
                    intent.getParcelableExtra(Intent.EXTRA_INTENT, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION") intent.getParcelableExtra(Intent.EXTRA_INTENT)
                } ?: return
                confirm.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                // 뒤에서 확인 창을 띄우는 건 막혀 있다 — 화면이 보일 때만 바로, 아니면 알림으로
                val shown = AppUpdate.foreground && runCatching { ctx.startActivity(confirm) }.isSuccess
                if (!shown) {
                    AppUpdate.notify(ctx, "새 버전 $stamp", "눌러서 설치하세요", confirm)
                }
            }
            PackageInstaller.STATUS_SUCCESS -> {
                AppUpdate.notify(ctx, "업데이트 완료 · $stamp", AppUpdate.lastNotes(ctx).ifBlank { "새 버전으로 바뀌었습니다" })
            }
            PackageInstaller.STATUS_FAILURE_ABORTED -> Unit   // 사용자가 취소 — 6시간 뒤 다시 묻는다
            else -> {
                val msg = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE) ?: "알 수 없는 오류"
                AppUpdate.notify(ctx, "업데이트 실패 · $stamp", msg)
            }
        }
    }
}
