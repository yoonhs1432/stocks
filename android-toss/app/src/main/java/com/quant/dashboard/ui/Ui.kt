package com.quant.dashboard.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quant.dashboard.ui.theme.Accent
import com.quant.dashboard.ui.theme.BgApp
import com.quant.dashboard.ui.theme.DividerColor
import com.quant.dashboard.ui.theme.Ghost
import com.quant.dashboard.ui.theme.OnAccent
import com.quant.dashboard.ui.theme.TextPrimary
import com.quant.dashboard.ui.theme.TextSecondary

/**
 * 전 탭 공통 UI 부품 — A-1 "토스 블루".
 *
 * 규격을 한 곳에 못 박아 두는 파일이다. 탭마다 버튼 모양·높이·색이 달라지던 걸 막는다.
 *  · 화면 좌우 여백 20dp, 헤더 52dp, 행 최소 44dp
 *  · 기본 버튼 = 파랑 채움 · 라운드 12 / 보조 버튼 = 고스트 · 라운드 10
 *  · 세그먼트 = 밑줄 탭 (활성 = 파랑 글씨 + 2dp 밑줄)
 */
val ScreenPad = 20.dp

/** 화면 상단 헤더 — 좌측 제목, 우측 액션. 아래 1px 구분선. */
@Composable
fun ScreenHeader(title: String, actions: @Composable RowScope.() -> Unit = {}) {
    Column {
        Row(
            Modifier.fillMaxWidth().height(52.dp).padding(horizontal = ScreenPad),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(title, color = TextPrimary, fontSize = 20.sp, fontWeight = FontWeight.ExtraBold,
                modifier = Modifier.weight(1f))
            actions()
        }
        HDivider()
    }
}

@Composable
fun HDivider(modifier: Modifier = Modifier) =
    Box(modifier.fillMaxWidth().height(1.dp).background(DividerColor))

/** 섹션 제목 — 작은 회색 글씨. 카드 대신 이걸로 묶는다. */
@Composable
fun SectionLabel(text: String, top: Dp = 18.dp) {
    Text(text, color = TextSecondary, fontSize = 12.sp, fontWeight = FontWeight.Bold,
        modifier = Modifier.padding(top = top, bottom = 6.dp))
}

/** 기본(주요) 버튼 — 파랑 채움, 풀폭. */
@Composable
fun PrimaryButton(label: String, enabled: Boolean = true, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Box(
        modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
            .background(if (enabled) Accent else Ghost)
            .clickable(enabled = enabled) { onClick() }
            .padding(vertical = 13.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(label, color = if (enabled) OnAccent else TextSecondary, fontSize = 15.sp,
            fontWeight = FontWeight.ExtraBold)
    }
}

/** 보조 버튼 — 고스트. 행 오른쪽의 작은 동작(복사·변경·삭제 등). */
@Composable
fun GhostButton(label: String, modifier: Modifier = Modifier, color: Color = TextPrimary,
                enabled: Boolean = true, onClick: () -> Unit) {
    Box(
        modifier.clip(RoundedCornerShape(10.dp)).background(Ghost)
            .clickable(enabled = enabled) { onClick() }
            .padding(horizontal = 12.dp, vertical = 6.dp),
    ) {
        Text(label, color = if (enabled) color else TextSecondary, fontSize = 13.sp,
            fontWeight = FontWeight.Bold)
    }
}

/**
 * 밑줄 세그먼트 — 선택 = 파랑 글씨 + 밑줄, 나머지 = 회색.
 * 시장 전환·차트 묶음·갱신 주기·원/달러 전부 이걸로.
 */
@Composable
fun UnderlineSegments(
    options: List<Pair<String, String>>,   // (id, label)
    selected: String,
    onSelect: (String) -> Unit,
    modifier: Modifier = Modifier,
    fontSize: Int = 13,
) {
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        options.forEach { (id, label) ->
            val on = id == selected
            // ⚠️ 폭을 글자에 맞춰 못 박아야 한다. 안 그러면 밑줄의 fillMaxWidth 가 Row 의 남은 폭을
            // 전부 가져가 첫 항목이 줄 전체를 삼키고 나머지 항목은 화면 밖으로 밀린다 (실제 발생).
            Column(
                Modifier.width(IntrinsicSize.Max).clickable { onSelect(id) }.padding(horizontal = 10.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(label, color = if (on) Accent else TextSecondary, fontSize = fontSize.sp,
                    fontWeight = if (on) FontWeight.ExtraBold else FontWeight.SemiBold,
                    modifier = Modifier.padding(vertical = 8.dp))
                Box(Modifier.fillMaxWidth().height(2.dp).background(if (on) Accent else Color.Transparent))
            }
        }
    }
}

/** 리스트 행 — 최소 44dp, 아래 구분선. 좌우 여백은 화면이 준다. */
@Composable
fun ListRow(
    modifier: Modifier = Modifier,
    divider: Boolean = true,
    minHeight: Dp = 44.dp,
    content: @Composable RowScope.() -> Unit,
) {
    Column(modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth().height(minHeight),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp),
            content = content,
        )
        if (divider) HDivider()
    }
}

/** 하단 고정 조작 바 — 탭바 바로 위. 배경은 화면과 같고 위에 구분선. */
@Composable
fun BottomActionBar(content: @Composable RowScope.() -> Unit) {
    Column {
        HDivider()
        Row(
            Modifier.fillMaxWidth().background(BgApp).height(48.dp).padding(horizontal = ScreenPad - 10.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = content,
        )
    }
}
