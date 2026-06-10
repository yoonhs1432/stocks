# SESSION_NOTES — 최근 작업 핸드오프

> 새 세션 시작 시 이 파일을 읽으면 직전 세션의 맥락을 이어받을 수 있음.
> 마지막 업데이트: 2026-06-10 (Fable 세션 — 전체 코드 리뷰 반영, 브랜치 claude/code-review-v5-0-fge8s1)

## 2026-06-10 코드 리뷰 반영 세션 (app.py 6,040 → ~5,600줄)

### 버그 수정
- **ET 시간대**: UTC-4 고정 → `zoneinfo.ZoneInfo("America/New_York")` (겨울철 1시간 오차 해소)
- **NYSE 휴장일**: Good Friday(부활절 알고리즘 `_easter_date`)·Juneteenth(2022~)·대체휴일(토→금, 일→월, 단 1/1 토요일은 미관측) 추가
- **매매 기록 저장 검증**: 수량/단가 0이면 저장 차단 + 에러 표시 (이전엔 저장돼도 모든 통계에서 invisible)
- **색상 체계 일원화**: `momentum_to_color`(7단계 정수) 삭제 → 전 화면 `momentum_pct_to_color`(5단계 백분위 20/40/60/80)로 통일. 버튼·종목명 헤더·Z/M 수치·탭2 표/산점도·매매일지 모두 동일
- **hover 복원**: 탭1 `fig.update_traces(hoverinfo='skip')` 블랭킷 제거 (trace별 skip 유지), `hovermode='closest'` — Z·M 산점도 날짜 hover 동작. 탭2 산점도·탭3 자산추이 `staticPlot` 해제 (축 `fixedrange`+`dragmode=False`로 모바일 스크롤 보호)
- `pnl_color(0)` → 중립 회색
- 차트 매직 row 번호(`row=4`, `row=5`) → `ROW = {name: idx}` 매핑 (패널 순서 변경에 안전)

### 액션 카드 복원 (핵심)
- `build_action_card_html`이 만들어 놓고 `return bar_html`만 하던 버그 → 위치바+현재위치+액션카드+서브액션 전부 반환
- 탭1 차트 위에 호출 연결 (다음 매수/익절 트리거 가격 + 몇 % 남았는지)
- 카드 내부 셀 배경 `var(--bg-subtle)`(다크) → `#ffffff` (다크 텍스트 가독성)

### 죽은 코드 제거 (~700줄)
- 인증 일체 (쿠키 HMAC/비밀번호 해시/`verify_password`) — `is_authenticated()`(항상 True)만 유지. `extra-streamlit-components`·`scikit-learn` requirements에서 제거
- `build_mini_gradient_bar`, `compute_halflife`/`halflife_color`, `pct_to_label`, `html_progress_bar`, `SIGNAL_STYLE`/`BUTTON_TEXT_STYLE`/`SIG_MARKER`/`SIGNAL_PRIORITY`, `_hline_g1` 블록, position tracker의 미사용 trend/pnl/cumulative/period HTML, 달력 CSS(`ov_cal_nav`), `_macd_marker_extra` session_state 임시변수(→로컬), flake8 F841 잔재 일괄
- `DEFAULT_TICKERS` 중복 정의 통합 (섹션 1에서만 정의)

### UX 개선
- **MDD 표시**: 계산만 하던 `dd_info_cache`를 탭3 손익 종합 카드에 노출 (고점 대비 현재 DD · MDD + 날짜, `dd_color`)
- **지표 설명 expander**: 탭1 분석 패널 상단 "ℹ️ 지표 설명 (σ·β·Z·M)" — 모바일에서 title 툴팁 안 되는 문제 보완
- **새로고침 전역화**: `full_data_refresh()` 헬퍼 추출, 탭2/탭3 상단에 `render_refresh_row()` 추가 (CSS `st-key-refresh_row_*`)
- **기준일 해제 버튼**: 기준일 시뮬레이션 활성 시 헤더 아래 원클릭 해제 (설정 탭 안 가도 됨)
- **스피너 통합**: 데이터 로드~분석 스피너 1개로 (연쇄 깜빡임 제거)
- 최소 폰트 0.55/0.58rem → 0.62rem
- 중립색 `MOM_HOLD` #9ca3af → #6b7280 (약매수 연빨강과 구분, 버튼 흰 글씨)
- 탭2 표 캡션: 정렬은 selectbox 기준임을 명시

### 주의
- `momentum_to_color`는 삭제됨 — M 색은 반드시 `momentum_pct_to_color(z_to_pct(smooth))` 경유
- `ticker_momentum_scores`(정수)는 **정렬용으로만** 사용 (색 아님)
- requirements `streamlit>=1.39` (st.container key 파라미터 의존)
- 로그인 복원 시: 비밀번호 해시를 코드에 두지 말 것 (이전 fallback 해시는 오프라인 사전공격 가능해서 제거함)

---

## (이전 세션) 2026-06-10 Opus 세션 인계 내용

## 이번 세션에서 구현된 주요 기능

### 1. 기준일(As-of) 시뮬레이션 모드
- 설정 탭에 "📅 기준일 시뮬레이션" 체크박스 + 날짜 입력
- 기준일 설정 시 (분석시작일~오늘) 기간 길이 유지한 채 종료일을 기준일로 당겨 과거 시점 재현
- `main()`에서 `df_close`를 기준일까지 슬라이싱 (재fetch 없음), `last_trading_date = asof_end_date`
- 헤더에 노란 기준일 배지 표시
- regime/거시 지표 배지도 기준일과 연동

### 2. 시장 체제(Regime) + 거시 지표 헤더 배지
- `get_market_regime()`: SPY SMA200 + 6M 수익률 → 🟢강세/🔴약세/🟠조정/⚪중립 (1h 캐시)
- `get_macro_indicators()`: VIX·US 10Y(^TNX 우선, >50이면 /10 정규화)·USD/KRW (1h 캐시)
- VIX 색: <15 녹 / 15-20 회 / 20-30 주황 / ≥30 빨강. 10Y: <3 녹 / 3-4 회 / 4-5 주황 / ≥5 빨강
- 배경: 하락장 취약성 분석에서 나옴. 미적용 옵션 — A(SMA200 추세 게이트로 매수신호 채도 약화),
  C(rolling 회귀로 Z 베타 stale 해결), E(약세장 시 M 가중 재조정)

### 3. M(모멘텀) 공식 개편 — 후행성 해소
- 문제: 레버리지 ETF에서 MACD_Pct 높이 항이 고정 임계(±2%)를 수 배 초과해 M 지배 (M86 vs RSI 43 사례)
- 해결: **변동성 적응 정규화** — 높이/변곡을 자기 역사 rolling std(120일)로 z화
- 현재 공식 (`compute_momentum_score_smooth`, `compute_momentum_series`):
  ```
  h = clip( MACD_Pct / (1.5σ), ±1 )      # M_SIGMA_SCALE=1.5
  d = clip( dMACD_Pct / (1.5σ), ±1 )
  r = clip( (RSI-50) / 30, ±1 )           # M_RSI_SCALE=30 (RSI 80/20에서 ±1)
  M_smooth = 2.5 × (0.30h + 0.15d + 0.55r)  # 가중치 Config: M_W_HEIGHT/INFLECT/RSI
  ```
- `process_asset_data`에 `MACD_Pct_Std`/`dMACD_Pct_Std` 컬럼 추가, 분석 캐시 `_version=9`
- 미세조정 포인트: Config의 `M_SIGMA_SCALE`, `M_RSI_SCALE` (감도), `M_W_*` (가중치)

### 4. 임계 체계 통일 — 전 화면 20/40/60/80 (5단계)
- `momentum_pct_to_color`: <20 강매수(빨강) / 20-40 약매수 / 40-60 중립 / 60-80 약매도 / ≥80 강매도(파랑)
- 종목 버튼 색, 그래프2(Z·M 산점도) 흰 점선, 그래프4(Z+M) 흰 점선, 종목비교 표 Z·M 색 모두 동일 기준
- 그래프4 빨강 면적: Z > 80
- 종목비교 표 Z 색 방향: 낮음=빨강(매수), 높음=파랑(매도) — M과 동일 방향

### 5. 한국 종목 지원
- `_korean_stock_names()`: `fdr.StockListing('KRX')` 코드→이름 매핑 (24h 캐시), 6자리 숫자 티커 자동 변환
- 종목명 사용자 변경: 설정→종목 관리 각 행에 이름 입력란, `display_name_overrides`로 settings(Gist) 저장
- `display_name` 우선순위: 사용자 override → TICKER_DISPLAY_NAMES 하드코딩 → KRX 자동 → 티커

### 6. 버그 수정
- **장중 가격 고정 버그**: `extra_close_cache`(SPY·매매이력 종목, session_state)에 TTL 없었음
  → 5분 TTL 추가 + refresh 버튼이 session_state 캐시(`extra_close_cache`, `fetch_blacklist`,
  `df_close_last` 등)도 클리어하도록 수정
- **그래프3 흰 종가선이 캔들 위에 보임**: Plotly는 scatterlayer가 boxlayer(캔들) 위에 렌더되므로
  trace 순서로 해결 불가 → `zorder=-1` 사용 (requirements.txt에 `plotly>=5.21` 명시), 굵기 0.8

### 7. 기본값 변경
- 자산 추이 단위: 일 → **주**
- 분석 시작일: 1년 → **2년** (730일)

## 주의사항 / 컨벤션 (이 세션에서 확립)
- M 공식 수정 시 반드시 `compute_momentum_score_smooth`(스칼라)와 `compute_momentum_series`(벡터)
  **둘 다** 수정 — 버튼/표/차트가 나뉘어 사용
- `process_asset_data` 출력 컬럼 변경 시 `compute_all_analyses` 호출부의 `_version` bump 필요
- session_state 기반 캐시는 `st.cache_data.clear()`로 안 지워짐 — refresh 버튼 로직 참고
- Plotly 캔들+라인 z-order는 `zorder` 속성으로만 제어 가능

## 미적용 아이디어 (후속 후보)
- 추세 게이트: 종가 < SMA200이면 매수(빨강) 버튼 채도 약화 (하락장 보호)
- Rolling 회귀(252일)로 Z 베타 stale 해결 (현재 full-sample polyfit)
- 약세장 시 M 가중 자동 재조정
- 곡선 역전(10Y-2Y) 경고 배지
