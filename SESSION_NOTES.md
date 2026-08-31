# SESSION_NOTES — 최근 작업 핸드오프

> 새 세션 시작 시 이 파일을 읽으면 직전 세션의 맥락을 이어받을 수 있음.
> 마지막 업데이트: 2026-08-07 (Android 차트 확대/이동 · 종목 전환 버그 수정, 브랜치 claude/fix-needed-tm9ina)

## 2026-08-07 Android UX 수정 세션

> 선행: `claude/android-native`(87커밋)를 **PR #3으로 main에 병합**. 이제 main이 Android 앱까지 포함.
> `app.py`는 변경 없음(Android 브랜치가 손댄 기존 파일은 SESSION_NOTES/.gitignore뿐).

### 1. 종목 전환 시 이전 종목이 남던 버그 (AnalysisViewModel/AnalysisScreen)
- 원인: `LaunchedEffect(dataVersion)`의 `sync()`와 `LaunchedEffect(pendingTicker)`의 `select()`가
  **각각 load()를 걸어 두 네트워크 요청이 경쟁** → 늦게 끝난 이전 종목 응답이 새 상태를 덮어씀
- 해결:
  - 두 효과를 하나로 합침 → `vm.sync(version, pending)` 단일 진입점
  - `loadJob?.cancel()` + `reqSeq` 시퀀스 가드로 오래된 응답 폐기
  - `select()`에서 종목이 바뀌면 `result`/`ohlc`를 즉시 비움 (로딩 중 이전 차트 노출 차단)
  - `autoRefresh()`는 `state.loading`이면 skip, 조용한 실패 시 `loading=false`로 해제
- ⚠️ `refresh()` 삭제됨 — 전체 새로고침은 `AppState.bump()` → `sync(changed)` 경로로 일원화

### 2. 확대 차트 축 확대/이동 + 현재값 십자선 + 축 눈금 (Charts.kt)

> ⚠️ **좌표계 규칙**: 제스처와 차트는 **반드시 같은 기준 폭(plotW = 캔버스폭 − AXIS_W)** 을 써야 함.
> 한쪽이 캔버스 폭을 쓰면 배율이 커질수록 확대 중심이 어긋나 그림이 조금씩 밀린다(실제 발생한 버그).
> `chartGestures`의 `axisPx`(=13sp×3.6)와 각 차트의 `AXIS_W`는 같은 값이어야 함 — 한쪽만 바꾸지 말 것.
>
> 손가락 수가 바뀌는 프레임(2번째 손가락 얹기/떼기)은 무게중심이 순간이동하므로
> `jumped` 플래그로 이동 처리에서 제외 (안 하면 핀치 시작 순간 확 밀림).
>
> **시계열 4종(가격·일봉/Z·M/MACD/RSI)은 x축 전용**(`xOnly=true`), y는 `visibleRange`로 보이는
> 구간에 자동 맞춤. Z·M/RSI는 `view.sx > 1f`일 때만 자동맞춤 — 원본 배율에서는 0~100 고정
> (20/40/60/80 임계선 의미 유지). 산점도 2종만 x·y 양축 확대.
- **`ChartView(sx, nx, sy, ny)`**: x·y 배율과 정규화 이동량. `n`을 `[-(s-1), 0]`으로 클램프해
  콘텐츠가 항상 화면을 채움. 최대 12배(`MAX_ZOOM`)
- **`Modifier.chartGestures(view, onChange, onTap)`**: 두 손가락 **가로** 벌리기=X축,
  **세로** 벌리기=Y축 (각 축 독립 — Compose 기본 `detectTransformGestures`는 단일 zoom이라 미사용).
  확대 상태에서 한 손가락 드래그=이동. 이동 없이 떼면 `onTap`(기존 재탭 닫기 동작 유지)
- 6개 차트 전부 `view`/`showCross` 파라미터 추가. 좌표는 `view.x(base, size.width)` 경유
- 현재값 표시: 시계열은 `currentCross`(시점 세로선 + 계열별 가로선/값 태그),
  산점도는 기존 `crosshair`(그동안 죽어 있던 함수 부활) + ★ 유지
- `DateAxis(dates, view)`: 확대 시 **보이는 인덱스 구간**의 날짜로 갱신, 3배 이상이면 yy/MM/dd
- 확대 다이얼로그 헤더에 배율(`⟲ 2.0×1.0`) 겸 원본 복귀 버튼

### 3. 비교 탭 "미장 TOP 30" 목록
- `Tickers.US_TOP30` — 미국 시총 상위 30개 **정적 목록**. 온디바이스라 시총 랭킹 소스가 없어 하드코딩
  (기준 2026 상반기). 순위가 바뀌면 이 목록만 갱신
- `OverviewRepo`를 목록별 캐시 슬롯(`watchSlot`/`topSlot`)으로 리팩터 — `load()`=워치리스트,
  `loadTop()`=TOP30. 분석 로직은 동일, 티커 소스만 다름
- TOP30은 **버튼을 처음 누를 때만** 로드 (요청 30건). 전환 시 보유 필터 해제 + 기본 정렬 `SortKey.RANK`(시총 순)
- 표에 시총 순위 숫자 표시 — 다른 기준으로 정렬해도 원래 순위를 알 수 있게
- `AnalysisViewModel.loadOverview`가 `cachedTop()`을 합쳐, 워치리스트 밖 종목으로 넘어가도
  헤더 일간 등락이 보임(추가 요청 없음)

### 4. 포트폴리오 → 분석 이동
- 평가금액 카드의 보유 종목 행 탭 → `onOpenAnalysis(h.ticker)` → 분석 탭 (비교 탭과 동일 경로)
- `AppState.pendingTicker`를 비교/포트폴리오 공용으로 사용

### 주의
- 차트 좌표 함수를 새로 만들 때는 반드시 `view.x()/view.y()`를 거칠 것 (안 그러면 확대 시 어긋남)
- `.github/workflows/android.yml` 릴리스 게시 조건을 `refs/heads/main`으로 변경
  (android-native 병합에 맞춤). APK는 **main에서 수동 실행**해야 `android-latest` 릴리스에 올라감

---

## 토스 API 미사용 엔드포인트 — 추가 기능 후보 (2026-08-29 정리, **미장 전용**)

> OpenAPI v1.2.14 전수 검토. **구현 완료**: prices · candles · holdings · orders(CLOSED) ·
> accounts · buying-power · exchange-rate · market-calendar · stocks · trades(진단용).
>
> ⚠️ **범위 (2026-08-30 갱신): 시세·랭킹·종목검색은 미국·국내 둘 다.**
> 2026-08-29 에는 미장 전용으로 좁혔으나, 미국주식 타사 출고가 막혀 국내 종목을 계속 보게 되면서
> 랭킹(`marketCountry=KR`)과 유니버스(KOSPI·KOSDAQ)를 다시 열었다.
> 여전히 **안 쓰는 것** — 투자자별 매매동향 / 공매도 / 신용거래 / 프로그램매매 · 대차잔고
> (KR 전용 수급 지표), market-indicators(KOSPI·KOSDAQ·국고채), KRX 투자자별 매매대금.
> **다시 살릴 일이 생기면 git 이력 `b1380a1` 에 원문이 있다.**

### 조회 가능 기간 — 2년 분석 그대로 가능 (오해 정정)

- 랭킹(`/rankings`)의 `duration` 이 최대 `1y` 인 것은 **"어느 기간 기준으로 순위를 매길지"**
  일 뿐, 과거 시세 조회 한도가 아니다.
- 일봉(`/api/v1/candles`)은 요청당 200봉 상한에 `before`/`nextBefore` 커서 페이징이라
  **더 과거로 계속 거슬러 올라간다.** 스펙에 기간 상한 명시 없음.
- 앱은 이미 `Quotes.barCount("2y") = 520봉`을 요청하고 `TossApi.dailyOhlc` 가 3페이지로
  나눠 받는다 → **2년 분석은 지금 코드 그대로 동작.** (`guard < 10` 이므로 최대 2000봉 ≈ 8년)
- 실제 상장 후 데이터가 짧으면 그만큼만 온다. 신규 상장 종목은 봉 수 부족으로 분석 실패 가능.

### A. ✅ 구현 완료 (2026-08-29)

**A1. 미장 TOP 목록 = 토스 랭킹** — `GET /api/v1/rankings` → `data/Rankings.kt`
- 비교 탭 상단에 기준 칩: 거래대금 · 거래량 · 토스대금 · 토스수량 · 급상승 · 급하락
  × 기간(실시간 · 1일 · 1주 · 1개월 · 3개월 · 6개월 · 1년). 설정은 `Store.rankType/rankDuration` 에 저장.
- **토스에 시총 랭킹이 없다** → 거래대금 상위(1일)가 대형주 목록에 가장 가까워 기본값.
- `TOP_GAINERS`/`TOP_LOSERS` 는 `realtime` 미지원(400 `unsupported-ranking-duration`) →
  급상승·급하락을 고르면 기간을 1일로 자동 보정하고 칩 목록에서 실시간을 뺀다.
- 폴백: 미연동 · 조회 실패 · 빈 배열(집계 없음)이면 `Tickers.US_TOP30` 을 쓰고
  사유를 `Rankings.fallbackReason` → 화면 ⚠️ 배지로 표시. **비교 탭이 비는 일은 없다.**
- 랭킹 기준이 바뀌면 종목 자체가 달라지므로 `OverviewRepo` 캐시 키에 `Rankings.cacheKey()` 를 포함시켰다.
  (안 넣으면 기준을 바꿔도 5분간 이전 종목이 남는다)
- 랭킹 자체는 10분 캐시. 응답 수 < count 일 수 있음(시세 조회 실패 종목 제외).

**A2. 종목 이름 검색** — `GET /api/v1/stocks/all` → `data/Universe.kt`
- NASDAQ · NYSE · AMEX 를 하루 1회 받아 `toss_universe.json` 에 캐시(파일 IO·파싱 모두 IO 디스패처).
- **토스는 미국 종목도 한글 `name` 을 준다** (AAPL → "애플") → 한글·영문·티커 아무거나 쳐도 찾힌다.
- 설정 → 종목 관리에서 입력하면 후보가 뜨고 탭하면 바로 추가. 이미 있는 종목은 "추가됨" 비활성.
- 정확도 순: 티커 완전일치 → 티커 접두 → 이름 접두 → 이름 포함 → 티커 포함.
  보통주·ETF 우선, 우선주·워런트류는 뒤로.
- 마켓 일부만 받아지면 나머지 마켓의 기존 캐시를 살리고 날짜를 비워 다음에 다시 시도한다.
- 표시명(`Tickers.displayName`)에는 **연결하지 않았다** — 비교 표가 갑자기 한글로 바뀌면 혼란스러워서.

**A3. 종목 일괄 추가** (`Store.addTickers`)
- 토스 Open API 에는 **관심종목·즐겨찾기·그룹 엔드포인트가 없다** (전체 33개 경로 전수 확인).
  앱 안의 사용자 UI 상태는 전혀 노출되지 않으므로 워치리스트는 이 앱에서 관리할 수밖에 없다.
  그래서 토스 앱 관심종목을 옮겨 적는 마찰을 줄이는 쪽으로 대응했다.
- 콤마·줄바꿈·세미콜론·공백으로 구분해 한 번에 붙여넣기. 이미 있는 종목은 건너뛰고 건수를 알려준다.
- 공백은 `NVDA AAPL` 같은 **ASCII 티커 나열일 때만** 구분자로 본다 —
  "버크셔 해서웨이" 처럼 한글이 섞이면 이름 검색어이므로 쪼개지 않는다.

### 기간 설정 · 자산추이 차트 (2026-08-30)

**모든 기간 설정을 1개월 단위 슬라이더로** (`MonthSlider`, 상한 `Store.MAX_MONTHS`=24)
- 분석 기간 · 차트 표시기간 · 자산추이 기간 3개 모두. 기존 6mo/1y/2y 단계 버튼은 제거.
- `Store.lookbackMonths()` 신설. 예전 `range` 토큰이 남아 있으면 개월 수로 한 번 이관한다.
- Yahoo 는 1개월 단위 구간을 안 받으므로 `Store.rangeToken(months)` 로 **덮는 가장 작은 토큰**을
  받아 `Quotes.trimMonths()` 가 정확히 잘라낸다. 토스는 `barCount = months*22+15` 봉으로 요청.
- ⚠️ 3개월 미만은 거래일이 30일(`Quant.EXPANDING_MIN`) 미만이라 분석이 실패할 수 있어
  슬라이더 아래에 경고를 띄운다. 막지는 않았다.
- `PortfolioViewModel` 이 `Yahoo` 를 직접 부르던 것도 `Quotes` 경유로 고침 (시세 소스 설정 무시되던 버그).

**자산추이 일/주/월 리샘플 토글 제거** — 같은 기간을 성기게 그릴 뿐이라 해상도만 떨어졌다.
항상 일 단위로 그리고, 기간은 슬라이더로 조절한다. `Store.equityUnit` 도 함께 삭제.

**EquityChart 전면 개편** (`ui/Charts.kt`)
- 가로 핀치 확대 · 확대 상태에서 드래그 이동 (`chartGestures(xOnly = true)`)
- **아무 지점이나 탭하면 그 시점의 금액**. `chartGestures` 에 `onTapAt: (Float) -> Unit` 추가 —
  확대·이동을 되돌려 콘텐츠 좌표(0~1)로 환산해 넘긴다.
- 헤더에 현재 금액을 큰 글씨로 + 구간/현재 대비 증감. 우측 y축 눈금에 금액 라벨.
  현재 금액 가로 기준선을 항상 그려 선택 지점과 바로 비교된다.
- `baseZero` 파라미터: 누적손익은 0선을 포함(true), **총자산은 보이는 구간에 맞춤(false)**.
  총자산을 0부터 그리면 선이 위에 눌려 붙어 변화가 안 보였다 — "너무 단순화돼 있다"의 원인.

### 증권사 이원화 대응 · Gist 제거 · 국내 종목 (2026-08-30)

**미국주식은 타사 출고가 안 된다** (국내 종목만 이관 가능). 그래서 전면 이관을 포기하고
**메리츠 = 기존 앱(`app-debug.apk`) / 토스 = 새 앱(`quant-toss-debug.apk`)** 으로 나눠 쓰기로 했다.
`applicationId` 가 달라 데이터가 완전히 분리되므로 추가 구현 없이 그대로 성립한다.
새로 사는 것만 토스에서 하고 메리츠 포지션은 사이클대로 매도되며 자연 소멸시킨다 (세금 이벤트 없음).

> 두 증권사 자산을 **합산**해 총자산을 보고 싶어지면: 토스에서 가져온 기록에는 `srcId` 가 있고
> 수기 입력분(메리츠)에는 없으므로 겹치지 않게 더할 수 있다. 메리츠 예수금 입력칸만 추가하면 된다.
> 지금은 만들지 않았다.

**Gist 연동 제거 (토스 앱만)** — `data/Gist.kt` 삭제, 설정 섹션·`Store.gistToken/gistId/setGist` 제거.
이유: "Gist에서 불러오기"가 로컬 매매기록을 **통째로 덮어써서** 토스에서 가져온 체결내역이 날아간다.
두 앱이 같은 Gist 를 보는 구도에서는 사고가 나기 쉬운 버튼이었다.
(Gist 는 읽기 전용이라 앱이 Gist 에 쓰지는 않았다 — 한 방향 덮어쓰기만 문제였다.)
`Store.saveTradesFromJson/saveTickersFromJson/saveSettingsFromJson` 은 호출부가 없어졌지만
파일 선택 가져오기를 붙일 때 재사용하려고 남겨 뒀다.
⚠️ 기존 앱(`android/`)의 Gist 는 그대로 둔다 — 데스크톱과 계속 동기화해야 한다.

**국내 종목 다시 열기**
- 랭킹: `Rankings.MARKETS`(미국·국내) 칩 추가, `Store.rankMarket()`. 국내는 **폴백 목록을 두지 않았다** —
  지어낸 종목코드를 보여주느니 "토스 연동이 필요합니다"라고 알리는 편이 낫다.
- 유니버스: KOSPI·KOSDAQ 추가 → "삼성전자" 검색 가능. 캐시 파일에 `v`(대상 마켓) 표식을 넣어
  미장 전용으로 받아둔 오늘자 캐시가 있어도 다시 받게 했다.
- 표시명: **국내 6자리 코드만** 유니버스 이름으로 채운다 (미국은 티커가 더 읽기 쉬워 그대로).
  `Universe.nameOf()` 는 메모리에 올라와 있을 때만 답한다 — 표시명은 그리는 중에 불려서 파일을 읽으면 안 된다.
  `AppScaffold` 가 앱 시작 시 IO 로 한 번 올리고, **처음 이름이 생겼을 때만** `AppState.bump()` 한다
  (매 실행마다 bump 하면 전 종목 재조회가 걸린다).

### B. 화면 보강

**B1. 미체결 주문 표시** — `GET /api/v1/orders?status=OPEN`
지금 걸어둔 주문을 포트폴리오 탭에 "대기 중 N건"으로. 시장 무관, 조회 전용 원칙에 어긋나지 않음.

**B2. 실제 수수료율** — `GET /api/v1/commissions`
시장별 수수료율. 매매 입력 시 예상 비용 표시 정도. (사이클 통계를 제거해 활용처는 줄었음)

### C. 확인 필요 (미장에서 값이 나오는지 불명)

**C1. 매수 유의사항 배지** — `GET /api/v1/stocks/{symbol}/warnings`
정리매매 · 단기과열 · 투자경고 · 투자위험 · VI 발동 · 신주인수권 — **전부 KRX 제도 용어**라
미국 종목은 빈 응답일 가능성이 높다. 붙이기 전에 AAPL 등으로 한 번 찔러 보고 판단할 것.
값이 나온다면 기존 "신중 매매"(그래프 순차 확인) 흐름에 배지로 얹는 게 자연스럽다.

### D. 이 앱엔 안 맞음 (기록만)

- **호가창** `/orderbook` — 스윙·포지션 트레이딩엔 의미 적음(데이트레이딩용)
- **판매가능수량** `/sellable-quantity` — 주문을 안 하므로 불필요
- **웹소켓 실시간** `wss://openapi-ws.tossinvest.com/ws/v1` — 3초 폴링(`/prices` 200종목 1요청)으로 충분.
  분석이 전부 일봉 기준이라 틱 단위가 필요 없다. AsyncAPI 스펙 문서 별도 필요.
- **주문·정정·취소·조건주문** — 의도적 미구현 (조회 전용 원칙)

### 헤더 배지는 Yahoo 유지

SPY·NASDAQ·**VIX·미 10년물**에 대응하는 토스 엔드포인트가 없다
(`/market-indicators` 는 KOSPI·KOSDAQ·한국 국채 전용). `MarketRepo` 는 Yahoo 그대로 둔다.

---

## 2026-06-19 Android 포팅 패리티 세션 (app.py 정답지 → Kotlin 전수 대조)

> `app.py`를 정답지로 삼아 분석수식/차트/화면기능 3영역 전수 감사 후 누락 일괄 구현. CI(android.yml) 빌드 그린 확인.

### 감사 결론
- **분석 핵심 수식(회귀 Z·모멘텀 M·RSI·MACD·β·σ·밴드·5단계 신호)·포트폴리오 손익/MDD/사이클 통계는 app.py와 완전 일치** — 손대지 않음.
- 차트 5종 색/면적/마커도 대체로 일치. 점선 임계선·눈금 라벨은 직전 커밋(74d0685)의 의도된 차이로 유지.

### 신규 구현 (전부 claude/android-native)
- **기준일(As-of) 시뮬레이션**: `Store.sliceAsof()`로 전 시계열 슬라이싱, 헤더 ✕배지·설정 체크박스/날짜.
- **AppState(전역 상태)**: `asof`·`dataVersion` 관찰 → 설정 변경 시 분석/비교/포트폴리오 탭이 `vm.sync(version)`으로 자동 재로드(수동 새로고침 불필요). ⚠️ `asof` 프로퍼티 자동 setter와 충돌 방지 위해 설정 함수는 `applyAsof()`로 명명(`setAsof` 금지).
- **봉 기준(일봉/주봉)** interval → Yahoo 전달, **KOSDAQ `.KS`→`.KQ` fallback**(Yahoo.symbolCandidates).
- **종목명 override 편집** + **개별/ETF 토글**(Store.individualTickers, isIndividual=저장셋∪한국6자리), MIN_TICKERS=3.
- 탭1: ETF/개별 필터·직접입력 분석·★보유/☆이력 표식·지표설명 expander·현재가 평가손익%·**현재 사이클 진행 게이지**(Portfolio.currentCycleProgress).
- 매매 **메모 인라인 편집**(Store.updateTradeMemo) + 포트폴리오 **📒 매매 일지**(전 종목 시간순).
- 포트폴리오 **MDD 날짜**·자산추이 **일/주/월 단위**(resampleEquity). 비교표 ★/☆.
- 차트: 회귀 산점도 **매매마커(markIdx)** + 가격/캔들 **사이클 화살표 CycleArrow**(평균매수→평균매도, 수익=녹색/손실=빨강).

### 의도적 미포팅 (복원 금지/주의)
- **7단계 통합신호(`score_to_signal`/`MACD_Hist_Z`)**: Android는 5단계 신호만 노출 → 생략. 분석 정확도 영향 없음.
- **KRX 종목명 자동조회**: 온디바이스라 데이터 소스(pykrx 등) 부재 → 불가. **종목명 override 편집**으로 대체.
- 1.5년 조회기간: Yahoo range 토큰에 없어 보류(6mo/1y/2y만).

---

## (이전) 2026-06-10 코드 리뷰 반영 세션 메모는 아래에 보존

## 2026-06-10 코드 리뷰 반영 세션 (app.py 6,040 → ~5,600줄)

### 버그 수정
- **ET 시간대**: UTC-4 고정 → `zoneinfo.ZoneInfo("America/New_York")` (겨울철 1시간 오차 해소)
- **NYSE 휴장일**: Good Friday(부활절 알고리즘 `_easter_date`)·Juneteenth(2022~)·대체휴일(토→금, 일→월, 단 1/1 토요일은 미관측) 추가
- **매매 기록 저장 검증**: 수량/단가 0이면 저장 차단 + 에러 표시 (이전엔 저장돼도 모든 통계에서 invisible)
- **색상 체계 일원화**: `momentum_to_color`(7단계 정수) 삭제 → 전 화면 `momentum_pct_to_color`(5단계 백분위 20/40/60/80)로 통일. 버튼·종목명 헤더·Z/M 수치·탭2 표/산점도·매매일지 모두 동일
- **hover 복원**: 탭1 `fig.update_traces(hoverinfo='skip')` 블랭킷 제거 (trace별 skip 유지), `hovermode='closest'` — Z·M 산점도 날짜 hover 동작. 탭2 산점도·탭3 자산추이 `staticPlot` 해제 (축 `fixedrange`+`dragmode=False`로 모바일 스크롤 보호)
- `pnl_color(0)` → 중립 회색
- 차트 매직 row 번호(`row=4`, `row=5`) → `ROW = {name: idx}` 매핑 (패널 순서 변경에 안전)

### ⛔ 위치 바(±3σ 스테이터스 바) + 액션 카드(매수 의견)는 사용자가 의도적으로 제거한 기능
- 리뷰 중 "만들고 버려지는 코드"로 보여 복원했다가 사용자 지시로 **재삭제** (함수 자체 삭제)
- 삭제된 것: `build_action_card_html`, `compute_cycle_avg_prices`, `momentum_score_to_signal`
- **다시 복원하지 말 것** — 탭1은 헤더(σ·β·Z·M) + 정보 카드 + 차트만 유지

### 죽은 코드 제거 (~1,000줄, 6,040 → 5,013줄)
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

### 헤더 배지 (2차 수정)
- fdr 실패 시 **Yahoo chart API 직접 fallback** (`_yahoo_closes`, query1/query2) — requirements 변경으로
  환경 재빌드되며 fdr 최신 버전의 VIX/^TNX/SPY fetch가 깨져 숫자가 사라졌던 문제 대응
- 배지는 **항상 제목 아래 별도 행** (1행 제목+장상태 / 2행 배지)
- SPY 배지: `🟢 SPY(6M) +12.3%` (6개월 등락률) — 체제 라벨·일간 등락·SMA200 위치는 툴팁
- `get_market_regime` 반환에 `spy_ret_1d`, `spy_above_sma200` 추가

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
