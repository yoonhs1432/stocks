# 네이티브 포팅 아키텍처 (Galaxy S24 Ultra)

> 브랜치 `claude/android-native`. 목표: Streamlit 대시보드를 **네이티브 Android(Kotlin/Compose) + Python 백엔드**로 포팅. **개인용**(사이드로드, 인증 없음).

## 큰 그림

```
┌─────────────────────────┐        HTTP/JSON         ┌──────────────────────────┐
│  Android (Kotlin/Compose)│ ───────────────────────▶ │  FastAPI 백엔드 (backend/) │
│  - 차트: Vico/MPAndroid  │ ◀─────────────────────── │  - analysis.py (순수 코어) │
│  - Retrofit + Moshi      │                          │  - data.py (시세+캐시)     │
│  - 화면: 분석/비교/포폴   │                          │  - store.py (Gist/JSON)    │
└─────────────────────────┘                          └──────────────────────────┘
                                                              │
                                              FinanceDataReader / Yahoo chart API
```

- **분석 로직은 백엔드에 1벌**(Python). 검증된 수식을 Kotlin으로 재구현하지 않음 → 드리프트/버그 위험 회피.
- **데이터는 Gist 공유** → 데스크톱 Streamlit과 모바일 앱이 같은 매매 기록을 봄.
- `app.py`(Streamlit)는 **데스크톱 버전으로 유지** — 폐기 아님.

## 진행 단계

### ✅ 1단계 — 백엔드 (완료)
- `backend/analysis.py` · `data.py` · `store.py` · `main.py` · `requirements.txt`
- app.py 수식 그대로 포팅, 구문 검사 통과. (실행 검증은 의존성 설치 후 사용자 환경에서)

### ⬜ 2단계 — Android 프로젝트 스캐폴드
- Jetpack Compose + Material3 (다크 테마, 한국식 매수=빨강/매도=파랑 컬러 토큰)
- Retrofit + Moshi + Coroutines, `BASE_URL` 설정 화면 (백엔드 PC IP)
- 데이터 모델: `MarketInfo`, `OverviewRow`, `AnalysisResponse`, `Portfolio`, `Trade`
- 하단 탭 4개: 분석 / 비교 / 포트폴리오 / 설정

### ⬜ 3단계 — 차트
- 차트 라이브러리 후보: **Vico**(Compose 네이티브, 추천) 또는 MPAndroidChart(성숙·기능 많음)
- 우선순위: ① 가격+회귀밴드, ② Z·M 라인(0~100), ③ RSI, ④ 캔들, ⑤ Z-M 산점도
- 모바일 세로 최적화: 패널 세로 스택, 핀치 줌은 가로축만(세로 스크롤은 페이지로)

### ⬜ 4단계 — 기능 완성
- 매매 기록 입력/삭제(POST/DELETE), 종목 추가/삭제, 기준일 시뮬레이션
- Pull-to-refresh → `/refresh` + 재조회
- (선택) 오프라인 캐시: 마지막 응답 Room 저장

## 미해결 결정
- 백엔드 호스팅: 집 PC 상시 구동 vs 무료 클라우드(Render/Fly.io 등) vs 스마트폰 Termux. 개인용이면 집 PC + DDNS가 가장 단순.
- 차트 라이브러리 최종 선택 (Vico 권장).

## 컨벤션
- 색상: 매수/수익=빨강(#dc2626), 매도/손실=파랑(#2563eb). 5단계 신호 라벨을 백엔드가 주고 클라이언트가 색 매핑.
- 그래프 라벨: 영문, 세로형.
- 백엔드 분석 수식 변경 시 `app.py`와 `backend/analysis.py` **동시 수정**.
