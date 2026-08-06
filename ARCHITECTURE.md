# 네이티브 포팅 아키텍처 (Galaxy S24 Ultra)

> 브랜치 `claude/android-native`. 목표: Streamlit 대시보드를 **완전 온디바이스 네이티브 Android(Kotlin/Compose)** 앱으로 포팅. **개인용**(사이드로드, 인증 없음).
> 결정: 서버 없는 **온디바이스** — 앱이 직접 Yahoo에서 시세를 받아 분석까지 폰에서 수행. APK는 **GitHub Actions가 클라우드 빌드** → 폰에 사이드로드 (PC·안드로이드 스튜디오 불필요).

## 큰 그림

```
┌──────────────────────────────────────────────┐      HTTPS       ┌──────────────────┐
│  Android 앱 (Kotlin/Compose) — 단일 APK         │ ───────────────▶ │ Yahoo chart API  │
│  - quant/Quant.kt : 분석 코어 (app.py 수식 포팅) │ ◀─────────────── │ (시세 종가)      │
│  - data/Yahoo.kt  : 시세 페치 (HttpURLConnection)│                  └──────────────────┘
│  - ui/*           : Compose 화면 + Canvas 차트   │
│  - 매매기록: 폰 로컬 저장 (추후 Room/파일)        │
└──────────────────────────────────────────────┘
```

- **서버 없음.** 분석 수식을 Kotlin으로 재구현 → 폰에서 직접 계산. 인터넷은 시세 조회에만 필요.
- `backend/`는 **참고용 포팅**으로 남김 (이 온디바이스 앱은 사용하지 않음). Kotlin 포팅의 정답지 역할.
- `app.py`(Streamlit 데스크톱)는 그대로 유지.

## APK 빌드 (PC 없이)
- `.github/workflows/android.yml` 이 push마다 클라우드에서 `assembleDebug` → APK 생성.
- APK는 워크플로 artifact + `android-latest` 릴리스에 첨부. **폰 브라우저로 릴리스 페이지에서 APK 내려받아 설치**("출처 불명 앱 설치" 허용).
- 디버그 APK는 자동 디버그 키로 서명되어 바로 설치 가능.

## 진행 단계

### ✅ 1단계 — (참고) FastAPI 백엔드
`backend/` — app.py 수식 포팅. 온디바이스로 방향 전환하며 **참고용**으로만 보존.

### ✅ 2단계 — Android v1 (완료)
- Gradle(Kotlin DSL) 프로젝트 + GitHub Actions 빌드 파이프라인
- `Quant.kt`: 회귀 Z-score, 모멘텀 M, RSI/MACD (app.py 수식 Kotlin 포팅, 전수 대조 일치)
- `Yahoo.kt`: 종가 페치 (US 티커. 한국 6자리는 `.KS`→`.KQ` fallback)
- 분석/비교/포트폴리오/설정 4탭 + 캔들·Z·M·MACD·RSI·산점도 Canvas 차트

### ✅ 3단계 — 기능 확장 (대부분 완료, 2026-06-19)
- 비교 표 / Z·M·σ·β 산점도 / 포트폴리오(손익·MDD·사이클통계·매매일지) / 매매기록 입력·삭제·메모편집
- 캔들 차트, **기준일(As-of) 시뮬레이션**, 회귀 매매마커 + 사이클 화살표
- 한국 종목 `.KS/.KQ` 매핑, 종목명 override 편집, 개별/ETF 필터, 봉기준(일/주봉)
- 설정 변경 시 전 탭 자동 반영(AppState.dataVersion)
- ⬜ 잔여: 7단계 통합신호(의도 생략), KRX 이름 자동조회(온디바이스 불가→override 대체), pull-to-refresh

## 컨벤션
- 색상: 매수/수익=빨강(#dc2626), 매도/손실=파랑(#2563eb). 신호 라벨 5단계.
- 분석 수식 변경 시 `app.py` · `backend/analysis.py` · `android/.../Quant.kt` 함께 점검.
- 그래프 라벨 영문, 세로형.

