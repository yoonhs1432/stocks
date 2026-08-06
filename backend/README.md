# Quant Dashboard — Backend API

Streamlit `app.py`의 분석 로직을 네이티브 Android 클라이언트가 쓸 수 있도록
분리한 FastAPI 서버. **개인용**이라 인증 없음(신뢰된 네트워크/사이드로드 전제).

## 구조
- `analysis.py` — 순수 퀀트 코어 (회귀 Z-score, 모멘텀 M, 사이클, 드로다운). app.py 수식 포팅, 의존성 없음.
- `data.py` — FinanceDataReader + Yahoo chart API fallback, 자체 TTL 캐시, NYSE 휴장일/시간대.
- `store.py` — 매매 기록/종목/설정 영속화 (루트의 `trade_history.json` 공유, 선택적 Gist 동기화).
- `main.py` — REST 엔드포인트.

## 실행
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
브라우저에서 `http://<PC-IP>:8000/docs` 로 자동 생성된 OpenAPI 문서 확인.
같은 Wi-Fi의 갤럭시에서 `http://<PC-IP>:8000` 으로 접속.

### Gist 동기화 (선택)
데스크톱 Streamlit과 매매 기록을 공유하려면:
```bash
export GITHUB_TOKEN=ghp_xxx
export GIST_ID=xxxxxxxx
```

## 엔드포인트 요약
| Method | Path | 용도 |
|---|---|---|
| GET | `/health` | 헬스체크 |
| GET | `/market?asof=` | 장 상태 + 체제(SPY) + VIX/10Y/KRW |
| GET | `/tickers` | 종목 목록 + 표시명 + 보유 여부 |
| GET | `/overview?start=&candle=&asof=` | 전 종목 요약 (비교 표/산점도) |
| GET | `/analysis/{ticker}?start=&candle=&asof=` | 단일 종목 차트 시리즈 + 요약 + OHLC |
| GET | `/portfolio?start=&candle=` | 보유/실현/자산추이/드로다운 |
| GET | `/trades?ticker=` | 매매 기록 조회 |
| POST | `/trades` | 매매 기록 추가 (qty·price > 0 필수) |
| DELETE | `/trades/{ticker}/{idx}` | 매매 기록 삭제 |
| POST | `/refresh` | 시세 캐시 강제 비움 |

## 주의
- app.py와 분석 수식이 **중복 포팅**됨 (CLAUDE.md: app.py 단일 파일 유지). 수식 변경 시 양쪽 동시 수정.
- 색상은 백엔드가 정하지 않음 — `signal` 라벨(strong_buy/buy/hold/sell/strong_sell)을 주고 클라이언트가 한국식 색(매수=빨강) 매핑.
