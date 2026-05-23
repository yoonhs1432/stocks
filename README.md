# 📈 퀀트 트레이딩 대시보드

FinanceDataReader 기반 주가 데이터로 **종목 분석 · 비교 · 포트폴리오 관리**를 지원하는 Streamlit 대시보드입니다.

## ✨ 주요 기능

- **종목 분석**: 가격 추세, 기술적 지표, 회귀선 기반 시그널
- **종목 비교**: 다중 종목 정규화 비교, 상관관계 분석
- **포트폴리오**: `trade_history.json` 기반 손익/수익률 트래킹
- 한국식 색상 체계 (수익/매수=빨강, 손실/매도=파랑)
- NYSE 휴장일 필터링 적용
- ThreadPoolExecutor 병렬 데이터 fetch

## 🛠️ 스택

Python 3.11 · Streamlit · FinanceDataReader · Plotly · Pandas · Numpy

## 🚀 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

## 📁 프로젝트 구조

```
stocks/
├── app.py               # Streamlit 메인 앱 (단일 파일)
├── requirements.txt     # 의존성
├── trade_history.json   # 거래 기록
└── .devcontainer/       # Streamlit Cloud 배포 설정
```

## ☁️ 배포

Streamlit Community Cloud에서 `main` 브랜치를 watch 하여 자동 재배포됩니다.

- **Live**: _(배포 URL을 여기에 입력하세요)_
