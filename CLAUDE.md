# Quant Trading Dashboard (stocks)

## Project structure
- `app.py`: Streamlit 메인 앱 (단일 파일 유지)
- `requirements.txt`: Python 의존성
- `trade_history.json`: 거래 기록
- `.devcontainer/`: Streamlit Cloud 배포 설정

## Coding conventions
- app.py는 단일 파일로 유지 (part1/part2/part3 분할 금지)
- 부분 수정 가능 (git이 버전 관리 담당, _vN 접미사 사용 안 함)
- 한국식 색상: 수익/매수=빨강, 손실/매도=파랑
- 그래프 라벨: 영문, 세로형 모바일 최적화
- NYSE 휴장일 필터링 적용
- 주석은 한국어
- Config dataclass(frozen)에 매직 넘버 집약
- sklearn 대신 numpy.polyfit 사용

## Git workflow
- 수정 후 자동으로 git add, commit, push (별도 지시 없으면 main 브랜치로)
- 커밋 메시지: conventional commits (feat:, fix:, refactor:, perf:, docs:)
- 큰 변경은 별도 브랜치 (claude/feature-xyz) 후 PR
- Streamlit Cloud가 main을 watch 중이므로 push 시 자동 재배포됨

## Stack
- Python 3.11, Streamlit, FinanceDataReader, Plotly, Pandas, Numpy
- ThreadPoolExecutor로 병렬 데이터 fetch

## Communication
- 답변과 주석은 한국어
- 작업 완료 후 변경사항 요약 + git push 결과 보고
