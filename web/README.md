# 포트폴리오 웹 (집 PC에서 실행)

토스 Open API 는 **허용 IP 목록 밖에서 오는 호출을 막는다.** 폰이 5G 를 쓰면 IP 가 매일
바뀌어 그때마다 WTS 에 다시 등록해야 했다. 이 서버를 IP 가 고정된 PC 에서 돌리면
**한 번만 등록하면 끝난다.** 폰은 이 화면을 브라우저로 볼 뿐 토스에 직접 붙지 않는다.

## 1. 준비

```
cd web
pip install -r requirements.txt
```

앱 키는 둘 중 하나로 준다.

- `web/config.json` 을 만든다 (`config.example.json` 을 복사해서 채운다)
- 또는 환경변수 `TOSS_APP_KEY`, `TOSS_APP_SECRET`

> `config.json` 은 `.gitignore` 에 들어 있다. **절대 커밋하지 말 것.**

## 2. 실행

```
python server.py
```

PC 브라우저에서 <http://localhost:8000> 을 연다.

처음 켤 때 `허용되지 않은 IP입니다` 가 나오면, 토스증권 WTS → 설정 → Open API →
허용 IP 관리에 **이 PC 의 공인 IP** 를 등록한다(<https://ifconfig.me> 에서 확인).

## 3. 폰에서 보기

같은 와이파이라면 `http://<PC의 사설 IP>:8000` 으로 바로 열린다
(`ipconfig` 로 확인. 윈도우 방화벽에서 8000 포트 허용이 필요할 수 있다).

밖에서 보려면 포트포워딩이나 터널이 필요하다 — 3단계에서 정한다.

## 구성

| 파일 | 하는 일 |
|---|---|
| `toss.py` | 토스 API 클라이언트 (**조회 전용**. 주문·정정·취소는 구현하지 않는다) |
| `server.py` | FastAPI — `/api/account` 하나. 20초 캐시로 한도(429)를 피한다 |
| `static/` | 화면. 안드로이드 앱과 같은 A-1 토스 블루 토큰 |
