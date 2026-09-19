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

## 3. 접속 암호

**이 서버는 계좌를 그대로 보여준다.** 그래서 PC 자신(`localhost`)이 아닌 곳에서 오는
요청은 전부 암호를 묻는다. 암호는 처음 실행할 때 `config.json` 에 자동으로 만들어지고
콘솔에 찍힌다.

```
  PC 에서:     http://localhost:8000   (암호 없이 열립니다)
  다른 기기에서: 접속 암호  xxxxxxxxxxxxxxxxxxxx
  즐겨찾기용:   <주소>/?key=xxxxxxxxxxxxxxxxxxxx
```

한 번 로그인하면 쿠키가 남아 다시 묻지 않는다. 폰에서는 `?key=...` 가 붙은 주소를
한 번 열면 그걸로 끝이다(주소창에서 key 는 바로 지워진다).

암호를 바꾸려면 `config.json` 의 `access_token` 을 고치고 서버를 다시 켜면 된다.

## 4. 폰에서 보기

**같은 와이파이** — `http://<PC 사설 IP>:8000`. 주소는 이렇게 확인한다.

```powershell
(Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" }).IPAddress
```

윈도우 방화벽이 물어보면 허용한다. 크롬 메뉴 → 홈 화면에 추가를 하면 앱처럼 쓸 수 있다.

**집 밖에서** — Cloudflare 터널이 제일 쉽다. 공유기 설정도, 폰에 앱 설치도 필요 없고
https 도 자동으로 붙는다.

```powershell
winget install --id Cloudflare.cloudflared
.\run.ps1 -Tunnel
```

콘솔에 `https://xxxx-xxxx.trycloudflare.com` 같은 주소가 뜬다. 폰에서 그 주소 뒤에
`/?key=<접속 암호>` 를 붙여 한 번 열면 그 뒤로는 암호를 묻지 않는다.

> ⚠️ 이 주소는 **서버를 껐다 켤 때마다 바뀐다.** Cloudflare 가 임시용이라고 못 박아 둔
> 기능이라 그렇다.

### 고정 주소로 쓰려면

도메인이 하나 필요하다(Cloudflare 에서 사면 원가, 연 몇천 원짜리도 있다).
도메인을 Cloudflare 에 올린 뒤:

```powershell
cloudflared tunnel login
cloudflared tunnel create quant
cloudflared tunnel route dns quant quant.내도메인.com
cloudflared tunnel run --url http://localhost:8000 quant
```

이제 `https://quant.내도메인.com` 이 계속 같은 주소로 열린다.

## 5. 부팅할 때 자동 실행

작업 스케줄러에 등록하면 로그인할 때 숨겨진 창으로 알아서 뜬다. 관리자 권한은 필요 없다.

```powershell
.\install-task.ps1            # 서버만
.\install-task.ps1 -Tunnel    # 터널까지 (외부 접속)
```

| 하고 싶은 것 | 명령 |
|---|---|
| 지금 바로 시작 | `Start-ScheduledTask -TaskName QuantDashboard` |
| 중지 | `Stop-ScheduledTask -TaskName QuantDashboard` |
| 해제 | `.\install-task.ps1 -Remove` |

네트워크가 늦게 붙는 경우가 있어 로그인 후 30초 뒤에 시작하고, 노트북이 배터리로 돌 때도
멈추지 않게 해 뒀다.

> ⚠️ `-Tunnel` 로 등록하면 **주소가 실행할 때마다 바뀐다.** 자동 실행과 고정 주소를 같이
> 쓰려면 아래 '고정 주소'대로 도메인을 붙이는 편이 낫다.

### 포트포워딩으로 하려면

공유기에서 외부 8000 → 이 PC 8000 으로 열면 `http://<공인IP>:8000` 으로 붙는다.

> ⚠️ 이 경우 **암호가 암호화되지 않은 채로 오간다.** 같은 네트워크를 지나는 누군가가
> 들여다볼 수 있다. 계좌를 보는 화면이니 터널(https)을 쓰는 편이 낫다.

## 화면

하단 탭 4개 — **비교 · 분석 · 포트폴리오 · 설정**. 안드로이드 앱과 같은 구성이다.

- **비교**: 종목별 현재가·일 등락률·Z·M. 종목 왼쪽에 당일 미니 캔들, 보유는 금색 점.
  열 제목을 누르면 정렬. 행을 누르면 분석으로.
- **분석**: `[시계열]` 캔들(+평단선·매매 마커) · Z·M · MACD · RSI, 일봉/1분 전환.
  `[산점도]` 회귀 산점도(SPY 대비) · Z·M 궤적.
  차트를 꾹 누르거나 마우스를 올리면 시고저종 상자가 뜬다.
- **포트폴리오**: 총자산·비중 파이·보유 목록·**자산 추이**·**평가손익**·**매매 일지**.
  원/$ 전환. 원금을 적어 두면 원금선과 총손익도.
- **설정**: 분석 기간, 실시간 갱신 주기, 원금(입금 장부), 종목 관리,
  체결내역 가져오기, 일봉 다시 받기.

> **자산 추이는 기록이 시작된 날부터 쌓인다.** 토스에 과거 잔고 API 가 없어서, 계좌를
> 조회할 때마다 그날 값을 `web/data/snapshots.json` 에 남기는 방식이다(하루 1회 덮어쓰기).
> 서버를 켜 두면 자동으로 쌓인다.

## 개발 중 화면 확인

토스 API 는 허용 IP 밖에서 막히고 개발 환경에는 앱 키도 없다. 그래서 **가짜 시세**로
띄워 화면을 직접 볼 수 있게 해 뒀다.

```powershell
$env:QUANT_MOCK = "1"; py server.py      # 숫자는 가짜, 레이아웃 확인용
```

`shot.py` 는 폰 크기(412×915)로 각 탭을 캡처한다(헤드리스 크로미움 필요).
칩이 줄바꿈되거나 차트가 탭바에 가리는 문제를 이걸로 잡았다.

`test_ui.py` 는 한 발 더 나아가 **직접 눌러 본다.** 서버를 가짜 시세로 띄우고
세그먼트·탭·행·종목 추가/삭제를 눌러, 누른 버튼에 칠이 옮겨 갔는지와 내용이 실제로
바뀌었는지를 같이 본다.

```powershell
py test_ui.py        # 실패가 있으면 종료코드 1
```

캡처만으로는 "한국을 눌렀는데 색은 미국에 남아 있고 데이터만 바뀌는" 문제를 못 잡는다.
실제로 그랬고, 그래서 이 검사를 만들었다. 검사용으로 임시 폴더(`QUANT_DATA`)를 쓰므로
진짜 입금·매매·스냅샷 기록은 건드리지 않는다.

## 구성

| 파일 | 하는 일 |
|---|---|
| `toss.py` | 토스 API 클라이언트 (**조회 전용**. 주문·정정·취소는 구현하지 않는다) |
| `quant.py` | 회귀·Z·M·MACD·RSI — `quant/Quant.kt` 를 그대로 옮긴 것 |
| `repo.py` | 일봉 캐시(6시간, 파일) + 비교/분석 계산. 동시 요청 3개 제한 + 429 재시도 |
| `store.py` | 설정·종목·매매기록·원금 (`web/data/` 아래 JSON) |
| `server.py` | FastAPI 엔드포인트 |
| `snapshots.py` | 일별 잔고 기록 (`data/Snapshots.kt`) — 자산 추이의 유일한 출처 |
| `static/` | 화면. 안드로이드와 같은 A-1 토스 블루 토큰. 차트는 lightweight-charts |
| `static/scatter.js` | 산점도 2종 — lightweight-charts 가 산점도를 지원하지 않아 캔버스로 |
| `run.ps1` / `install-task.ps1` | 실행 / 자동 실행 등록 |
| `mock.py` / `shot.py` / `test_ui.py` | **개발용** — 가짜 시세, 화면 캡처, 눌러 보는 검사 |

`web/data/` 는 이 PC 안에만 있는 것이라 커밋되지 않는다.

## 알아 둘 것

- **비교 탭 첫 조회는 20~30초** 걸린다. 종목 수만큼 2년치 일봉을 받기 때문이다.
  받은 일봉은 파일로 캐시(6시간)하므로 그 다음부터, 그리고 서버를 껐다 켜도 빠르다.
- 분석 화면의 **평단선**은 토스 보유 정보를 우선 쓰고, 없으면 체결내역에서 역산한다.
- 차트 라이브러리는 `static/vendor/` 에 같이 두었다. 인터넷이 끊겨도 차트가 뜬다.
