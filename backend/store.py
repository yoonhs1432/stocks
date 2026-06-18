"""영속화 — 매매 기록 / 종목 리스트 / 설정 (로컬 JSON + 선택적 Gist).

app.py와 동일한 파일/Gist 포맷을 공유하므로 데스크톱(Streamlit)과
모바일(백엔드)이 같은 데이터를 읽고 쓴다.
환경변수 GITHUB_TOKEN, GIST_ID 설정 시 Gist 동기화.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import requests

log = logging.getLogger("quant.store")

# app.py와 동일 경로/파일명 (저장소 루트 기준)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADE_FILE = os.path.join(_ROOT, "trade_history.json")
TICKERS_FILE = os.path.join(_ROOT, "target_tickers.json")
SETTINGS_FILE = os.path.join(_ROOT, "settings.json")
GIST_FILENAME = "quant_trade_history.json"
TICKERS_GIST_FILENAME = "quant_target_tickers.json"
SETTINGS_GIST_FILENAME = "quant_settings.json"

DEFAULT_TICKERS = [
    "FNGU", "TQQQ", "SOXL", "HIBL", "QPUX", "LABU", "DFEN", "DPST",
    "GDXU", "KORU", "005930", "AVXX", "SPYU", "TARK", "URTY", "TNA",
    "BNKU", "BTC-USD", "ETH-USD", "GLD",
]
TICKER_DISPLAY_NAMES = {"BTC-USD": "BTC", "ETH-USD": "ETH", "005930": "삼전", "000660": "하닉"}
MIN_TICKERS = 3
HTTP_TIMEOUT = 6


def _gist_cfg() -> tuple[str, str]:
    return (os.environ.get("GITHUB_TOKEN", "").strip(),
            os.environ.get("GIST_ID", "").strip())


def _gist_read(gist_id: str, token: str, filename: str) -> Optional[dict]:
    try:
        resp = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            timeout=HTTP_TIMEOUT,
        )
        if resp.ok:
            files = resp.json().get("files", {})
            if filename in files:
                return json.loads(files[filename]["content"])
    except (requests.RequestException, json.JSONDecodeError) as e:
        log.warning(f"gist read {filename}: {e}")
    return None


def _gist_write(gist_id: str, token: str, filename: str, data: dict) -> None:
    try:
        requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            json={"files": {filename: {"content": json.dumps(data, indent=4, ensure_ascii=False)}}},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as e:
        log.warning(f"gist write {filename}: {e}")


def _load(local_file: str, gist_filename: str) -> dict:
    token, gist_id = _gist_cfg()
    if token and gist_id:
        data = _gist_read(gist_id, token, gist_filename)
        if data is not None:
            return data
    if os.path.exists(local_file):
        try:
            with open(local_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.error(f"local read {local_file}: {e}")
    return {}


def _save(local_file: str, gist_filename: str, data: dict) -> None:
    try:
        with open(local_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except OSError as e:
        log.error(f"local write {local_file}: {e}")
    token, gist_id = _gist_cfg()
    if token and gist_id:
        _gist_write(gist_id, token, gist_filename, data)


# ── 매매 기록 ──
def load_trades() -> dict:
    return _load(TRADE_FILE, GIST_FILENAME)


def save_trades(h: dict) -> None:
    _save(TRADE_FILE, GIST_FILENAME, h)


# ── 종목 리스트 ──
def load_tickers() -> list[str]:
    data = _load(TICKERS_FILE, TICKERS_GIST_FILENAME)
    tickers = data.get("tickers") if isinstance(data, dict) else None
    if isinstance(tickers, list) and len(tickers) >= MIN_TICKERS:
        seen, out = set(), []
        for t in tickers:
            t = str(t).strip().upper()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        if len(out) >= MIN_TICKERS:
            return out
    return list(DEFAULT_TICKERS)


def save_tickers(tickers: list[str]) -> None:
    _save(TICKERS_FILE, TICKERS_GIST_FILENAME, {"tickers": tickers})


# ── 설정 ──
def load_settings() -> dict:
    return _load(SETTINGS_FILE, SETTINGS_GIST_FILENAME)


def save_settings(s: dict) -> None:
    _save(SETTINGS_FILE, SETTINGS_GIST_FILENAME, s)


def display_name(ticker: str, overrides: Optional[dict] = None,
                 krx: Optional[dict] = None) -> str:
    overrides = overrides or {}
    if ticker in overrides and overrides[ticker]:
        return overrides[ticker]
    if ticker in TICKER_DISPLAY_NAMES:
        return TICKER_DISPLAY_NAMES[ticker]
    if ticker.isdigit() and len(ticker) == 6 and krx:
        return krx.get(ticker, ticker)
    return ticker
