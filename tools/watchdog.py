"""Watchdog: har ~20 min check karta hai ki bots theek chal rahe hain.

Telegram alert bhejta hai jab:
  1) Koi bhi workflow FAIL ho (error / timeout) - jo "Bot Stopped Alert" me pehle se nahi hai.
  2) Koi bot chupchaap ruk jaye - uska run time par na chale (tools/watchdog_config.json).

Ek hi problem par baar-baar message nahi aata: failure ek baar, aur "bot ruka hua" har 2 ghante me
yaad dilata hai jab tak theek na ho.  State .watchdog/state.json me (GitHub cache) rehti hai.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
REPO = os.getenv("GITHUB_REPOSITORY", "arvindpca/unknown")
REPO_NAME = REPO.split("/")[-1]
TOKEN = os.getenv("GITHUB_TOKEN", "")
CONFIG_PATH = Path("tools/watchdog_config.json")
STATE_PATH = Path(".watchdog/state.json")
ALERTED_WORKFLOWS_FILE = Path(".github/workflows/bot-stopped-alert.yml")
BAD = {"failure", "timed_out", "startup_failure"}
ALIVE = {"in_progress", "queued", "pending", "waiting", "requested"}
FAILURE_LOOKBACK_H = 3
STALE_REPEAT_MIN = 120
MARKET_FROM, MARKET_TO = "09:30", "15:15"  # IST, Mon-Fri


# ---------------------------------------------------------------- helpers
def gh(path: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
                 "User-Agent": "watchdog"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def hm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def ist_at(day: datetime, s: str) -> datetime:
    h, m = hm(s)
    return day.replace(hour=h, minute=m, second=0, microsecond=0)


def send_telegram(text: str) -> bool:
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not tok or not chat:
        print("Telegram secrets missing")
        return False
    data = json.dumps({"chat_id": chat, "text": text, "disable_web_page_preview": True}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception as e:  # never print the URL (has the token)
        print(f"Telegram send failed: {type(e).__name__}")
        return False


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"alerted_runs": [], "stale": {}}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["alerted_runs"] = state["alerted_runs"][-300:]
    STATE_PATH.write_text(json.dumps(state, indent=1))


def instant_alert_names() -> set[str]:
    """Workflows already covered by bot-stopped-alert.yml (instant alert) - skip to avoid double msgs."""
    try:
        text = ALERTED_WORKFLOWS_FILE.read_text()
    except OSError:
        return set()
    block = text.split("workflows:", 1)[1].split("types:", 1)[0] if "workflows:" in text else ""
    return set(re.findall(r'-\s*"([^"]+)"', block))


def runs_of(wf: str, per_page: int = 10) -> list[dict]:
    return gh(f"actions/workflows/{wf}/runs?per_page={per_page}").get("workflow_runs", [])


def is_market_day(now_ist: datetime) -> bool:
    return now_ist.weekday() < 5


# ---------------------------------------------------------------- checks
def check_failures(state: dict, now: datetime) -> list[str]:
    skip = instant_alert_names()
    since = (now - timedelta(hours=FAILURE_LOOKBACK_H)).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs = gh(f"actions/runs?per_page=100&created=>={since}").get("workflow_runs", [])
    msgs = []
    for r in runs:
        if r.get("status") != "completed" or r.get("conclusion") not in BAD:
            continue
        if r["name"] in skip or r["name"].startswith("Watchdog") or r["id"] in state["alerted_runs"]:
            continue
        state["alerted_runs"].append(r["id"])
        when = parse_ts(r["updated_at"]).astimezone(IST).strftime("%d-%b %H:%M IST")
        msgs.append(f"🛑 ERROR: {r['name']}\nResult: {r['conclusion']} ({when})\nDekho: {r['html_url']}")
    return msgs


def last_activity(wf: str) -> tuple[bool, datetime | None, str]:
    """(alive_now, last_time, link)"""
    runs = runs_of(wf)
    link = f"https://github.com/{REPO}/actions/workflows/{wf}"
    if not runs:
        return False, None, link
    if any(r.get("status") in ALIVE for r in runs):
        return True, None, link
    latest = max(parse_ts(r["updated_at"]) for r in runs)
    return False, latest, link


def should_repeat(state: dict, key: str, now: datetime) -> bool:
    last = state["stale"].get(key)
    if last and now.timestamp() - last < STALE_REPEAT_MIN * 60:
        return False
    state["stale"][key] = now.timestamp()
    return True


def check_interval(rule: dict, state: dict, now: datetime) -> str | None:
    wf, gap = rule["wf"], int(rule["gap_min"])
    now_ist = now.astimezone(IST)
    if rule.get("window") == "market":
        if not is_market_day(now_ist):
            return None
        start = ist_at(now_ist, MARKET_FROM) + timedelta(minutes=gap)
        if not (start <= now_ist <= ist_at(now_ist, MARKET_TO)):
            return None
    alive, last, link = last_activity(wf)
    if alive:
        state["stale"].pop(wf, None)
        return None
    age = (now - last).total_seconds() / 60 if last else None
    if age is not None and age <= gap:
        state["stale"].pop(wf, None)
        return None
    if not should_repeat(state, wf, now):
        return None
    when = last.astimezone(IST).strftime("%d-%b %H:%M IST") if last else "kabhi nahi"
    extra = "\n(Agar aaj market holiday hai to ignore karo)" if rule.get("window") == "market" else ""
    return (f"🕐 BOT CHUP HAI: {rule.get('label', wf)}\n"
            f"Pichhle {gap} min me koi run nahi hua. Aakhri run: {when}\n"
            f"Dekho: {link}{extra}")


def check_daily(rule: dict, state: dict, now: datetime) -> str | None:
    wf = rule["wf"]
    now_ist = now.astimezone(IST)
    if not is_market_day(now_ist):
        return None
    deadline = ist_at(now_ist, rule["by"])
    if now_ist < deadline:
        return None
    day_start = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    runs = runs_of(wf, 20)
    today = [r for r in runs if parse_ts(r["created_at"]).astimezone(IST) >= day_start]
    key = f"{wf}@{now_ist.date()}"
    if today:
        return None
    if key in state["stale"]:
        return None  # aaj ke liye ek hi baar
    state["stale"][key] = now.timestamp()
    return (f"🕐 BOT NAHI CHALA: {rule.get('label', wf)}\n"
            f"Aaj {rule['by']} IST tak iska run hona chahiye tha, nahi hua.\n"
            f"Dekho: https://github.com/{REPO}/actions/workflows/{wf}\n"
            f"(Agar aaj market holiday hai to ignore karo)")


# ---------------------------------------------------------------- main
def main() -> int:
    now = datetime.now(timezone.utc)
    state = load_state()
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    msgs: list[str] = []
    try:
        msgs += check_failures(state, now)
    except Exception as e:
        print(f"failure check error: {e}")
    # purani daily keys saaf karo
    cutoff = now.timestamp() - 3 * 86400
    state["stale"] = {k: v for k, v in state["stale"].items() if v >= cutoff}
    for rule in cfg.get("interval", []):
        try:
            m = check_interval(rule, state, now)
            if m:
                msgs.append(m)
        except Exception as e:
            print(f"interval check error {rule.get('wf')}: {e}")
    for rule in cfg.get("daily", []):
        try:
            m = check_daily(rule, state, now)
            if m:
                msgs.append(m)
        except Exception as e:
            print(f"daily check error {rule.get('wf')}: {e}")

    for m in msgs:
        print(m.replace("\n", " | "))
    if msgs:
        text = f"⚠️ WATCHDOG - {REPO_NAME}\n\n" + "\n\n".join(msgs)
        if not send_telegram(text):
            save_state(state)
            print("::error::Telegram alert nahi gaya")
            return 1
    else:
        print("Sab bots theek chal rahe hain ✅")
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
