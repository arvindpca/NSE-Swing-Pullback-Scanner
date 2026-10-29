"""API health check: har API ka login/key check karta hai.

Agar koi API key expire / galat ho gayi ho to Telegram par alert bhejta hai,
jisme 2 link hote hain:
  1) Nayi key kahan banani hai (CoinDCX / Angel One / GitHub token)
  2) GitHub me kahan update karni hai (repo ka Secrets page)

Koi trade nahi karta, sirf check karta hai. Sirf wahi API check hoti hai
jiski secret is repo me set hai.
Agar Telegram hi kaam na kare to workflow fail hota hai -> GitHub email bhejta hai.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.request

REPO = os.getenv("GITHUB_REPOSITORY", "arvindpca/unknown")
REPO_NAME = REPO.split("/")[-1]
SECRETS_PAGE = f"https://github.com/{REPO}/settings/secrets/actions"

LINKS = {
    "coindcx": "https://coindcx.com/api-dashboard",
    "angel": "https://smartapi.angelone.in/",
    "github": "https://github.com/settings/tokens",
    "telegram": "https://t.me/BotFather",
}

AUTH_WORDS = ("invalid", "expire", "unauthor", "not authorized", "authentication",
              "api key", "apikey", "signature", "forbidden", "password", "totp", "pin")


def env(name: str) -> str:
    """Secret padho, invisible characters hata ke."""
    raw = os.getenv(name, "") or ""
    return "".join(ch for ch in raw.strip() if unicodedata.category(ch) != "Cf")


def http(url: str, *, data: bytes | None = None, headers: dict | None = None,
         method: str | None = None, timeout: int = 20) -> tuple[int, str]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def looks_like_auth_problem(code: int, body: str) -> bool:
    if code in (401, 403):
        return True
    low = body.lower()
    return any(w in low for w in AUTH_WORDS)


# ---------------------------------------------------------------- checks
# Har check (ok, detail) return karta hai.  ok=None matlab "check nahi ho paya"
# (network / server down) - us par alert nahi bhejte, taaki faltu message na aaye.

def check_coindcx() -> tuple[bool | None, str]:
    key, secret = env("COINDCX_API_KEY"), env("COINDCX_API_SECRET")
    body = json.dumps({"timestamp": int(time.time() * 1000)}, separators=(",", ":"))
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    try:
        code, text = http("https://api.coindcx.com/exchange/v1/users/balances",
                          data=body.encode(), method="POST",
                          headers={"Content-Type": "application/json",
                                   "X-AUTH-APIKEY": key, "X-AUTH-SIGNATURE": sig})
    except Exception as e:  # network
        return None, f"network error: {e}"
    if code == 200:
        return True, "OK"
    if looks_like_auth_problem(code, text):
        return False, f"HTTP {code}: {text[:200]}"
    return None, f"HTTP {code}: {text[:200]}"


def check_angel() -> tuple[bool | None, str]:
    try:
        import pyotp
        from SmartApi import SmartConnect
    except Exception as e:
        return None, f"library missing: {e}"
    client = env("ANGEL_CLIENT_ID") or env("ANGEL_CLIENT_CODE")
    password = env("ANGEL_PASSWORD") or env("ANGEL_PIN")
    try:
        otp = pyotp.TOTP(env("ANGEL_TOTP_SECRET")).now()
    except Exception as e:
        return False, f"ANGEL_TOTP_SECRET galat lag raha hai: {e}"
    try:
        api = SmartConnect(api_key=env("ANGEL_API_KEY"))
        res = api.generateSession(client, password, otp)
    except Exception as e:
        msg = str(e)
        if looks_like_auth_problem(0, msg):
            return False, msg[:200]
        return None, f"network/server error: {msg[:200]}"
    if res and res.get("status"):
        # logout nahi karte - warna chal rahe bot ka session bhi band ho sakta hai
        return True, "OK"
    msg = (res or {}).get("message") or "login failed"
    code = (res or {}).get("errorcode") or ""
    return False, f"{msg} {code}".strip()


def check_github_pat(name: str) -> tuple[bool | None, str]:
    try:
        code, text = http("https://api.github.com/user",
                          headers={"Authorization": f"Bearer {env(name)}",
                                   "Accept": "application/vnd.github+json",
                                   "User-Agent": "api-health-check"})
    except Exception as e:
        return None, f"network error: {e}"
    if code == 200:
        return True, "OK"
    if code == 401:
        return False, "token expire ya galat hai (HTTP 401)"
    return None, f"HTTP {code}"


def check_telegram() -> tuple[bool | None, str]:
    try:
        code, text = http(f"https://api.telegram.org/bot{env('TELEGRAM_BOT_TOKEN')}/getMe")
    except Exception as e:
        return None, f"network error: {e}"
    if code == 200:
        return True, "OK"
    return False, f"HTTP {code}: {text[:150]}"


def send_telegram(text: str) -> bool:
    data = json.dumps({"chat_id": env("TELEGRAM_CHAT_ID"), "text": text,
                       "disable_web_page_preview": True}).encode()
    try:
        code, body = http(f"https://api.telegram.org/bot{env('TELEGRAM_BOT_TOKEN')}/sendMessage",
                          data=data, headers={"Content-Type": "application/json"})
        if code != 200:
            print(f"Telegram send failed: HTTP {code} {body[:200]}")
        return code == 200
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


# ---------------------------------------------------------------- main

def secret_link(name: str) -> str:
    return f"{SECRETS_PAGE}/{name}"


def main() -> int:
    problems: list[str] = []

    if env("COINDCX_API_KEY") and env("COINDCX_API_SECRET"):
        ok, detail = check_coindcx()
        print(f"CoinDCX: {ok} | {detail}")
        if ok is False:
            problems.append(
                "🔴 CoinDCX API kaam nahi kar rahi (expire / galat key)\n"
                f"Error: {detail}\n"
                f"1) Nayi key banao: {LINKS['coindcx']}\n"
                "2) GitHub me update karo:\n"
                f"   COINDCX_API_KEY → {secret_link('COINDCX_API_KEY')}\n"
                f"   COINDCX_API_SECRET → {secret_link('COINDCX_API_SECRET')}")

    if env("ANGEL_API_KEY"):
        ok, detail = check_angel()
        print(f"Angel One: {ok} | {detail}")
        if ok is False:
            names = ["ANGEL_API_KEY"]
            names += ["ANGEL_CLIENT_ID", "ANGEL_PASSWORD"] if env("ANGEL_CLIENT_ID") else ["ANGEL_CLIENT_CODE", "ANGEL_PIN"]
            names += ["ANGEL_TOTP_SECRET"]
            lines = "\n".join(f"   {n} → {secret_link(n)}" for n in names)
            problems.append(
                "🔴 Angel One login fail ho raha hai\n"
                f"Error: {detail}\n"
                f"1) API key / app check karo: {LINKS['angel']}\n"
                "   (password ya PIN badla ho to woh bhi update karna)\n"
                f"2) GitHub me update karo:\n{lines}")

    for pat in ("DISPATCH_PAT",):
        if env(pat):
            ok, detail = check_github_pat(pat)
            print(f"GitHub token {pat}: {ok} | {detail}")
            if ok is False:
                problems.append(
                    f"🔴 GitHub token ({pat}) expire ho gaya - market clock bot ko chalu nahi kar payega\n"
                    f"1) Naya token banao: {LINKS['github']}\n"
                    f"2) Yahan update karo: {secret_link(pat)}")

    tg_ok, tg_detail = check_telegram() if env("TELEGRAM_BOT_TOKEN") else (False, "TELEGRAM_BOT_TOKEN set nahi hai")
    print(f"Telegram: {tg_ok} | {tg_detail}")

    if tg_ok is False:
        # Telegram hi kharab hai -> message nahi ja sakta. Workflow fail karo, GitHub email bhejega.
        print("::error::Telegram bot kaam nahi kar raha. Naya token: "
              f"{LINKS['telegram']}  Update: {secret_link('TELEGRAM_BOT_TOKEN')}")
        for p in problems:
            print("::error::" + p.replace("\n", " | "))
        return 1

    if problems:
        text = (f"⚠️ API ALERT - {REPO_NAME}\n"
                "Bot is API ke bina nahi chalega. Neeche diye link se update karo 👇\n\n"
                + "\n\n".join(problems)
                + f"\n\nSab secrets: {SECRETS_PAGE}")
        send_telegram(text)
        for p in problems:
            print("::error::" + p.replace("\n", " | "))
        return 1  # GitHub email bhi aayega

    print("Sab API theek hain ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
