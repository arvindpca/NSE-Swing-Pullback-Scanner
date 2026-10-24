"""
Swing Pullback Scanner + 2 Saal Backtest (Nifty 500, Daily)
===========================================================
Kaise chalayein:
    pip install yfinance pandas numpy
    python swing_backtest.py

Pehli baar: ~500 stocks ka 3 saal data download hoga (5-10 min), data/ folder mein save.
Agli baar: sirf pichle ~15 din ka naya data aayega (1-2 min).

Output files:
    output/summary.txt        -> backtest ka result (win rate, P&L, saal-wise)
    output/trades.csv         -> har trade ki detail
    output/today_signals.csv  -> KAL ke liye stock list (entry, SL, target ke saath)
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
import json
import urllib.request
import urllib.parse
import yfinance as yf

# ------------------------- SETTINGS (yahan badlo) -------------------------
BACKTEST_YEARS = 2
RISK_PER_TRADE = 1000        # har trade mein kitna Rs risk (position size isse banega)
REWARD_RISK = 2.0            # target = risk x 2
MAX_HOLD_DAYS = 15           # itne din mein target/SL nahi aaya to close pe exit
COST_PCT = 0.2               # brokerage + slippage, round trip (%)
DATA_DIR = "data"
OUT_DIR = "output"
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
# --------------------------------------------------------------------------


# ========================= 1. STOCK LIST =========================
def get_symbols():
    """Nifty 500 list. Pehle local file dekho, nahi to NSE se lo."""
    local = os.path.join(DATA_DIR, "nifty500.csv")
    if not os.path.exists(local):
        os.makedirs(DATA_DIR, exist_ok=True)
        df = pd.read_csv(NIFTY500_URL, storage_options={"User-Agent": "Mozilla/5.0"})
        df.to_csv(local, index=False)
    df = pd.read_csv(local)
    return [s.strip() + ".NS" for s in df["Symbol"].dropna()]


# ========================= 2. DATA (cache ke saath) =========================
def load_data(symbols):
    cache = os.path.join(DATA_DIR, "prices.pkl")
    data = {}
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            data = pickle.load(f)

    fresh = False
    if data:
        last = max(df.index.max() for df in data.values())
        fresh = (pd.Timestamp.today().normalize() - last).days <= 20
        print(f"Cache mila (last date {last.date()}).")
    if not fresh:
        data = {}
        print("Pura 3 saal ka data download ho raha hai (2 saal backtest + 1 saal warmup)...")

    # Jo stock cache mein hai unka sirf 1 mahina, naye stock ka 3 saal
    old = [s for s in symbols if s in data]
    new = [s for s in symbols if s not in data]
    jobs = [(old[i:i + 50], "1mo") for i in range(0, len(old), 50)] + \
           [(new[i:i + 50], "3y") for i in range(0, len(new), 50)]

    for n, (batch, p) in enumerate(jobs, 1):
        try:
            raw = yf.download(batch, period=p, interval="1d", group_by="ticker",
                              auto_adjust=True, progress=False, threads=True)
        except Exception as e:
            print("  batch fail:", e)
            continue
        for s in batch:
            try:
                d = raw[s] if len(batch) > 1 else raw
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if d.empty:
                    continue
                if s in data and p == "1mo":
                    d = pd.concat([data[s], d])
                    d = d[~d.index.duplicated(keep="last")].sort_index()
                data[s] = d
            except Exception:
                pass
        print(f"  batch {n}/{len(jobs)} done")
        time.sleep(1)

    with open(cache, "wb") as f:
        pickle.dump(data, f)
    return data


# ========================= 3. INDICATORS =========================
def add_indicators(d):
    d = d.copy()
    c = d["Close"]
    # RSI 14 (Wilder - Chartink jaisa)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss)
    # MACD (26,12,9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACD"] = macd
    d["SIG"] = macd.ewm(span=9, adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()
    d["VOLSMA20"] = d["Volume"].rolling(20).mean()
    return d


# ========================= 4. CONDITIONS =========================
def signals_original(d):
    """Aapka original Chartink screener."""
    s = (d["Close"] > d["Open"]) & (d["RSI"] < 30) & \
        (d["Volume"] > d["Volume"].shift(1) * 1.2) & (d["MACD"] > d["SIG"])
    for k in range(1, 6):
        s &= d["Low"] >= d["Low"].shift(k)
    return s


def signals_final(d):
    """FINAL: uptrend mein pullback ke baad bounce."""
    macd_cross = (d["MACD"] > d["SIG"]) & (d["MACD"].shift(1) <= d["SIG"].shift(1))
    return (
        (d["Close"] > d["Open"])                           # 1. green candle
        & (d["Close"] > d["EMA200"])                       # 2. bada trend upar
        & (d["RSI"] > 30) & (d["RSI"] < 50)                # 3. RSI 30-50
        & (d["RSI"].rolling(5).min() < 35)                 # 4. pichle 5 din mein RSI 35 se neeche gaya tha
        & macd_cross                                       # 5. MACD aaj cross upar
        & (d["Volume"] > d["VOLSMA20"] * 1.5)              # 6. volume 20 din avg se 1.5x
        & (d["Low"] > d["Low"].shift(1))                   # 7. higher low
        & (d["Close"] > 50) & (d["VOLSMA20"] > 100000)     # 8. liquidity
    )


VARIANTS = {"Original (aapka)": signals_original, "FINAL (naya)": signals_final}


# ========================= 5. BACKTEST =========================
def backtest(sym, d, sig_fn, start):
    sig = sig_fn(d).fillna(False).values
    o, h, l, c = d["Open"].values, d["High"].values, d["Low"].values, d["Close"].values
    idx = d.index
    trades, i, n = [], 0, len(d)
    while i < n - 1:
        if not sig[i] or idx[i] < start:
            i += 1
            continue
        trig, sl = h[i], l[i]
        j = i + 1
        # Entry sirf tab jab agle din price signal-candle ke high ke upar jaye
        if h[j] <= trig:
            i += 1
            continue
        entry = max(o[j], trig)
        risk = entry - sl
        if risk <= 0 or risk / entry > 0.08:     # 8% se bada SL -> skip
            i += 1
            continue
        target = entry + REWARD_RISK * risk
        exit_px, reason, k = None, None, j
        for k in range(j, min(j + MAX_HOLD_DAYS, n)):
            if l[k] <= sl:                       # SL pehle maano (safe side)
                exit_px, reason = (min(o[k], sl) if k > j else sl), "SL"
                break
            if h[k] >= target:
                exit_px, reason = (max(o[k], target) if k > j else target), "Target"
                break
        if exit_px is None:
            k = min(j + MAX_HOLD_DAYS - 1, n - 1)
            exit_px, reason = c[k], ("Time" if k < n - 1 else "Open")
        ret = (exit_px - entry) / entry * 100 - COST_PCT
        r_mult = (exit_px - entry) / risk - (COST_PCT / 100 * entry / risk)
        trades.append({
            "Stock": sym.replace(".NS", ""), "SignalDate": idx[i].date(),
            "EntryDate": idx[j].date(), "Entry": round(entry, 2), "SL": round(sl, 2),
            "Target": round(target, 2), "ExitDate": idx[k].date(), "Exit": round(exit_px, 2),
            "Reason": reason, "Days": k - j + 1, "Return%": round(ret, 2),
            "R": round(r_mult, 2), "PnL_Rs": round(r_mult * RISK_PER_TRADE),
        })
        i = k + 1                                # ek stock mein ek time pe ek hi trade
    return trades


def summarize(name, t):
    if t.empty:
        return f"\n=== {name} ===\nKoi trade nahi mila.\n"
    t = t[t["Reason"] != "Open"]
    wins = t[t["R"] > 0]
    loss = t[t["R"] <= 0]
    pf = wins["R"].sum() / abs(loss["R"].sum()) if len(loss) else float("inf")
    eq = t.sort_values("ExitDate")["PnL_Rs"].cumsum()
    dd = (eq - eq.cummax()).min()
    yr = t.groupby(pd.to_datetime(t["ExitDate"]).dt.year)["PnL_Rs"].agg(["count", "sum"])
    out = [
        f"\n=== {name} ===",
        f"Total trades      : {len(t)}",
        f"Win rate          : {len(wins) / len(t) * 100:.1f}%",
        f"Avg return/trade  : {t['Return%'].mean():.2f}%",
        f"Avg holding days  : {t['Days'].mean():.1f}",
        f"Profit factor     : {pf:.2f}   (1.5 se upar achha)",
        f"Total P&L         : Rs {t['PnL_Rs'].sum():,.0f}  (har trade Rs {RISK_PER_TRADE} risk)",
        f"Max drawdown      : Rs {dd:,.0f}",
        f"Exit reasons      : {t['Reason'].value_counts().to_dict()}",
        "Saal-wise (trades, P&L Rs):",
        yr.to_string(),
    ]
    return "\n".join(out) + "\n"


# ========================= 6. MAIN =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    symbols = get_symbols()
    print(f"{len(symbols)} stocks")
    data = load_data(symbols)
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=BACKTEST_YEARS)

    ind = {s: add_indicators(d) for s, d in data.items() if len(d) > 220}
    report = [f"Backtest period: {start.date()} se aaj tak | Stocks: {len(ind)}\n"]

    for name, fn in VARIANTS.items():
        all_t = []
        for s, d in ind.items():
            all_t += backtest(s, d, fn, start)
        t = pd.DataFrame(all_t)
        report.append(summarize(name, t))
        if name.startswith("FINAL") and not t.empty:
            t.sort_values("EntryDate").to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    # Aaj ke signals -> kal ki list
    today = []
    for s, d in ind.items():
        if bool(signals_final(d).iloc[-1]):
            r = d.iloc[-1]
            entry = r["High"] * 1.001
            risk = entry - r["Low"]
            qty = int(RISK_PER_TRADE / risk) if risk > 0 else 0
            today.append({
                "Stock": s.replace(".NS", ""), "Date": d.index[-1].date(),
                "Close": round(r["Close"], 2), "BuyAbove": round(entry, 2),
                "SL": round(r["Low"], 2), "Target": round(entry + REWARD_RISK * risk, 2),
                "Qty": qty, "RSI": round(r["RSI"], 1),
                "VolX": round(r["Volume"] / r["VOLSMA20"], 1),
            })
    tl = pd.DataFrame(today)
    if not tl.empty:
        tl = tl.sort_values("VolX", ascending=False)
    tl.to_csv(os.path.join(OUT_DIR, "today_signals.csv"), index=False)
    report.append(f"\nKAL ke liye stocks ({len(tl)}):\n" +
                  (tl.to_string(index=False) if not tl.empty else "Aaj koi signal nahi."))

    text = "\n".join(report)
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text)

    # Telegram pe kal ki list (agar GitHub secrets set hain)
    if tl.empty:
        msg = f"Swing Scanner {pd.Timestamp.today().date()}: aaj koi signal nahi."
    else:
        lines = [f"{r.Stock}: Buy>{r.BuyAbove} SL {r.SL} Tgt {r.Target} Qty {r.Qty}"
                 for r in tl.itertuples()]
        msg = f"Swing Scanner - kal ke stocks ({len(tl)}):\n" + "\n".join(lines)
    send_telegram(msg)


def send_telegram(msg):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg[:4000]}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=20)
        print("Telegram bheja.")
    except Exception as e:
        print("Telegram fail:", e)


if __name__ == "__main__":
    main()
