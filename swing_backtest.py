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
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()
    d["VOLSMA20"] = d["Volume"].rolling(20).mean()
    d["HH20"] = d["High"].shift(1).rolling(20).max()     # pichle 20 din ka high (aaj ke bina)
    d["LL5"] = d["Low"].rolling(5).min()                 # pichle 5 din ka lowest low (aaj samet)
    return d


# ========================= 4. SETUPS (sab ek saath test honge) =========================
def liquid(d):
    return (d["Close"] > 50) & (d["VOLSMA20"] > 100000)


def s_original(d):
    """A. Aapka original Chartink screener."""
    s = (d["Close"] > d["Open"]) & (d["RSI"] < 30) & \
        (d["Volume"] > d["Volume"].shift(1) * 1.2) & (d["MACD"] > d["SIG"])
    for k in range(1, 6):
        s &= d["Low"] >= d["Low"].shift(k)
    return s


def s_final(d):
    """B. Pehla FINAL: RSI 30-50 + MACD cross + volume 1.5x."""
    macd_cross = (d["MACD"] > d["SIG"]) & (d["MACD"].shift(1) <= d["SIG"].shift(1))
    return ((d["Close"] > d["Open"]) & (d["Close"] > d["EMA200"])
            & (d["RSI"] > 30) & (d["RSI"] < 50) & (d["RSI"].rolling(5).min() < 35)
            & macd_cross & (d["Volume"] > d["VOLSMA20"] * 1.5)
            & (d["Low"] > d["Low"].shift(1)) & liquid(d))


def s_ema20_pullback(d):
    """C. Uptrend mein EMA20 tak pullback, phir green candle se bounce."""
    return ((d["Close"] > d["EMA50"]) & (d["EMA50"] > d["EMA200"])        # strong uptrend
            & (d["Low"].rolling(3).min() <= d["EMA20"] * 1.01)             # 3 din mein EMA20 chhua
            & (d["Close"] > d["EMA20"]) & (d["Close"] > d["Open"])         # upar band, green
            & (d["Close"] > d["High"].shift(1))                            # kal ka high toda
            & (d["RSI"] > 40) & (d["RSI"] < 65)
            & (d["Volume"] > d["VOLSMA20"] * 1.2) & liquid(d))


def s_rsi40_recovery(d):
    """D. Uptrend mein RSI 40 ke neeche gaya, aaj wapas 40 ke upar."""
    return ((d["Close"] > d["EMA200"])
            & (d["RSI"] > 40) & (d["RSI"].shift(1) <= 40)                  # RSI crossed above 40
            & (d["RSI"].rolling(5).min() < 40)
            & (d["Close"] > d["Open"])
            & (d["Volume"] > d["VOLSMA20"] * 1.2) & liquid(d))


def s_breakout(d):
    """E. 20 din ka high breakout, volume ke saath."""
    return ((d["Close"] > d["EMA200"]) & (d["EMA50"] > d["EMA200"])
            & (d["Close"] > d["HH20"])                                     # 20 din ka high toda
            & (d["Close"] > d["Open"])
            & (d["RSI"] > 55) & (d["RSI"] < 75)
            & (d["Volume"] > d["VOLSMA20"] * 1.5) & liquid(d))


SETUPS = {
    "A Original": s_original,
    "B Final-v1": s_final,
    "C EMA20 Pullback": s_ema20_pullback,
    "D RSI40 Recovery": s_rsi40_recovery,
    "E 20D Breakout": s_breakout,
}
SL_MODES = {"candle": "Signal candle ka Low", "swing5": "Pichle 5 din ka Lowest Low"}
MIN_TRADES = 40            # isse kam trades wala result bharosemand nahi


# ========================= 5. BACKTEST =========================
def backtest(sym, d, sig_fn, start, sl_mode):
    sig = sig_fn(d).fillna(False).values
    o, h, l, c = d["Open"].values, d["High"].values, d["Low"].values, d["Close"].values
    slv = (d["Low"] if sl_mode == "candle" else d["LL5"]).values
    idx = d.index
    trades, i, n = [], 0, len(d)
    while i < n - 1:
        if not sig[i] or idx[i] < start:
            i += 1
            continue
        trig, sl = h[i], slv[i]
        j = i + 1
        # Entry sirf tab jab agle din price signal-candle ke high ke upar jaye
        if h[j] <= trig:
            i += 1
            continue
        entry = max(o[j], trig)
        risk = entry - sl
        if risk <= 0 or risk / entry > 0.10:     # 10% se bada SL -> skip
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


def stats(t):
    t = t[t["Reason"] != "Open"] if not t.empty else t
    if t.empty:
        return {"Trades": 0, "Win%": 0, "AvgRet%": 0, "PF": 0, "PnL_Rs": 0, "MaxDD_Rs": 0, "AvgDays": 0}
    wins, loss = t[t["R"] > 0], t[t["R"] <= 0]
    pf = wins["R"].sum() / abs(loss["R"].sum()) if len(loss) else 99
    eq = t.sort_values("ExitDate")["PnL_Rs"].cumsum()
    return {"Trades": len(t), "Win%": round(len(wins) / len(t) * 100, 1),
            "AvgRet%": round(t["Return%"].mean(), 2), "PF": round(pf, 2),
            "PnL_Rs": int(t["PnL_Rs"].sum()), "MaxDD_Rs": int((eq - eq.cummax()).min()),
            "AvgDays": round(t["Days"].mean(), 1)}


def detail(name, t):
    t = t[t["Reason"] != "Open"]
    yr = t.groupby(pd.to_datetime(t["ExitDate"]).dt.year)["PnL_Rs"].agg(["count", "sum"])
    return (f"\n=== BEST: {name} ===\n"
            f"Exit reasons: {t['Reason'].value_counts().to_dict()}\n"
            f"Saal-wise (trades, P&L Rs):\n{yr.to_string()}\n")


# ========================= 6. MAIN =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    symbols = get_symbols()
    print(f"{len(symbols)} stocks")
    data = load_data(symbols)
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=BACKTEST_YEARS)

    ind = {s: add_indicators(d) for s, d in data.items() if len(d) > 220}
    report = [f"Backtest period: {start.date()} se aaj tak | Stocks: {len(ind)}",
              f"Rules: entry = signal High ke upar, target = {REWARD_RISK}x risk, "
              f"max {MAX_HOLD_DAYS} din, cost {COST_PCT}%, har trade Rs {RISK_PER_TRADE} risk\n"]

    rows, all_trades = [], {}
    for name, fn in SETUPS.items():
        for sl_mode in SL_MODES:
            tr = []
            for s, d in ind.items():
                tr += backtest(s, d, fn, start, sl_mode)
            t = pd.DataFrame(tr)
            key = f"{name} | SL={sl_mode}"
            all_trades[key] = (t, fn, sl_mode)
            rows.append({"Setup": key, **stats(t)})

    cmp_df = pd.DataFrame(rows).sort_values("PF", ascending=False)
    cmp_df.to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)
    report.append("SAB SETUPS KA COMPARISON (PF = profit factor, 1.5+ achha):")
    report.append(cmp_df.to_string(index=False))

    ok = cmp_df[(cmp_df["Trades"] >= MIN_TRADES) & (cmp_df["PnL_Rs"] > 0)]
    if ok.empty:
        report.append(f"\nKoi setup {MIN_TRADES}+ trades ke saath profit mein nahi. Live mat lagao.")
        best_key = cmp_df.iloc[0]["Setup"]
    else:
        best_key = ok.iloc[0]["Setup"]
    best_t, best_fn, best_sl = all_trades[best_key]
    if not best_t.empty:
        best_t.sort_values("EntryDate").to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
        report.append(detail(best_key, best_t))

    # Aaj ke signals (BEST setup se) -> kal ki list
    today = []
    for s, d in (ind.items() if not ok.empty else []):
        if bool(best_fn(d).fillna(False).iloc[-1]):
            r = d.iloc[-1]
            sl = r["Low"] if best_sl == "candle" else r["LL5"]
            entry = r["High"] * 1.001
            risk = entry - sl
            if risk <= 0 or risk / entry > 0.10:
                continue
            today.append({
                "Stock": s.replace(".NS", ""), "Date": d.index[-1].date(),
                "Close": round(r["Close"], 2), "BuyAbove": round(entry, 2),
                "SL": round(sl, 2), "Target": round(entry + REWARD_RISK * risk, 2),
                "Qty": int(RISK_PER_TRADE / risk), "RSI": round(r["RSI"], 1),
                "VolX": round(r["Volume"] / r["VOLSMA20"], 1),
            })
    tl = pd.DataFrame(today)
    if not tl.empty:
        tl = tl.sort_values("VolX", ascending=False)
    tl.to_csv(os.path.join(OUT_DIR, "today_signals.csv"), index=False)
    report.append(f"\nKAL ke liye stocks [{best_key}] ({len(tl)}):\n" +
                  (tl.to_string(index=False) if not tl.empty else
                   ("Aaj koi signal nahi." if not ok.empty else
                    "Koi setup backtest mein profit mein nahi, isliye list nahi bheji.")))

    text = "\n".join(report)
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text)

    # Telegram pe kal ki list (agar GitHub secrets set hain)
    if tl.empty:
        msg = (f"Swing Scanner {pd.Timestamp.today().date()}: aaj koi signal nahi."
               if not ok.empty else "Swing Scanner: koi setup backtest mein profitable nahi. Trade mat lo.")
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
