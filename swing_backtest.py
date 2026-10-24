"""
Swing Scanner + Backtest (Nifty 500, Daily) - v3
================================================
Kaise chalayein:
    pip install -r requirements.txt
    python swing_backtest.py

Kya karta hai:
  1. Nifty 500 + Nifty 50 index ka data (cache ke saath) laata hai
  2. 3 setups x 2 filter x 2 SL x 3 exit = 36 combination backtest karta hai
  3. Real jaisa: ek time pe max 5 trades, ek stock mein ek hi trade
  4. Walk-forward: pehle 18 mahine (TRAIN) pe best chuna, aakhri 6 mahine (TEST) pe jaancha
  5. Best combo dono jagah profit mein ho tabhi kal ki stock list bhejta hai

Output (output/ folder):
    summary.txt         -> poora result simple bhasha mein
    comparison.csv      -> 36 combination ka table
    trades.csv          -> best combination ki saari trades
    today_signals.csv   -> kal ke liye stocks
open_positions.csv (repo root) -> jo stock aapne kharida hai, uska naam likho; woh list mein nahi aayega
"""

import os
import time
import pickle
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import yfinance as yf

# ------------------------- SETTINGS (yahan badlo) -------------------------
BACKTEST_YEARS = 2
TEST_MONTHS = 6              # aakhri itne mahine "unseen" test ke liye
RISK_PER_TRADE = 1000        # har trade mein kitna Rs risk
MAX_POSITIONS = 5            # ek time pe max open trades
COST_PCT = 0.2               # brokerage + slippage, round trip (%)
MAX_SL_PCT = 0.10            # 10% se door SL ho to trade skip
MIN_TRAIN_TRADES = 30        # train mein isse kam trades -> bharosemand nahi
DATA_DIR = "data"
OUT_DIR = "output"
OPEN_POS_FILE = "open_positions.csv"
INDEX = "^NSEI"              # Nifty 50
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
# --------------------------------------------------------------------------


# ========================= 1. STOCK LIST =========================
def get_symbols():
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
        print("Pura 3 saal ka data download ho raha hai...")

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
                try:
                    d = raw[s]
                except KeyError:
                    d = raw
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if d.empty:
                    continue
                d.index = pd.to_datetime(d.index).tz_localize(None)
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
def add_indicators(d, nifty):
    d = d.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / loss)
    d["EMA20"] = c.ewm(span=20, adjust=False).mean()
    d["EMA50"] = c.ewm(span=50, adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()
    d["VOLSMA20"] = d["Volume"].rolling(20).mean()
    d["VOLX"] = d["Volume"] / d["VOLSMA20"]
    d["HH20"] = h.shift(1).rolling(20).max()
    d["LL5"] = l.rolling(5).min()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["RET63"] = c / c.shift(63) - 1
    # Market (Nifty 50) data stock ki dates pe
    n = nifty.reindex(d.index, method="ffill")
    d["MKT_OK"] = n["Close"] > n["EMA200"]
    d["RS_OK"] = d["RET63"] > n["RET63"]             # stock Nifty se zyada bhaaga
    return d


def nifty_ind(n):
    n = n.copy()
    n["EMA200"] = n["Close"].ewm(span=200, adjust=False).mean()
    n["RET63"] = n["Close"] / n["Close"].shift(63) - 1
    return n


# ========================= 4. SETUPS =========================
def liquid(d):
    return (d["Close"] > 50) & (d["VOLSMA20"] > 100000)


def s_ema20_pullback(d):
    """C. Uptrend mein EMA20 tak pullback, phir bounce."""
    return ((d["Close"] > d["EMA50"]) & (d["EMA50"] > d["EMA200"])
            & (d["Low"].rolling(3).min() <= d["EMA20"] * 1.01)
            & (d["Close"] > d["EMA20"]) & (d["Close"] > d["Open"])
            & (d["Close"] > d["High"].shift(1))
            & (d["RSI"] > 40) & (d["RSI"] < 65)
            & (d["VOLX"] > 1.2) & liquid(d))


def s_rsi40_recovery(d):
    """D. Uptrend mein RSI 40 ke neeche gaya, aaj wapas upar."""
    return ((d["Close"] > d["EMA200"])
            & (d["RSI"] > 40) & (d["RSI"].shift(1) <= 40)
            & (d["Close"] > d["Open"])
            & (d["VOLX"] > 1.2) & liquid(d))


def s_breakout(d):
    """E. 20 din ka high breakout, volume ke saath."""
    return ((d["Close"] > d["EMA200"]) & (d["EMA50"] > d["EMA200"])
            & (d["Close"] > d["HH20"]) & (d["Close"] > d["Open"])
            & (d["RSI"] > 55) & (d["RSI"] < 75)
            & (d["VOLX"] > 1.5) & liquid(d))


SETUPS = {"EMA20-Pullback": s_ema20_pullback,
          "RSI40-Recovery": s_rsi40_recovery,
          "20D-Breakout": s_breakout}
FILTERS = ["None", "Mkt+RS"]          # Mkt+RS = Nifty > 200EMA aur stock Nifty se strong
SLS = ["swing5", "atr1.5"]            # 5 din ka low  /  entry - 1.5 x ATR
EXITS = ["T1.5", "T2", "Trail"]       # 1.5x target / 2x target / EMA20 trailing


def get_signal(d, setup, flt):
    s = SETUPS[setup](d)
    if flt == "Mkt+RS":
        s = s & d["MKT_OK"] & d["RS_OK"]
    return s.fillna(False)


def sl_price(row, entry, sl_mode):
    return row["LL5"] if sl_mode == "swing5" else entry - 1.5 * row["ATR"]


# ========================= 5. BACKTEST =========================
def trades_for_stock(sym, d, sig, start, sl_mode, exit_mode):
    """Ek stock ki saari trades (ek time pe ek hi). Portfolio limit baad mein lagti hai."""
    o, h, l, c = (d[k].values for k in ["Open", "High", "Low", "Close"])
    ema20, ll5, atr, volx = d["EMA20"].values, d["LL5"].values, d["ATR"].values, d["VOLX"].values
    idx, sig = d.index, sig.values
    max_hold = 30 if exit_mode == "Trail" else 15
    rr = {"T1.5": 1.5, "T2": 2.0}.get(exit_mode)
    out, i, n = [], 0, len(d)
    while i < n - 1:
        if not sig[i] or idx[i] < start:
            i += 1
            continue
        trig, j = h[i], i + 1
        if h[j] <= trig:                                   # agle din high nahi toda -> no entry
            i += 1
            continue
        entry = max(o[j], trig)
        sl = ll5[i] if sl_mode == "swing5" else entry - 1.5 * atr[i]
        risk = entry - sl
        if risk <= 0 or risk / entry > MAX_SL_PCT:
            i += 1
            continue
        target = entry + rr * risk if rr else None
        exit_px = reason = None
        for k in range(j, min(j + max_hold, n)):
            if l[k] <= sl:
                exit_px, reason = (min(o[k], sl) if k > j else sl), "SL"
                break
            if target and h[k] >= target:
                exit_px, reason = (max(o[k], target) if k > j else target), "Target"
                break
            if exit_mode == "Trail" and k > j and c[k] < ema20[k]:
                exit_px, reason = c[k], "Trail"
                break
        if exit_px is None:
            k = min(j + max_hold - 1, n - 1)
            exit_px, reason = c[k], ("Time" if k < n - 1 else "Open")
        r_mult = (exit_px - entry) / risk - (COST_PCT / 100 * entry / risk)
        out.append({"Stock": sym.replace(".NS", ""), "SignalDate": idx[i], "EntryDate": idx[j],
                    "Entry": round(entry, 2), "SL": round(sl, 2), "ExitDate": idx[k],
                    "Exit": round(exit_px, 2), "Reason": reason, "Days": k - j + 1,
                    "Return%": round((exit_px - entry) / entry * 100 - COST_PCT, 2),
                    "R": round(r_mult, 3), "VolX": round(volx[i], 2)})
        i = k + 1
    return out


def apply_portfolio_limit(t):
    """Ek time pe max MAX_POSITIONS trades. Ek din zyada signal -> zyada VolX wale pehle."""
    if t.empty:
        return t
    t = t.sort_values(["EntryDate", "VolX"], ascending=[True, False])
    taken, open_exits = [], []
    for r in t.itertuples():
        open_exits = [x for x in open_exits if x >= r.EntryDate]   # jo band ho gayi hata do
        if len(open_exits) < MAX_POSITIONS:
            taken.append(r.Index)
            open_exits.append(r.ExitDate)
    t = t.loc[taken].copy()
    t["PnL_Rs"] = (t["R"] * RISK_PER_TRADE).round()
    return t


def stats(t):
    t = t[t["Reason"] != "Open"] if not t.empty else t
    if t.empty:
        return {"Trades": 0, "Win%": 0, "PF": 0, "PnL_Rs": 0, "MaxDD_Rs": 0}
    w, ls = t[t["R"] > 0], t[t["R"] <= 0]
    pf = w["R"].sum() / abs(ls["R"].sum()) if len(ls) else 99
    eq = t.sort_values("ExitDate")["PnL_Rs"].cumsum()
    return {"Trades": len(t), "Win%": round(len(w) / len(t) * 100, 1), "PF": round(pf, 2),
            "PnL_Rs": int(t["PnL_Rs"].sum()), "MaxDD_Rs": int((eq - eq.cummax()).min())}


# ========================= 6. MAIN =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    symbols = get_symbols()
    data = load_data(symbols + [INDEX])
    if INDEX not in data:
        raise SystemExit("Nifty 50 data nahi mila, dobara chalao.")
    nifty = nifty_ind(data[INDEX])

    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=BACKTEST_YEARS)
    split = today - pd.DateOffset(months=TEST_MONTHS)

    ind = {s: add_indicators(d, nifty) for s, d in data.items()
           if s != INDEX and len(d) > 220}
    print(f"{len(ind)} stocks ready")

    # ---- Benchmark: agar sab stocks/Nifty simple kharid ke rakhte ----
    def period_ret(df, a, b):
        x = df["Close"][(df.index >= a) & (df.index <= b)]
        return (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 5 else np.nan
    bench = {}
    for name, a, b in [("2 saal", start, today), ("TRAIN", start, split), ("TEST", split, today)]:
        bench[name] = (period_ret(nifty, a, b),
                       np.nanmean([period_ret(d, a, b) for d in ind.values()]))
    mkt_up = (nifty["Close"] > nifty["EMA200"])[nifty.index >= start].mean() * 100

    # ---- 36 combinations ----
    rows, store = [], {}
    for setup in SETUPS:
        for flt in FILTERS:
            sigs = {s: get_signal(d, setup, flt) for s, d in ind.items()}
            for sl in SLS:
                for ex in EXITS:
                    tr = []
                    for s, d in ind.items():
                        tr += trades_for_stock(s, d, sigs[s], start, sl, ex)
                    t = apply_portfolio_limit(pd.DataFrame(tr))
                    key = f"{setup} | {flt} | SL={sl} | {ex}"
                    store[key] = t
                    tr_t = t[t["EntryDate"] < split] if not t.empty else t
                    te_t = t[t["EntryDate"] >= split] if not t.empty else t
                    a, b, full = stats(tr_t), stats(te_t), stats(t)
                    rows.append({"Combo": key,
                                 "TrainTrades": a["Trades"], "TrainWin%": a["Win%"],
                                 "TrainPF": a["PF"], "TrainPnL": a["PnL_Rs"],
                                 "TestTrades": b["Trades"], "TestWin%": b["Win%"],
                                 "TestPF": b["PF"], "TestPnL": b["PnL_Rs"],
                                 "Full2yPnL": full["PnL_Rs"], "MaxDD": full["MaxDD_Rs"]})
                    print("done:", key)

    cmp_df = pd.DataFrame(rows).sort_values("TrainPF", ascending=False)
    cmp_df.to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)

    # ---- Best sirf TRAIN se chuno, TEST se jaancho ----
    cand = cmp_df[cmp_df["TrainTrades"] >= MIN_TRAIN_TRADES]
    best = cand.iloc[0] if not cand.empty else cmp_df.iloc[0]
    passed = bool(best["TrainPnL"] > 0 and best["TestPnL"] > 0 and best["TestPF"] > 1.1)
    best_key = best["Combo"]
    bt = store[best_key]
    if not bt.empty:
        bt.sort_values("EntryDate").to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    rep = [f"Period: {start.date()} -> {today.date()} | Stocks: {len(ind)} | "
           f"TRAIN: {start.date()}-{split.date()} | TEST: {split.date()}-{today.date()}",
           f"Rules: max {MAX_POSITIONS} trades ek saath, Rs {RISK_PER_TRADE} risk/trade, cost {COST_PCT}%\n",
           "MARKET KAISA THA (simple buy & hold):"]
    for k, (nr, sr) in bench.items():
        rep.append(f"  {k:7s}: Nifty 50 {nr:+.1f}% | Nifty 500 stocks avg {sr:+.1f}%")
    rep.append(f"  Nifty 200 EMA ke upar tha: {mkt_up:.0f}% din\n")
    rep.append("TOP 15 COMBINATIONS (TRAIN PF ke hisaab se):")
    rep.append(cmp_df.head(15).to_string(index=False))
    rep.append(f"\nBEST (train se chuna): {best_key}")
    rep.append(f"  TRAIN: {best['TrainTrades']} trades, win {best['TrainWin%']}%, "
               f"PF {best['TrainPF']}, P&L Rs {best['TrainPnL']:,}")
    rep.append(f"  TEST : {best['TestTrades']} trades, win {best['TestWin%']}%, "
               f"PF {best['TestPF']}, P&L Rs {best['TestPnL']:,}")
    rep.append("  RESULT: " + ("PASS - test mein bhi profit" if passed else
                               "FAIL - test mein profit nahi, live mat lagao"))
    if not bt.empty:
        rep.append(f"  Exit reasons: {bt['Reason'].value_counts().to_dict()}")

    # ---- Kal ki list (sirf PASS hone par) ----
    held = set()
    if os.path.exists(OPEN_POS_FILE):
        held = {x.strip().upper() for x in pd.read_csv(OPEN_POS_FILE).iloc[:, 0].dropna()}
    setup, flt, sl_mode, ex = [x.strip() for x in best_key.split("|")]
    sl_mode = sl_mode.replace("SL=", "")
    rr = {"T1.5": 1.5, "T2": 2.0}.get(ex)
    sig_rows = []
    if passed:
        for s, d in ind.items():
            name = s.replace(".NS", "")
            if name in held or not bool(get_signal(d, setup, flt).iloc[-1]):
                continue
            r = d.iloc[-1]
            entry = r["High"] * 1.001
            sl = sl_price(r, entry, sl_mode)
            risk = entry - sl
            if risk <= 0 or risk / entry > MAX_SL_PCT:
                continue
            sig_rows.append({"Stock": name, "Date": d.index[-1].date(),
                             "BuyAbove": round(entry, 2), "SL": round(sl, 2),
                             "Target": round(entry + rr * risk, 2) if rr else "Trail EMA20",
                             "Qty": int(RISK_PER_TRADE / risk), "VolX": round(r["VOLX"], 1)})
    tl = pd.DataFrame(sig_rows)
    if not tl.empty:
        tl = tl.sort_values("VolX", ascending=False).head(MAX_POSITIONS)
    tl.to_csv(os.path.join(OUT_DIR, "today_signals.csv"), index=False)
    rep.append(f"\nKAL ke liye stocks ({len(tl)}):")
    rep.append(tl.to_string(index=False) if not tl.empty else
               ("Aaj koi signal nahi." if passed else "Best setup test mein fail, list nahi bheji."))

    text = "\n".join(rep)
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text)

    if not passed:
        msg = "Swing Scanner: best setup TEST mein fail. Aaj trade mat lo."
    elif tl.empty:
        msg = f"Swing Scanner {today.date()}: aaj koi signal nahi."
    else:
        msg = f"Swing Scanner [{best_key}] kal ke stocks:\n" + "\n".join(
            f"{r.Stock}: Buy>{r.BuyAbove} SL {r.SL} Tgt {r.Target} Qty {r.Qty}"
            for r in tl.itertuples())
    send_telegram(msg)


def send_telegram(msg):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
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
