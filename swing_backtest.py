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
    df = pd.read_csv(local).dropna(subset=["Symbol"])
    global SECTOR_OF
    if "Industry" in df.columns:
        SECTOR_OF = {r.Symbol.strip() + ".NS": r.Industry for r in df.itertuples()}
    return [s.strip() + ".NS" for s in df["Symbol"]]


SECTOR_OF = {}


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
    d["HH252"] = h.rolling(252, min_periods=150).max()          # 52 hafte ka high
    d["HH252_PREV"] = h.shift(1).rolling(251, min_periods=150).max()
    d["RANGE20"] = h.shift(1).rolling(20).max() / l.shift(1).rolling(20).min() - 1
    # ADX 14
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr_.ewm(alpha=1 / 14, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr_
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr_
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    d["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    rng = (h - l).replace(0, np.nan)
    d["CLOSE_POS"] = (c - l) / rng                               # 1 = candle ke top pe close
    # Market (Nifty 50) data stock ki dates pe
    n = nifty.reindex(d.index, method="ffill")
    d["MKT_OK"] = n["Close"] > n["EMA200"]
    d["MKT_STRONG"] = (n["Close"] > n["EMA200"]) & (n["Close"] > n["EMA50"])
    d["RS_OK"] = d["RET63"] > n["RET63"]             # stock Nifty se zyada bhaaga
    return d


def nifty_ind(n):
    n = n.copy()
    n["EMA200"] = n["Close"].ewm(span=200, adjust=False).mean()
    n["EMA50"] = n["Close"].ewm(span=50, adjust=False).mean()
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

# Pichle run (v3) ke 2 best base combination - dono mein Market + RS filter
# Pichle run (v4) ka winner: 20D Breakout + Quality, Market+RS filter ke saath
BASES = {
    "Breakout+Quality | SL=atr1.5 | T1.5":
        {"setup": "20D-Breakout", "sl": "atr1.5", "rr": 1.5, "flags": {"quality"}},
}
# Naye sudhaar (har ek alag test hoga, phir mila ke)
IMPROVEMENTS = {
    "consol": "Breakout se pehle 20 din range 15% ke andar (shaant phase)",
    "no_resist": "52 hafte ka high hi toda (upar resistance nahi)",
    "bo_sl": "SL breakout level (20 din high) ke thoda neeche, agar ATR SL se paas",
    "no_chase": "Close EMA20 se 8% se zyada upar nahi",
    "sector": "Sector ka 3 mahine return Nifty se zyada",
    "adx": "ADX > 20 (trend mein taakat)",
}


def get_signal(d, setup, flags):
    s = SETUPS[setup](d) & d["MKT_OK"] & d["RS_OK"]
    if "quality" in flags:
        s &= (d["RSRANK"] >= 0.80) & (d["Close"] >= 0.85 * d["HH252"])
    if "strong_close" in flags:
        s &= d["CLOSE_POS"] >= 0.75
    if "consol" in flags:
        s &= d["RANGE20"] <= 0.15
    if "no_resist" in flags:
        s &= d["Close"] >= d["HH252_PREV"]
    if "no_chase" in flags:
        s &= d["Close"] <= d["EMA20"] * 1.08
    if "sector" in flags:
        s &= d["SECTOR_OK"]
    if "adx" in flags:
        s &= d["ADX"] > 20
    return s.fillna(False)


def sl_price(row, entry, sl_mode, flags=()):
    sl = row["LL5"] if sl_mode == "swing5" else entry - 1.5 * row["ATR"]
    if "bo_sl" in flags:
        bo = row["HH20"] - 0.25 * row["ATR"]
        if sl < bo < entry:
            sl = bo
    return sl


# ========================= 5. BACKTEST =========================
def trades_for_stock(sym, d, sig, start, sl_mode, rr, flags):
    o, h, l, c = (d[k].values for k in ["Open", "High", "Low", "Close"])
    ll5, atr, volx = d["LL5"].values, d["ATR"].values, d["VOLX"].values
    hh20 = d["HH20"].values
    strong = d["MKT_STRONG"].values
    idx, sig = d.index, sig.values
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
        if "bo_sl" in flags:
            bo = hh20[i] - 0.25 * atr[i]
            if sl < bo < entry:
                sl = bo
        risk = entry - sl
        if risk <= 0 or risk / entry > MAX_SL_PCT:
            i += 1
            continue
        target, half_lvl = entry + rr * risk, entry + risk
        cur_sl, half_done, booked = sl, False, 0.0        # booked = pehle aadhe ka R
        exit_px = reason = None
        for k in range(j, min(j + 15, n)):
            if l[k] <= cur_sl:                             # SL pehle maano (safe side)
                exit_px = min(o[k], cur_sl) if k > j else cur_sl
                reason = "BE" if half_done else "SL"
                break
            if "partial" in flags and not half_done and h[k] >= half_lvl:
                half_done, booked, cur_sl = True, 0.5 * 1.0, entry
            if h[k] >= target:
                exit_px, reason = (max(o[k], target) if k > j else target), "Target"
                break
            if "timestop" in flags and not half_done and k - j + 1 >= 7 \
                    and c[k] < entry + 0.5 * risk:
                exit_px, reason = c[k], "TimeStop"
                break
        if exit_px is None:
            k = min(j + 14, n - 1)
            exit_px, reason = c[k], ("Time" if k < n - 1 else "Open")
        rest = (exit_px - entry) / risk
        r_mult = (booked + 0.5 * rest if half_done else rest) - COST_PCT / 100 * entry / risk
        mult = 1.0 if ("size" not in flags or strong[i]) else 0.5
        out.append({"Stock": sym.replace(".NS", ""), "SignalDate": idx[i], "EntryDate": idx[j],
                    "Entry": round(entry, 2), "SL": round(sl, 2), "Target": round(target, 2),
                    "ExitDate": idx[k], "Exit": round(exit_px, 2), "Reason": reason,
                    "Days": k - j + 1, "R": round(r_mult, 3), "Size": mult,
                    "VolX": round(volx[i], 2)})
        i = k + 1
    return out


def apply_portfolio_limit(t):
    """Ek time pe max MAX_POSITIONS trades. Ek din zyada signal -> zyada VolX wale pehle."""
    if t.empty:
        return t
    t = t.sort_values(["EntryDate", "VolX"], ascending=[True, False])
    taken, open_exits = [], []
    for r in t.itertuples():
        open_exits = [x for x in open_exits if x >= r.EntryDate]
        if len(open_exits) < MAX_POSITIONS:
            taken.append(r.Index)
            open_exits.append(r.ExitDate)
    t = t.loc[taken].copy()
    t["PnL_Rs"] = (t["R"] * t["Size"] * RISK_PER_TRADE).round()
    return t


def stats(t):
    t = t[t["Reason"] != "Open"] if not t.empty else t
    if t.empty:
        return {"Trades": 0, "Win%": 0.0, "PF": 0.0, "PnL": 0, "MaxDD": 0}
    w, ls = t[t["PnL_Rs"] > 0], t[t["PnL_Rs"] <= 0]
    pf = w["PnL_Rs"].sum() / abs(ls["PnL_Rs"].sum()) if ls["PnL_Rs"].sum() else 99
    eq = t.sort_values("ExitDate")["PnL_Rs"].cumsum()
    return {"Trades": len(t), "Win%": round(len(w) / len(t) * 100, 1), "PF": round(pf, 2),
            "PnL": int(t["PnL_Rs"].sum()), "MaxDD": int((eq - eq.cummax()).min())}


def run(ind, start, base, flags):
    flags = set(flags) | base.get("flags", set())
    sigs_tr = []
    for s, d in ind.items():
        sig = get_signal(d, base["setup"], flags)
        sigs_tr += trades_for_stock(s, d, sig, start, base["sl"], base["rr"], flags)
    return apply_portfolio_limit(pd.DataFrame(sigs_tr))


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
    # Relative strength rank: har din sab stocks mein 3 mahine return ka percentile
    rank = pd.DataFrame({s: d["RET63"] for s, d in ind.items()}).rank(axis=1, pct=True)
    for s, d in ind.items():
        d["RSRANK"] = rank[s].reindex(d.index)
    # Sector strength: sector ke stocks ka average 3 mahine return > Nifty
    rets = pd.DataFrame({s: d["RET63"] for s, d in ind.items()})
    n63 = nifty["RET63"].reindex(rets.index, method="ffill")
    sec_ok = {}
    for sec in set(SECTOR_OF.get(s, "NA") for s in ind):
        cols = [s for s in ind if SECTOR_OF.get(s, "NA") == sec]
        sec_ok[sec] = rets[cols].mean(axis=1) > n63
    for s, d in ind.items():
        d["SECTOR_OK"] = sec_ok[SECTOR_OF.get(s, "NA")].reindex(d.index).fillna(False)
    print(f"{len(ind)} stocks ready")

    def period_ret(df, a, b):
        x = df["Close"][(df.index >= a) & (df.index <= b)]
        return (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 5 else np.nan

    def split_stats(t):
        if t.empty:
            return stats(t), stats(t), stats(t)
        return stats(t[t["EntryDate"] < split]), stats(t[t["EntryDate"] >= split]), stats(t)

    def row(base_name, label, t):
        a, b, f = split_stats(t)
        return {"Base": base_name, "Variant": label,
                "Trades": f["Trades"], "Win%": f["Win%"], "PF": f["PF"], "PnL_2y": f["PnL"],
                "MaxDD": f["MaxDD"], "TrainPnL": a["PnL"], "TestTrades": b["Trades"],
                "TestPnL": b["PnL"]}

    rows, store = [], {}
    for bname, base in BASES.items():
        # 1) Base + har sudhaar alag
        base_t = run(ind, start, base, set())
        store[(bname, "Base")] = (base_t, set(base["flags"]))
        rows.append(row(bname, "Base", base_t))
        base_train = split_stats(base_t)[0]["PnL"]
        helped = set()
        for flag in IMPROVEMENTS:
            t = run(ind, start, base, {flag})
            store[(bname, "+" + flag)] = (t, {flag} | base["flags"])
            rows.append(row(bname, "+" + flag, t))
            if split_stats(t)[0]["PnL"] > base_train:       # sirf TRAIN dekh ke faisla
                helped.add(flag)
            print("done:", bname, flag)
        # 2) Sab ek saath
        t = run(ind, start, base, set(IMPROVEMENTS))
        store[(bname, "ALL")] = (t, set(IMPROVEMENTS) | base["flags"])
        rows.append(row(bname, "ALL", t))
        # 3) SMART = sirf woh sudhaar jo TRAIN mein kaam aaye
        if helped:
            lbl = "SMART(" + "+".join(sorted(helped)) + ")"
            t = run(ind, start, base, helped)
            store[(bname, lbl)] = (t, helped | base["flags"])
            rows.append(row(bname, lbl, t))

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)

    # Best: TRAIN PnL se chuno (min 30 trades total), TEST se jaancho
    cand = cmp_df[cmp_df["Trades"] >= MIN_TRAIN_TRADES]
    best = (cand if not cand.empty else cmp_df).sort_values("TrainPnL", ascending=False).iloc[0]
    passed = bool(best["TrainPnL"] > 0 and best["TestPnL"] >= 0 and best["PF"] > 1.1)
    bkey = (best["Base"], best["Variant"])
    bt, bflags = store[bkey]
    if not bt.empty:
        bt.sort_values("EntryDate").to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    rep = [f"Period: {start.date()} -> {today.date()} | Stocks: {len(ind)} | "
           f"TRAIN: {start.date()}-{split.date()} | TEST: {split.date()}-{today.date()}",
           f"Rules: Market+RS filter hamesha, max {MAX_POSITIONS} trades ek saath, "
           f"Rs {RISK_PER_TRADE} risk/trade, cost {COST_PCT}%\n",
           "MARKET (buy & hold):"]
    for k, a, b in [("2 saal", start, today), ("TRAIN", start, split), ("TEST", split, today)]:
        rep.append(f"  {k:7s}: Nifty 50 {period_ret(nifty, a, b):+.1f}% | Nifty 500 avg "
                   f"{np.nanmean([period_ret(d, a, b) for d in ind.values()]):+.1f}%")
    rep.append("\nSUDHAAR:")
    for k, v in IMPROVEMENTS.items():
        rep.append(f"  +{k:13s} = {v}")
    rep.append("\nRESULT (har sudhaar ka asar):")
    rep.append(cmp_df.to_string(index=False))
    rep.append(f"\nBEST (TRAIN se chuna): {bkey[0]} -> {bkey[1]}")
    rep.append(f"  2 saal: {best['Trades']} trades, win {best['Win%']}%, PF {best['PF']}, "
               f"P&L Rs {best['PnL_2y']:,}, MaxDD Rs {best['MaxDD']:,}")
    rep.append(f"  TRAIN P&L Rs {best['TrainPnL']:,} | TEST: {best['TestTrades']} trades, "
               f"P&L Rs {best['TestPnL']:,}")
    rep.append("  RESULT: " + ("PASS" if passed else "FAIL - live mat lagao"))
    if not bt.empty:
        rep.append(f"  Exit reasons: {bt['Reason'].value_counts().to_dict()}")

    # ---- Kal ki list ----
    held = set()
    if os.path.exists(OPEN_POS_FILE):
        held = {x.strip().upper() for x in pd.read_csv(OPEN_POS_FILE).iloc[:, 0].dropna()}
    base = BASES[bkey[0]]
    sig_rows = []
    if passed:
        for s, d in ind.items():
            name = s.replace(".NS", "")
            if name in held or not bool(get_signal(d, base["setup"], bflags).iloc[-1]):
                continue
            r = d.iloc[-1]
            entry = r["High"] * 1.001
            sl = sl_price(r, entry, base["sl"], bflags)
            risk = entry - sl
            if risk <= 0 or risk / entry > MAX_SL_PCT:
                continue
            size = 1.0 if ("size" not in bflags or r["MKT_STRONG"]) else 0.5
            sig_rows.append({"Stock": name, "Date": d.index[-1].date(),
                             "BuyAbove": round(entry, 2), "SL": round(sl, 2),
                             "Half@1R": round(entry + risk, 2) if "partial" in bflags else "-",
                             "Target": round(entry + base["rr"] * risk, 2),
                             "Qty": int(RISK_PER_TRADE * size / risk), "VolX": round(r["VOLX"], 1)})
    tl = pd.DataFrame(sig_rows)
    if not tl.empty:
        tl = tl.sort_values("VolX", ascending=False).head(MAX_POSITIONS)
    tl.to_csv(os.path.join(OUT_DIR, "today_signals.csv"), index=False)
    rep.append(f"\nKAL ke liye stocks ({len(tl)}):")
    rep.append(tl.to_string(index=False) if not tl.empty else
               ("Aaj koi signal nahi." if passed else "Best setup fail, list nahi bheji."))

    text = "\n".join(rep)
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text)

    if not passed:
        msg = "Swing Scanner: best setup backtest mein fail. Aaj trade mat lo."
    elif tl.empty:
        msg = f"Swing Scanner {today.date()}: aaj koi signal nahi."
    else:
        msg = f"Swing Scanner [{bkey[0]} {bkey[1]}] kal ke stocks:\n" + "\n".join(
            f"{r['Stock']}: Buy>{r['BuyAbove']} SL {r['SL']} Half {r['Half@1R']} "
            f"Tgt {r['Target']} Qty {r['Qty']}" for r in tl.to_dict("records"))
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
