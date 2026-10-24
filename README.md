# NSE Swing Pullback Scanner

Nifty 500 stocks pe daily swing scanner + 2 saal ka backtest. Data Yahoo Finance (yfinance) se.

## Strategy (FINAL conditions)
1. Green candle (Close > Open)
2. Close > EMA 200 (bada trend upar)
3. RSI(14) 30 se 50 ke beech
4. Pichle 5 din mein RSI 35 se neeche gaya tha (pullback)
5. MACD line aaj Signal ke upar cross
6. Volume > 20 din average x 1.5
7. Higher low (Low > kal ka Low)
8. Close > 50, average volume > 1 lakh

**Trade:** Agle din signal candle ke High ke upar entry, SL = signal candle ka Low, Target = 2x risk, max 15 din hold.

## Kaise chalta hai
- GitHub Actions har Mon-Fri **4:15 PM IST** pe khud chalta hai (`.github/workflows/scan.yml`)
- Manually: **Actions** tab -> *Swing Scanner + Backtest* -> **Run workflow**
- Price data cache hota hai, isliye pehli baar ke baad sirf naya data aata hai

## Output (`output/` folder)
- `summary.txt` - backtest result (original vs FINAL)
- `trades.csv` - 2 saal ki saari trades
- `today_signals.csv` - kal ke liye stock list (BuyAbove, SL, Target, Qty)

## Telegram alert (optional)
Repo -> Settings -> Secrets and variables -> Actions mein add karo:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

## Local chalana
```
pip install -r requirements.txt
python swing_backtest.py
```

*Yeh financial advice nahi hai.*
