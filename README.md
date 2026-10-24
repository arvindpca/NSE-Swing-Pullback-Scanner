# NSE Swing Pullback Scanner

Nifty 500 stocks pe daily swing scanner + 2 saal ka backtest. Data Yahoo Finance (yfinance) se.

## Strategy (v3)
Script 36 combination test karta hai:
- **Setups:** EMA20 Pullback, RSI 40 Recovery, 20 din Breakout
- **Filter:** koi nahi / Market (Nifty 50 > 200 EMA) + Relative Strength (stock 3 mahine mein Nifty se zyada bhaaga)
- **SL:** 5 din ka lowest low / entry - 1.5 x ATR
- **Exit:** 1.5x target / 2x target / EMA20 trailing

Real jaisa: ek time pe max 5 trades, ek stock mein ek hi trade.
Walk-forward: pehle 18 mahine pe best chuna jaata hai, aakhri 6 mahine pe test. **Test pass hone par hi** kal ki list bheji jaati hai.

## Open positions
`open_positions.csv` mein jo stock aapne kharida hai uska naam (jaise `CYIENT`) likho. Woh stock list mein dobara nahi aayega. Bechne ke baad naam hata do.

## Kaise chalta hai
- GitHub Actions har Mon-Fri **4:15 PM IST** pe khud chalta hai (`.github/workflows/scan.yml`)
- Manually: **Actions** tab -> *Swing Scanner + Backtest* -> **Run workflow**
- Price data cache hota hai, isliye pehli baar ke baad sirf naya data aata hai

## Output (`output/` folder)
- `summary.txt` - poora result (market, top combinations, best ka train/test)
- `comparison.csv` - 36 combination ka table
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
