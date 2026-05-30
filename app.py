from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return jsonify({
        "message": "Indian Stock Analyzer API Running 🚀"
    })


@app.route("/analyz", methods=["GET"])
def analyze():

    symbol = request.args.get("symbol")

    if not symbol:
        return jsonify({
            "error": "Please provide stock symbol"
        }), 400

    symbol = symbol.strip().upper()

    # Add NSE suffix automatically
    if not symbol.endswith((".NS", ".BO")):
        symbol += ".NS"

    try:
        # Download stock data
        df = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if df.empty:
            return jsonify({
                "error": "No stock data found"
            }), 404

        # Fix MultiIndex issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Required columns
        needed = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in needed if c in df.columns]].copy()
        df.dropna(inplace=True)

        if len(df) < 50:
            return jsonify({
                "error": "Not enough stock data"
            }), 400

        # EMA
        df["EMA_20"] = df["Close"].ewm(
            span=20,
            adjust=False
        ).mean()

        df["EMA_50"] = df["Close"].ewm(
            span=50,
            adjust=False
        ).mean()

        # RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss

        df["RSI"] = 100 - (
            100 / (1 + rs)
        )

        # ATR
        prev_close = df["Close"].shift(1)

        true_range = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs()
        ], axis=1).max(axis=1)

        df["ATR"] = true_range.rolling(14).mean()

        latest = df.iloc[-1]

        current_price = round(
            float(latest["Close"]), 2
        )

        current_rsi = round(
            float(latest["RSI"]), 2
        )

        ema_20 = round(
            float(latest["EMA_20"]), 2
        )

        ema_50 = round(
            float(latest["EMA_50"]), 2
        )

        atr = round(
            float(latest["ATR"]), 2
        )

        # Trading Logic
        if current_price > ema_20 and ema_20 > ema_50:

            trade_type = "Long Term / Positional Trade"

            if current_rsi < 70:
                decision = "BUY NOW"

                buy_price = current_price

                stop_loss = round(
                    current_price - (2 * atr),
                    2
                )

                target_price = round(
                    current_price + (4 * atr),
                    2
                )

                reason = (
                    "Uptrend intact, RSI not overbought."
                )

            else:
                decision = "WAIT"

                buy_price = ema_20

                stop_loss = round(
                    ema_20 - atr,
                    2
                )

                target_price = current_price

                reason = (
                    f"RSI {current_rsi} > 70. "
                    f"Wait near EMA-20 support."
                )

        elif current_rsi < 40:

            trade_type = "Intraday / Swing Trade"

            decision = "BUY ON DIPS"

            buy_price = current_price

            stop_loss = round(
                current_price - atr,
                2
            )

            target_price = round(
                current_price + (2 * atr),
                2
            )

            reason = (
                f"RSI {current_rsi} < 40 "
                f"— oversold, reversal possible."
            )

        else:

            trade_type = "No-Trade Zone"

            decision = "WAIT"

            buy_price = current_price

            stop_loss = round(
                current_price - atr,
                2
            )

            target_price = round(
                current_price + atr,
                2
            )

            reason = (
                "No strong trend or oversold signal."
            )

        # Risk Reward
        risk = round(
            buy_price - stop_loss,
            2
        )

        reward = round(
            target_price - buy_price,
            2
        )

        rr = (
            round(reward / risk, 2)
            if risk > 0 else "N/A"
        )

        return jsonify({
            "symbol": symbol.replace(".NS", ""),
            "price": current_price,
            "rsi": current_rsi,
            "ema20": ema_20,
            "ema50": ema_50,
            "buy_price": buy_price,
            "stop_loss": stop_loss,
            "target": target_price,
            "rr": f"1:{rr}",
            "trade_type": trade_type,
            "decision": decision,
            "reason": reason
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )