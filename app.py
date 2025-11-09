# TOMAS WAEL AI TRADER
# ---------------------
# يعتمد على الذكاء الاصطناعي + التعلم الآلي لتحليل الفوركس والكريبتو
# يستخدم المؤشرات الفنية التالية: Fibonacci, Ichimoku, Volume, MA, RSI, MACD
# يعرض أفضل فرص التداول مع 3 أهداف ووقف خسارة، ويظهرها على الشارت.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import sqlite3
import plotly.graph_objects as go
import datetime
import os

app = FastAPI(title="TOMAS WAEL AI Trader", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

DB_PATH = "signals.db"
MODEL_PATH = "ml_model.joblib"
scaler = StandardScaler()

symbols = ["EURUSD=X", "BTC-USD", "ETH-USD", "XAUUSD=X", "GBPUSD=X"]

# 🔹 إنشاء قاعدة البيانات إذا لم تكن موجودة
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        signal TEXT,
        entry REAL,
        stop_loss REAL,
        tp1 REAL,
        tp2 REAL,
        tp3 REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()
init_db()

# 🔹 حساب المؤشرات الفنية
def compute_indicators(df):
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["ema"] = ta.ema(df["Close"], length=50)
    df["sma"] = ta.sma(df["Close"], length=200)
    df["macd"] = ta.macd(df["Close"]).iloc[:, 0]
    df["volume_mean"] = df["Volume"].rolling(20).mean()
    df["fibonacci_high"] = df["High"].rolling(20).max()
    df["fibonacci_low"] = df["Low"].rolling(20).min()
    df["ichimoku_base"] = (df["High"].rolling(9).max() + df["Low"].rolling(9).min()) / 2
    df.dropna(inplace=True)
    return df

# 🔹 نموذج تعلم آلي بسيط
def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    df = yf.download("EURUSD=X", period="6mo", interval="1h")
    df = compute_indicators(df)
    df["target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    X = scaler.fit_transform(df[["rsi", "ema", "sma", "macd", "volume_mean", "ichimoku_base"]])
    y = df["target"]
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

model = train_or_load_model()

# 🔹 تحليل السوق واستخراج أفضل إشارة
def analyze_market():
    opportunities = []
    for sym in symbols:
        df = yf.download(sym, period="7d", interval="1h")
        df = compute_indicators(df)
        X = scaler.transform(df[["rsi", "ema", "sma", "macd", "volume_mean", "ichimoku_base"]])
        pred = model.predict_proba(X)[-1][1]
        signal = "BUY" if pred > 0.6 else "SELL" if pred < 0.4 else "WAIT"
        last_price = df["Close"].iloc[-1]
        tp1 = round(last_price + 0.0010 * (1 if signal == "BUY" else -1), 5)
        tp2 = round(last_price + 0.0020 * (1 if signal == "BUY" else -1), 5)
        tp3 = round(last_price + 0.0030 * (1 if signal == "BUY" else -1), 5)
        stop_loss = round(last_price - 0.0010 * (1 if signal == "BUY" else -1), 5)

        opportunities.append({
            "symbol": sym,
            "signal": signal,
            "entry": last_price,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "score": float(pred)
        })

    best = max(opportunities, key=lambda x: abs(x["score"] - 0.5))
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO signals (symbol, signal, entry, stop_loss, tp1, tp2, tp3) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (best["symbol"], best["signal"], best["entry"], best["stop_loss"], best["tp1"], best["tp2"], best["tp3"]),
    )
    conn.commit()
    conn.close()
    return best

@app.get("/")
def home():
    return JSONResponse({"message": "Welcome to TOMAS WAEL AI Trader 🚀"})

@app.get("/api/best_opportunity")
def get_best_opportunity():
    return analyze_market()

@app.get("/api/chart/{symbol}")
def chart(symbol: str):
    df = yf.download(symbol, period="7d", interval="1h")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    fig.update_layout(title=f"{symbol} Chart", template="plotly_dark")
    return JSONResponse({"symbol": symbol, "chart_url": f"https://finance.yahoo.com/quote/{symbol}"})
