# ₿ Bitcoin Price Predictor

A deep learning web app that predicts Bitcoin prices using LSTM neural networks trained on 7 years of historical data.

## 🔗 Live App
[BTC Price Predictor](https://btc-price-predictor-sirufrwelrwqfbzebwrhbb.streamlit.app/)
## 📊 Model Performance
| Metric | Value |
|--------|-------|
| RMSE | $4,386 |
| Training Data | 2018–2024 |
| Window Size | 90 days |
| Architecture | 2-layer LSTM |

## 🧠 How It Works
1. Downloads live BTC-USD data via yfinance API
2. Uses last 90 days of closing prices as input
3. LSTM neural network predicts next day's price
4. Results displayed on interactive Streamlit dashboard

## ⚙️ Tech Stack
- Python | TensorFlow/Keras | LSTM
- yfinance | Scikit-learn | Plotly | Streamlit

## 📁 Project Structure
