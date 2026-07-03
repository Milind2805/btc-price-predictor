import streamlit as st
import numpy as np
import pandas as pd
import pickle
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Price Predictor",
    page_icon="₿",
    layout="wide"
)

# ── Load model and scaler ─────────────────────────────────────
@st.cache_resource
def load_assets():
    model = load_model('btc_lstm_model.h5')
    with open('btc_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# ── Fetch live BTC data ───────────────────────────────────────
@st.cache_data(ttl=3600)  # refresh every 1 hour
def fetch_btc_data():
    btc = yf.download('BTC-USD', start='2018-01-01', 
                       end=datetime.today().strftime('%Y-%m-%d'))
    return btc[['Close']].copy()

df = fetch_btc_data()

# ── Title ─────────────────────────────────────────────────────
st.title("₿ Bitcoin Price Predictor")
st.markdown("### LSTM Neural Network trained on 2018–2024 BTC data")
st.divider()

# ── Key metrics row ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

current_price = float(df['Close'].iloc[-1].item())
prev_price = float(df['Close'].iloc[-2].item())
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

with col1:
    st.metric("Current BTC Price", 
              f"${current_price:,.2f}",
              f"{price_change_pct:+.2f}%")
with col2:
    st.metric("Model RMSE", "$4,386")
with col3:
    st.metric("Training Data", "2018–2024")
with col4:
    st.metric("Window Size", "90 days")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Next Day Prediction",
    "📈 Historical Analysis", 
    "📊 Model Performance"
])

# ════════════════════════════════════════════════
# TAB 1 — NEXT DAY PREDICTION
# ════════════════════════════════════════════════
with tab1:
    st.subheader("🔮 Next Day BTC Price Prediction")
    
    if st.button("🚀 Predict Tomorrow's Price", use_container_width=True):
        # Get last 90 days
        last_90 = df['Close'].values[-90:]
        last_90_scaled = scaler.transform(last_90.reshape(-1, 1))
        
        # Reshape for LSTM
        X_pred = last_90_scaled.reshape(1, 90, 1)
        
        # Predict
        pred_scaled = model.predict(X_pred)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        
        # Display
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Today's Price", f"${current_price:,.2f}")
        with col_b:
            change = pred_price - current_price
            change_pct = (change / current_price) * 100
            st.metric("Predicted Tomorrow", 
                      f"${pred_price:,.2f}",
                      f"{change_pct:+.2f}%")
        with col_c:
            direction = "📈 UP" if pred_price > current_price else "📉 DOWN"
            st.metric("Predicted Direction", direction)
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="number+delta",
            value=pred_price,
            delta={
                'reference': current_price,
                'relative': True,
                'valueformat': '.2%'
            },
            title={'text': "Predicted Next Day Price (USD)"},
            number={'prefix': "$", 'valueformat': ',.0f'}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=250
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.warning("⚠️ This is a research project, not financial advice. \
                    Crypto markets are highly volatile and unpredictable.")

# ════════════════════════════════════════════════
# TAB 2 — HISTORICAL ANALYSIS
# ════════════════════════════════════════════════
with tab2:
    st.subheader("📈 BTC Price History")
    
    # Date range selector
    col_d, col_e = st.columns(2)
    with col_d:
        start_date = st.date_input("From", 
                                    value=pd.to_datetime('2023-01-01'),
                                    min_value=pd.to_datetime('2018-01-01'))
    with col_e:
        end_date = st.date_input("To", 
                                  value=pd.to_datetime(datetime.today()))
    
    filtered_df = df[
        (df.index >= pd.to_datetime(start_date)) & 
        (df.index <= pd.to_datetime(end_date))
    ]
    
    # Price chart with MA
    filtered_df = filtered_df.copy()
    filtered_df['MA50'] = filtered_df['Close'].rolling(50).mean()
    filtered_df['MA200'] = filtered_df['Close'].rolling(200).mean()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=filtered_df.index, y=filtered_df['Close'],
        name='BTC Price', line=dict(color='orange', width=1.5)
    ))
    fig2.add_trace(go.Scatter(
        x=filtered_df.index, y=filtered_df['MA50'],
        name='50-day MA', line=dict(color='blue', width=1, dash='dash')
    ))
    fig2.add_trace(go.Scatter(
        x=filtered_df.index, y=filtered_df['MA200'],
        name='200-day MA', line=dict(color='red', width=1, dash='dash')
    ))
    fig2.update_layout(
        title='BTC-USD Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Summary stats
    st.subheader("📋 Summary Statistics")
    col_f, col_g, col_h, col_i = st.columns(4)
with col_f:
    st.metric("Highest Price", 
              f"${float(filtered_df['Close'].max().item()):,.2f}")
with col_g:
    st.metric("Lowest Price", 
              f"${float(filtered_df['Close'].min().item()):,.2f}")
with col_h:
    st.metric("Average Price", 
              f"${float(filtered_df['Close'].mean().item()):,.2f}")
with col_i:
    total_return = ((float(filtered_df['Close'].iloc[-1].item()) / 
                     float(filtered_df['Close'].iloc[0].item())) - 1) * 100
    st.metric("Period Return", f"{total_return:+.1f}%")

# ════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Model Performance on Test Data")
    
    # Recreate test predictions for visualization
    scaled_data = scaler.transform(df[['Close']])
    train_size = int(len(scaled_data) * 0.8)
    test_data = scaled_data[train_size:]
    
    X_test, y_test = [], []
    for i in range(90, len(test_data)):
        X_test.append(test_data[i-90:i, 0])
        y_test.append(test_data[i, 0])
    
    X_test = np.array(X_test).reshape(-1, 90, 1)
    y_test = np.array(y_test)
    
    y_pred = model.predict(X_test)
    y_pred_orig = scaler.inverse_transform(y_pred).flatten()
    y_test_orig = scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()
    
    # Metrics
    rmse = np.sqrt(np.mean((y_test_orig - y_pred_orig) ** 2))
    mae = np.mean(np.abs(y_test_orig - y_pred_orig))
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) 
                           / y_test_orig)) * 100
    
    col_j, col_k, col_l = st.columns(3)
    with col_j:
        st.metric("RMSE", f"${rmse:,.2f}")
    with col_k:
        st.metric("MAE", f"${mae:,.2f}")
    with col_l:
        st.metric("MAPE", f"{mape:.2f}%")
    
    # Actual vs Predicted chart
    test_dates = df.index[train_size + 90:]
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=test_dates, y=y_test_orig,
        name='Actual Price', 
        line=dict(color='blue', width=1.5)
    ))
    fig3.add_trace(go.Scatter(
        x=test_dates, y=y_pred_orig,
        name='Predicted Price', 
        line=dict(color='red', width=1.5)
    ))
    fig3.update_layout(
        title='Actual vs Predicted BTC Price (Test Set)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig3, use_container_width=True)