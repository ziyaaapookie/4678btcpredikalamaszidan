import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import timedelta, date
from utils.preprocessing import load_and_preprocess_data

# Load model dan scaler
with open('model/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

with open('model/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Load data
df = load_and_preprocess_data('data/btc_15m_data_2018_to_2025.csv')
last_date = df['date'].max()
max_forecast_date = last_date + timedelta(days=360)

# UI Layout
st.set_page_config(page_title="Bitcoin Forecast", layout="centered")

st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/2048px-Bitcoin.svg.png", width=150)
st.title("📈 Prediksi Harga Bitcoin ")
st.title("Dibuat Oleh Mas Zidan 22.11.4678 ")
st.markdown(f"""
Pilih tanggal maksimal 1 tahun setelah data terakhir: **{last_date.date()}**
""")

# Date input
selected_date = st.date_input("Pilih tanggal prediksi:",
                              value=last_date.date() + timedelta(days=1),
                              min_value=last_date.date() + timedelta(days=1),
                              max_value=max_forecast_date.date())

# Tombol prediksi
if st.button("🔮 Prediksi Harga"):
    # Hitung index prediksi (berapa hari dari last_date)
    delta_days = (selected_date - last_date.date()).days

    if delta_days < 1 or delta_days > 360:
        st.warning("Tanggal di luar jangkauan prediksi model.")
    else:
        last_features = df[['high_shifted', 'low_shifted', 'open_shifted', 'volume_shifted']].iloc[-360:]
        last_scaled = scaler_X.transform(last_features)
        future_scaled = model.predict(last_scaled)
        future_prices = scaler_y.inverse_transform(future_scaled.reshape(-1, 1))

        predicted_price = future_prices[delta_days - 1][0]
        st.success(f"💰 Prediksi harga Bitcoin untuk {selected_date.strftime('%d %B %Y')}: **${predicted_price:,.2f}**")

        # Garis prediksi hingga tanggal itu
        prediction_df = pd.DataFrame({
            'Tanggal': [last_date + timedelta(days=i+1) for i in range(delta_days)],
            'Harga': future_prices[:delta_days].flatten()
        })

       
