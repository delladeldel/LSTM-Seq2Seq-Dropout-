import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta

st.title("🚀 LSTM Seq2Seq Forecasting (Encoder-Decoder)")

# Load model dan scaler
@st.cache_resource
def load_models():
    encoder_model = load_model("encoder_model (2).keras")
    decoder_model = load_model("decoder_model (2).keras")
    scaler = joblib.load("scaler (4).pkl")  # pastikan scaler file sesuai
    return encoder_model, decoder_model, scaler

encoder_model, decoder_model, scaler = load_models()

# Konstanta
input_len = 60
output_len = 60
n_features = 1

# Upload file
uploaded_file = st.file_uploader("📤 Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("❌ CSV harus mengandung kolom 'ddate' dan 'tag_value'")
    else:
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate')

        st.subheader("🧾 Preview Data")
        st.dataframe(df.tail(10))

        # Ambil 60 data terakhir
        data_input = df['tag_value'].values[-input_len:]
        last_ddate = df['ddate'].iloc[-1]

        # Normalisasi dan reshape
        data_input_scaled = scaler.transform(data_input.reshape(-1, 1))
        encoder_input = data_input_scaled.reshape(1, input_len, n_features)

        # Encode input
        state_h, state_c = encoder_model.predict(encoder_input)
        states = [state_h, state_c]

        # Mulai decoding step by step
        decoder_input = np.zeros((1, 1, n_features))  # awal = nol
        predictions_scaled = []

        for _ in range(output_len):
            pred, h, c = decoder_model.predict([decoder_input] + states, verbose=0)
            pred_value = pred[0, 0, 0]
            predictions_scaled.append(pred_value)

            # Update input dan state
            decoder_input = np.array(pred_value).reshape(1, 1, 1)
            states = [h, c]

        # Invers transform prediksi
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

        # Buat timestamp prediksi
        time_interval = df['ddate'].diff().mode()[0] if df['ddate'].diff().mode().size > 0 else timedelta(seconds=10)
        future_dates = [last_ddate + (i + 1) * time_interval for i in range(output_len)]
        pred_df = pd.DataFrame({'ddate': future_dates, 'predicted_value': predictions.flatten()})

        # Plot hasil
        st.subheader("📈 Prediksi 60 Langkah ke Depan")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['ddate'].iloc[-200:], df['tag_value'].iloc[-200:], label='Data Historis')
        ax.plot(pred_df['ddate'], pred_df['predicted_value'], label='Prediksi', color='red')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Tampilkan tabel hasil prediksi
        st.subheader("📊 Tabel Prediksi Terakhir")
        st.dataframe(pred_df.tail(10))
