import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import joblib

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense

# =====================
# KONFIGURASI MODEL
# =====================
input_len = 60
output_len = 60
n_features = 1
latent_dim = 64  # sesuai decoder LSTM (64 units)

# =====================
# LOAD MODEL & SCALER
# =====================
encoder_model = load_model("encoder_model (1).keras")
decoder_training_model = load_model("decoder_model (1).keras")
scaler = joblib.load("scaler (4).pkl")

# =====================
# BANGUN MODEL INFERENCE DECODER
# =====================
decoder_input_inf = Input(shape=(1, n_features))
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Ambil layer dari model training
decoder_lstm_layer = decoder_training_model.layers[3]  # LSTM
decoder_dense_1 = decoder_training_model.layers[5]     # TimeDistributed(Dense 32 relu)
decoder_dense_2 = decoder_training_model.layers[6]     # TimeDistributed(Dense 1)

# Pemanggilan layer (FIXED unpack)
decoder_lstm_outputs = decoder_lstm_layer(
    decoder_input_inf, initial_state=decoder_states_inputs
)
decoder_outputs, state_h, state_c = decoder_lstm_outputs

decoder_outputs = decoder_dense_1(decoder_outputs)
decoder_outputs = decoder_dense_2(decoder_outputs)

decoder_model_inf = Model(
    [decoder_input_inf] + decoder_states_inputs,
    [decoder_outputs, state_h, state_c]
)

# =====================
# STREAMLIT APP
# =====================
st.title("LSTM Seq2Seq Forecasting 60 Langkah ke Depan")

uploaded_file = st.file_uploader("Upload file CSV (harus punya kolom 'ddate' & 'tag_value')", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("File harus memiliki kolom 'ddate' dan 'tag_value'")
    else:
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate')

        st.subheader("Preview Data")
        st.dataframe(df.tail(10))

        # Ambil 60 data terakhir
        data_input = df['tag_value'].values[-input_len:]
        last_ddate = df['ddate'].iloc[-1]

        # Normalisasi dan reshape
        data_input_scaled = scaler.transform(data_input.reshape(-1, 1))
        encoder_input = data_input_scaled.reshape(1, input_len, 1)

        # Encode input
        state_h, state_c = encoder_model.predict(encoder_input)
        states = [state_h, state_c]

        # Awal decoder
        decoder_input = np.zeros((1, 1, 1))
        predictions_scaled = []

        for _ in range(output_len):
            pred, h, c = decoder_model_inf.predict([decoder_input] + states)
            pred_value = pred[0, 0, 0]
            predictions_scaled.append(pred_value)

            # Update input dan state
            decoder_input = np.array(pred_value).reshape(1, 1, 1)
            states = [h, c]

        # Inverse transform
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

        # Buat waktu prediksi
        time_interval = df['ddate'].diff().mode()[0] if df['ddate'].diff().mode().size > 0 else timedelta(seconds=10)
        future_dates = [last_ddate + (i + 1) * time_interval for i in range(output_len)]
        pred_df = pd.DataFrame({'ddate': future_dates, 'predicted_value': predictions.flatten()})

        # Plot
        st.subheader("Prediksi 60 Langkah ke Depan")
        fig, ax = plt.subplots()
        ax.plot(df['ddate'].iloc[-200:], df['tag_value'].iloc[-200:], label='Data Historis')
        ax.plot(pred_df['ddate'], pred_df['predicted_value'], label='Prediksi', color='red')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Prediksi Terakhir")
        st.dataframe(pred_df.tail(10))
