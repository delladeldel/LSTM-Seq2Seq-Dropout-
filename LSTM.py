import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model

# --- Load model dan scaler ---
encoder_model = load_model("encoder_model (1).keras")
decoder_model = load_model("decoder_model (1).keras")
scaler = joblib.load("scaler (4).pkl")

# --- Fungsi prediksi seq2seq iteratif ---
def predict_seq2seq(input_seq, n_steps):
    # Reshape dan scale
    input_seq = np.array(input_seq).reshape(1, -1, 1)
    input_seq_scaled = scaler.transform(input_seq[0])  # hanya nilai (batch, timesteps, 1)
    input_seq_scaled = input_seq_scaled.reshape(1, -1, 1)

    # Encode input
    states_value = encoder_model.predict(input_seq_scaled)

    # Buat input decoder awal (timestep = 1)
    target_seq = np.zeros((1, 1, 1))

    decoded_sequence = []

    for _ in range(n_steps):
        output_tokens = decoder_model.predict([target_seq] + states_value)

        yhat = output_tokens[:, -1, 0]
        decoded_sequence.append(yhat[0])

        # Update target_seq (jadi output sebelumnya)
        target_seq = np.array(yhat).reshape(1, 1, 1)

    # Balik scale hasil prediksi
    decoded_sequence = np.array(decoded_sequence).reshape(-1, 1)
    decoded_sequence = scaler.inverse_transform(decoded_sequence)

    return decoded_sequence.flatten()

# --- Streamlit UI ---
st.title("Prediksi 60 Langkah ke Depan - Seq2Seq LSTM")

st.write("Masukkan 60 nilai historis (tag_value) untuk memprediksi 60 langkah ke depan.")

# Input manual atau file
input_data = st.text_area("Masukkan 60 nilai historis (pisahkan dengan koma)", placeholder="Contoh: 2.3, 2.4, 2.5, ..., 3.1")

if st.button("Prediksi"):
    try:
        # Parse input
        input_list = [float(x.strip()) for x in input_data.strip().split(",")]
        if len(input_list) != 60:
            st.error("Harus 60 nilai historis.")
        else:
            prediction = predict_seq2seq(input_list, 60)
            st.success("Prediksi berhasil!")

            # Tampilkan hasil
            fig, ax = plt.subplots()
            ax.plot(range(60), prediction, label="Prediksi", color='orange')
            ax.set_title("Hasil Prediksi 60 Langkah ke Depan")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

