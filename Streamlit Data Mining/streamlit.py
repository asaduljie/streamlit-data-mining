import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide", page_title="Prediksi Model Neural Network")

st.title("Prediksi dengan Model Orange Neural Network")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    MODEL_PATH = SCRIPT_DIR / "iris.pkcls"

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
except FileNotFoundError:
    st.error(f"Error: File model 'iris.pkcls' tidak ditemukan.")
    st.warning("Pastikan Anda sudah meng-upload file 'iris.pkcls' ke repositori GitHub Anda.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi error saat memuat model: {e}")
    st.stop()


st.header("Masukkan nilai fitur")

col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0, format="%.2f")
    feature2 = st.number_input("Feature 2", value=0.0, format="%.2f")

with col2:
    feature3 = st.number_input("Feature 3", value=0.0, format="%.2f")
    feature4 = st.number_input("Feature 4", value=0.0, format="%.2f")

data = pd.DataFrame([[feature1, feature2, feature3, feature4]],
                    columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"])

st.divider()

if st.button("Prediksi", type="primary"):
    try:
        pred_array = model(data.values)
        hasil_prediksi = pred_array[0]

        st.success(f"Hasil prediksi: {hasil_prediksi}")
        
        st.write("Data yang digunakan untuk prediksi:")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e)