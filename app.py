import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeansModel
from pyspark.sql import Row

# -----------------------------------
# Inisialisasi SparkSession
# -----------------------------------
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("BitcoinClusterApp") \
        .master("local[*]") \
        .getOrCreate()

spark = get_spark()

# -----------------------------------
# Load Pipeline dan Model KMeans
# -----------------------------------
@st.cache_resource
def load_models():
    pipeline_model = PipelineModel.load("pipeline_kmeans_bitcoin")
    kmeans_model = KMeansModel.load("model_kmeans_bitcoin")
    return pipeline_model, kmeans_model

pipeline_model, kmeans_model = load_models()

# -----------------------------------
# Data Cluster Analysis
# -----------------------------------
cluster_data = {
    "Cluster": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "Ciri Khas": [
        "Volume sangat kecil, Price Change mendekati nol",
        "Volume sangat tinggi, Price Change kecil",
        "Price turun, volume sedang",
        "Price turun banyak, volume tinggi",
        "Price naik besar, volume tinggi",
        "Volume tinggi, perubahan harga kecil",
        "Volume menengah-tinggi, perubahan harga kecil",
        "Volume sedang, perubahan harga kecil hingga sedang",
        "Volume Menengah, Price Change kecil"
    ],
    "Analisis": [
    "Pasar sedang sepi; tidak banyak transaksi atau perubahan harga.",
    "Ada transaksi besar tapi harga tetap stabil; mungkin ada yang akumulasi dalam jumlah besar.",
    "Harga berpotensi turun; mulai terlihat minat untuk jual, tapi belum terlalu ramai.",
    "Harga turun drastis dengan volume besar; kemungkinan karena banyak orang panik dan menjual aset.",
    "Harga naik tajam; bisa jadi karena berita positif atau banyak yang spekulasi.",
    "Volume besar tapi harga tidak banyak berubah; mungkin hanya perpindahan aset antar pemilik besar (whale).",
    "Harga cenderung stabil meski volume tinggi; kemungkinan pasar sedang konsolidasi.",
    "Aktivitas pasar normal seperti biasanya; tidak ada pergerakan mencolok.",
    "Mirip seperti poin 1 dan 5; volume besar tapi harga tetap tidak banyak berubah."
    ]

}

# -----------------------------------
# UI Streamlit
# -----------------------------------
st.title("Prediksi Klaster Bitcoin")
st.write("Masukkan data untuk memprediksi masuk klaster berapa")

# Display K-Means visualization
st.subheader(" Visualisasi K-Means Clustering")
try:
    st.image("kmeansbtc.png", caption="K-Means Clustering Bitcoin", use_column_width=True)
except:
    st.warning(" File gambar 'kmeansbtc.png' tidak ditemukan. Pastikan file ada di direktori yang sama.")

# Input section
st.subheader(" Input Data")
col1, col2 = st.columns(2)

with col1:
    price_change = st.number_input(" Price Change", value=0.0)

with col2:
    volume = st.number_input(" Volume", value=100000.0)

if st.button("🎯 Prediksi", type="primary"):
    # Buat DataFrame Spark dari input user
    input_data = spark.createDataFrame(
        [Row(price_change=price_change, Volume=volume)]
    )

    # Transformasi dengan pipeline
    transformed_data = pipeline_model.transform(input_data)

    # Prediksi klaster
    result = kmeans_model.transform(transformed_data)

    # Ambil hasil prediksi
    prediction = result.select("prediction").collect()[0]["prediction"]

    # Display result with additional analysis
    st.success(f" Data tersebut termasuk dalam **Klaster: {prediction}**")
    
    # Show specific cluster analysis
    cluster_info = cluster_data
    selected_cluster = cluster_info["Cluster"].index(prediction)
    
    st.subheader(f" Analisis Klaster {prediction}")
    st.info(f"**Ciri Khas:** {cluster_info['Ciri Khas'][selected_cluster]}")
    st.info(f"**Analisis:** {cluster_info['Analisis'][selected_cluster]}")

# Display cluster information table
st.subheader("Informasi Semua Klaster")
df_clusters = pd.DataFrame(cluster_data)
st.dataframe(df_clusters, use_container_width=True)

# Additional information
st.subheader("ℹ️ Informasi Tambahan")
st.markdown("""
**Cara Menggunakan:**
1. Masukkan nilai **Price Change** (perubahan harga)
2. Masukkan nilai **Volume** (volume perdagangan)
3. Klik tombol **Prediksi** untuk mengetahui klaster
4. Lihat analisis detail untuk memahami karakteristik klaster

**Catatan:**
- Model ini menggunakan algoritma K-Means dengan 9 klaster
- Setiap klaster memiliki karakteristik unik berdasarkan volume dan perubahan harga
- Hasil prediksi dapat membantu memahami kondisi pasar Bitcoin
""")
