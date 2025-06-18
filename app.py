import streamlit as st
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
# UI Streamlit
# -----------------------------------
st.title("🚀 Prediksi Klaster Bitcoin")
st.write("Masukkan data untuk memprediksi masuk klaster berapa")

price_change = st.number_input("📈 Price Change", value=0.0)
volume = st.number_input("🔊 Volume", value=100000.0)

if st.button("Prediksi"):
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

    st.success(f"📊 Data tersebut termasuk dalam Klaster: **{prediction}**")
