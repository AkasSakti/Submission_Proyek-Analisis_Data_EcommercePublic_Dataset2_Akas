import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


file_path = "../Data/sellers_dataset.csv"

# Load dataset
df = pd.read_csv(file_path)

# Cek missing values
df.info()
print(df.isnull().sum())

# Menghapus duplikasi jika ada
df.drop_duplicates(inplace=True)

# Mengisi missing values jika diperlukan
df.fillna(method='ffill', inplace=True)


# Hitung jumlah seller per kota
city_counts = df['seller_city'].value_counts().reset_index()
city_counts.columns = ['seller_city', 'seller_count']

# Tentukan kuantil untuk binning
low_threshold = city_counts['seller_count'].quantile(0.25)  # 25% terbawah
high_threshold = city_counts['seller_count'].quantile(0.75)  # 75% teratas

# Buat kategori binning
def categorize_city(seller_count):
    if seller_count > high_threshold:
        return 'High Density'
    elif seller_count < low_threshold:
        return 'Low Density'
    else:
        return 'Medium Density'

city_counts['seller_category'] = city_counts['seller_count'].apply(categorize_city)

# Streamlit Dashboard
st.title("Dashboard Analisis Seller Marketplace")
st.markdown("Visualisasi Data Seller Berdasarkan Kota dan Kategori Clustering")

# Pilihan filter kategori seller
selected_category = st.selectbox("Pilih Kategori Seller", ["All"] + list(city_counts['seller_category'].unique()))

# Filter data berdasarkan kategori
if selected_category != "All":
    filtered_data = city_counts[city_counts['seller_category'] == selected_category]
else:
    filtered_data = city_counts

# Tampilkan DataFrame
st.write("### Data Seller per Kota")
st.dataframe(filtered_data)

# Visualisasi Distribusi Kategori
st.write("### Distribusi Kategori Seller per Kota")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x=city_counts['seller_category'], palette='coolwarm', ax=ax)
plt.xlabel("Kategori Kota")
plt.ylabel("Jumlah Kota")
plt.title("Distribusi Kota Berdasarkan Jumlah Seller")
st.pyplot(fig)

# Bar Chart Top 10 Kota dengan Seller Terbanyak
st.write("### Top 10 Kota dengan Seller Terbanyak")
top_10_cities = city_counts.nlargest(10, 'seller_count')
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_10_cities['seller_city'], y=top_10_cities['seller_count'], palette="viridis", ax=ax)
plt.xticks(rotation=45)
plt.xlabel("Kota")
plt.ylabel("Jumlah Seller")
plt.title("Top 10 Kota dengan Seller Terbanyak")
st.pyplot(fig)

st.markdown("Dashboard ini membantu dalam memahami **distribusi seller** berdasarkan lokasi dan membantu dalam strategi ekspansi bisnis.")