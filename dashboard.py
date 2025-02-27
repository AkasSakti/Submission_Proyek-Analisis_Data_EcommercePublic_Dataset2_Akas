import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st


file_path = "Data/sellers_dataset.csv"

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

# Hitung jumlah seller per negara bagian
state_counts = df['seller_state'].value_counts().reset_index()
state_counts.columns = ['seller_state', 'seller_count']


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

# Pilihan filter negara bagian
selected_state = st.selectbox("Pilih Negara Bagian", ["All"] + list(state_counts['seller_state'].unique()))

# Filter data berdasarkan kategori
filtered_data = city_counts.copy()
if selected_category != "All":
    filtered_data = filtered_data[filtered_data['seller_category'] == selected_category]

# Filter data berdasarkan negara bagian
if selected_state != "All":
    filtered_data = filtered_data[filtered_data['seller_city'].isin(df[df['seller_state'] == selected_state]['seller_city'])]

# Tampilkan DataFrame
st.write("### Data Seller per Kota")
st.write("### Visualisasi data seller per kota dan negara bagian berikut akan membantu menjawab pertanyaan No 1, namun untuk mengekspansikan bisnis secara lebih tepat kita memerlukan data customer agar terbentuk pola antara seller dan customer dengan lebih baik")
st.dataframe(filtered_data)

# Visualisasi Distribusi Kategori
st.write("### Distribusi Kategori Seller per Kota")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x=filtered_data['seller_category'], palette='coolwarm', ax=ax)
plt.xlabel("Kategori Kota")
plt.ylabel("Jumlah Kota")
plt.title("Distribusi Kota Berdasarkan Jumlah Seller")
st.pyplot(fig)

# Visualisasi Distribusi Seller per Negara Bagian (Hasil Filter)
st.write("### Distribusi Seller per Negara Bagian (Hasil Filter)")
filtered_state_counts = state_counts[state_counts['seller_state'] == selected_state] if selected_state != "All" else state_counts
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=filtered_state_counts['seller_state'], y=filtered_state_counts['seller_count'], palette="magma", ax=ax)
plt.xticks(rotation=45)
plt.xlabel("Negara Bagian")
plt.ylabel("Jumlah Seller")
plt.title("Distribusi Seller Berdasarkan Negara Bagian")
st.pyplot(fig)

# Bar Chart Top 10 Kota dengan Seller Terbanyak
st.write("### Top 10 Kota dengan Seller Terbanyak (Hasil Filter)")
top_10_cities = filtered_data.nlargest(10, 'seller_count')
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=top_10_cities['seller_city'], y=top_10_cities['seller_count'], palette="viridis", ax=ax)
plt.xticks(rotation=45)
plt.xlabel("Kota")
plt.ylabel("Jumlah Seller")
plt.title("Top 10 Kota dengan Seller Terbanyak")
st.pyplot(fig)

# scater plot distribusi seller berdasarkan negara bagian untuk melihat wilayah dengan potensi ekspansi
st.write("### Visualisasi Berikut menjawab pertanyaan No 2, dengan visualisasi ini kita bisa melihat hubungan distribusi seller terhadap negara bagian yang minim seller sehingga dapat dilakukan untuk ekspansi bisnis pemerataan seller tiap negara bagian dengan tetap melihat data customer")
st.write("### Jumlah Seller di Setiap Negara Bagian")
seller_per_state = df.groupby('seller_state')['seller_id'].nunique().reset_index()
seller_per_state.columns = ['seller_state', 'jumlah_seller']

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x='seller_state', y='jumlah_seller', data=seller_per_state, color='green', alpha=0.7, ax=ax)
plt.xticks(rotation=45)
plt.xlabel("Negara Bagian")
plt.ylabel("Jumlah Seller")
plt.title("Jumlah Seller di Setiap Negara Bagian")

st.pyplot(fig)


st.markdown("Dashboard ini membantu dalam memahami **distribusi seller** berdasarkan lokasi dan membantu dalam strategi ekspansi bisnis.")
