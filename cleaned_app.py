!pip install streamlit
!pip install cloudflared
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

# Path ke dataset di Windows (gunakan format raw string r"..." untuk menghindari masalah dengan backslash)
file_path = "Data\sellers_dataset.csv"

# Load dataset
df = pd.read_csv(file_path)

# Tampilkan 5 baris pertama
df.head()


# Data Assessing
print("=== INFORMASI DATASET ===")
df.info()

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLIKAT DATA ===")
print(f"Jumlah data duplikat: {df.duplicated().sum()}")

print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe(include="all"))

# Menghapus duplikat data jika ada
df.drop_duplicates(inplace=True)

# Mengisi missing values jika ada
df.fillna(method='ffill', inplace=True)

# Cek ulang setelah preprocessing
print("\n=== CEK ULANG SETELAH PREPROCESSING ===")
print(df.isnull().sum())
print(f"Jumlah data duplikat setelah penghapusan: {df.duplicated().sum()}")

# Import library tambahan untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Menampilkan 5 data teratas
print("\n--- 5 Data Teratas ---")
print(df.head())

# Informasi dataset
print("\n--- Informasi Dataset ---")
df.info()

# Menampilkan statistik deskriptif dari dataset
print("\n--- Statistik Deskriptif ---")
print(df.describe(include="all"))

# Mengecek jumlah data unik dalam setiap kolom
print("\n--- Jumlah Data Unik ---")
print(df.nunique())

### === AGREGASI DATA === ###
# Jumlah seller per kota
seller_per_city = df.groupby('seller_city').size().reset_index(name='jumlah_seller').sort_values(by='jumlah_seller', ascending=False)

# Jumlah seller per negara bagian
seller_per_state = df.groupby('seller_state').size().reset_index(name='jumlah_seller').sort_values(by='jumlah_seller', ascending=False)

# Jumlah rata-rata seller per kode pos (jika ada variabel numerik terkait)
if 'seller_zip_code_prefix' in df.columns:
    seller_per_zip = df.groupby('seller_zip_code_prefix').size().reset_index(name='jumlah_seller').sort_values(by='jumlah_seller', ascending=False)

# Menampilkan hasil agregasi
print("\n--- Jumlah Seller per Kota (Top 10) ---")
print(seller_per_city.head(10))

print("\n--- Jumlah Seller per Negara Bagian (Top 10) ---")
print(seller_per_state.head(10))

# Visualisasi distribusi seller berdasarkan kota (Top 10)
plt.figure(figsize=(10, 5))
sns.barplot(data=seller_per_city.head(10), x='jumlah_seller', y='seller_city', palette="Blues_r")
plt.xlabel("Jumlah Seller")
plt.ylabel("Kota")
plt.title("Distribusi Seller berdasarkan Kota (Top 10)")
plt.show()

# Visualisasi distribusi seller berdasarkan negara bagian
plt.figure(figsize=(12, 6))
sns.barplot(data=seller_per_state, x='seller_state', y='jumlah_seller', palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Negara Bagian")
plt.ylabel("Jumlah Seller")
plt.title("Distribusi Seller berdasarkan Negara Bagian")
plt.show()

# Distribusi Seller berdasarkan Kode Pos (Top 10)
zip_counts = df['seller_zip_code_prefix'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=zip_counts.values, y=zip_counts.index, palette="Greens_r")
plt.xlabel("Jumlah Seller")
plt.ylabel("Kode Pos")
plt.title("Distribusi Seller berdasarkan Kode Pos (Top 10)")
plt.show()

# Analisis distribusi seller berdasarkan jumlah seller per kota (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(seller_per_city['jumlah_seller'], bins=10, kde=True, color="blue")
plt.xlabel("Jumlah Seller")
plt.ylabel("Frekuensi")
plt.title("Histogram Distribusi Seller per Kota")
plt.show()

# Boxplot untuk melihat distribusi seller berdasarkan jumlah seller per kota
plt.figure(figsize=(8, 5))
sns.boxplot(x=seller_per_city['jumlah_seller'], color="orange")
plt.xlabel("Jumlah Seller")
plt.title("Boxplot Distribusi Seller berdasarkan Kota")
plt.show()

# Korelasi antar variabel numerik
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap Korelasi Antar Variabel Numerik")
plt.show()


print("\n--- Analisis Distribusi Geografis Seller ---")
print("Kami menganalisis distribusi seller berdasarkan kota dan negara bagian untuk mengoptimalkan strategi pemasaran.")
# --- 1. Jumlah Seller per Negara Bagian ---
print("\n--- Jumlah Seller per Negara Bagian ---")
seller_per_state = df.groupby("seller_state")["seller_id"].nunique().reset_index()
seller_per_state = seller_per_state.sort_values(by="seller_id", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=seller_per_state, x="seller_state", y="seller_id", palette="viridis")
plt.xlabel("Negara Bagian")
plt.ylabel("Jumlah Seller")
plt.title("Jumlah Seller per Negara Bagian")
plt.xticks(rotation=90)
plt.show()


# --- 2. Rata-rata Seller per Kota dalam Negara Bagian ---
print("\n--- Rata-rata Seller per Kota dalam Negara Bagian ---")
seller_avg_city = df.groupby(["seller_state", "seller_city"])["seller_id"].nunique().groupby("seller_state").mean().reset_index()
seller_avg_city.rename(columns={"seller_id": "avg_seller_per_city"}, inplace=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=seller_avg_city, x="seller_state", y="avg_seller_per_city", palette="coolwarm")
plt.xlabel("Negara Bagian")
plt.ylabel("Rata-rata Seller per Kota")
plt.title("Rata-rata Seller per Kota dalam Negara Bagian")
plt.xticks(rotation=90)
plt.show()


print("\n--- Analisis Konsentrasi Seller di kota atau negara bagian tertentu ---")
# --- 1. Seller Terbanyak dan Paling Sedikit per Negara Bagian ---
print("\n--- Seller Terbanyak dan Paling Sedikit per Negara Bagian ---")
seller_count_per_city = df.groupby(["seller_state", "seller_city"])["seller_id"].nunique().reset_index()
max_sellers = seller_count_per_city.loc[seller_count_per_city.groupby("seller_state")["seller_id"].idxmax()]
min_sellers = seller_count_per_city.loc[seller_count_per_city.groupby("seller_state")["seller_id"].idxmin()]

print("Kota dengan jumlah seller terbanyak per negara bagian:")
print(max_sellers)

print("\nKota dengan jumlah seller paling sedikit per negara bagian:")
print(min_sellers)


# --- 2. Jumlah Seller per Kode Pos ---
print("\n--- Jumlah Seller per Kode Pos ---")
seller_per_zip = df.groupby("seller_zip_code_prefix")["seller_id"].nunique().reset_index()
seller_per_zip = seller_per_zip.sort_values(by="seller_id", ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(data=seller_per_zip, x="seller_zip_code_prefix", y="seller_id", palette="magma")
plt.xlabel("Kode Pos")
plt.ylabel("Jumlah Seller")
plt.title("Top 20 Kode Pos dengan Seller Terbanyak")
plt.xticks(rotation=90)
plt.show()

# Hitung jumlah seller per kota
city_counts = df['seller_city'].value_counts().reset_index()
city_counts.columns = ['seller_city', 'seller_count']

# Tentukan kuantil untuk binning (pembagian kelompok)
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

# Tampilkan hasil distribusi kategori
print(city_counts['seller_category'].value_counts())

# Visualisasi hasil binning
plt.figure(figsize=(10, 5))
sns.barplot(x=city_counts['seller_category'].value_counts().index,
            y=city_counts['seller_category'].value_counts().values,
            palette='coolwarm')
plt.title('Distribusi Kota berdasarkan Kategori Seller')
plt.xlabel('Kategori')
plt.ylabel('Jumlah Kota')
plt.show()

print("sebagian besar seller ada di kota High Density, marketplace mungkin perlu memperluas ke daerah lain.")
print("potensi ekspansi seller ke kota-kota dengan sedikit seller.")

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

%cd "Submission_Proyek Analisis_Data_EcommercePublic_Dataset2_Akas/Dashboard"

#!streamlit run "/content/drive/My Drive/Colab Notebooks/Submission_Proyek Analisis_Data_EcommercePublic_Dataset2_Akas/Dashboard/dashboard.py" & npx localtunnel --port 8501
streamlit run C:\Users\WAWAN\AppData\Local\Programs\Python\Python310\lib\site-packages\ipykernel_launcher.py 

!cloudflared tunnel --url http://localhost:8501




