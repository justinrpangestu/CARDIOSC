import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="Cardio Scan",
                   layout="wide")


    # Introduction
st.write("""
         # Cardio Scan: Aplikasi Prediksi Penyakit Jantung
         
         Created by: Justin Ryan Pangestu
         """)
img=Image.open("gambar_jantung.jpg")
st.image(img, width=600)

sex_labels = {
    0: "Perempuan",
    1: "Laki - laki"}
integer_labels = {
    0: "Tidak",
    1: "Ya"
}
restecg_labels = {0: 'probable or definite left ventricular hypertrophy',
                  1: 'normal',
                  2: 'ST-T Wave abnormal'}
slope_labels = {
    0: "upsloping",
    1: "flat",
    2: "downsloping"
}
cp_labels = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-anginal Pain",
    4: "Asimptomatik"
}
thal_labels = {
    1: "Normal (Tidak ada thalassemia yang terdeteksi)",
    2: "Fixed defect (Kelainan tetap pada thalassemia)",
    3: "Reversable defect (Kelainan thalassemia yang dapat diperbaiki)"}
    
with st.expander("Tentang Cardio Scan"):
            st.write("""Aplikasi ini dirancang untuk mendeteksi risiko penyakit jantung dengan menggunakan model prediksi berbasis pembelajaran mesin. 
                     Dengan menganalisis data kesehatan pengguna, seperti tekanan darah, kadar kolesterol, dan riwayat kesehatan, aplikasi ini memberikan prediksi yang akurat untuk membantu pengguna dalam mengambil tindakan pencegahan yang tepat dan meningkatkan kesehatan jantung mereka.
    """)
with st.expander("Pentingnya menjaga kesehatan jantung"):
            st.write("""Menjaga kesehatan jantung sangat penting karena jantung adalah organ utama yang memompa darah ke seluruh tubuh. 
                     Berikut adalah beberapa alasan mengapa kesehatan jantung sangat penting:

1. **Menjaga Fungsi Vital Tubuh**:
                     Jantung yang sehat memastikan darah mengalir dengan baik ke seluruh organ dan jaringan tubuh, memungkinkan mereka berfungsi dengan optimal.
                     Ini penting untuk kesehatan secara keseluruhan dan kesejahteraan.

2. **Mencegah Penyakit Jantung**:
                     Penyakit jantung, seperti penyakit jantung koroner, serangan jantung, dan stroke, adalah penyebab utama kematian di banyak negara.
                     Dengan menjaga kesehatan jantung, Anda dapat mengurangi risiko terkena penyakit ini.

3. **Meningkatkan Kualitas Hidup**:
                     Jantung yang sehat memungkinkan Anda menjalani aktivitas sehari-hari dengan lebih mudah dan penuh energi.
                     Anda juga dapat lebih aktif secara fisik tanpa merasa kelelahan atau sesak napas.

4. **Menurunkan Risiko Penyakit Lain**:
                     Kondisi jantung yang buruk dapat mempengaruhi kesehatan organ lain dan berkontribusi pada masalah kesehatan lainnya, seperti tekanan darah tinggi, diabetes, dan gangguan ginjal.
                     Menjaga kesehatan jantung dapat membantu mencegah masalah ini.

5. **Memperpanjang Umur**:
                     Dengan menjaga kesehatan jantung, Anda dapat meningkatkan peluang untuk hidup lebih lama dan menikmati masa tua yang sehat.

6. **Meningkatkan Kesejahteraan Mental**:
                     Kesehatan jantung yang baik sering kali dikaitkan dengan kesejahteraan mental yang lebih baik.
                     Aktivitas fisik dan pola makan sehat yang baik untuk jantung juga berkontribusi pada kesehatan mental yang lebih baik.

Untuk menjaga kesehatan jantung, penting untuk menerapkan gaya hidup sehat seperti makan makanan bergizi, rutin berolahraga, menghindari merokok, dan mengelola stres.""")
    # Collect user input features into dataframe
def user_input_features():
    # Header
    st.header("Formulir Kesehatan Jantung (Isilah Terlebih Dahulu)")
    # Input Data - Bagian 1: Informasi Pribadi
    st.header("Informasi Pribadi")
    age = st.number_input("Masukkan usia kamu:", min_value=0, max_value=100, step=1, value=20)
    sex = st.selectbox("Pilih jenis kelamin kamu", options=list(sex_labels.keys()), format_func=lambda x: f" {sex_labels[x]}")
        
    # Input Data - Bagian 2: Gejala dan Tes
    st.header("Gejala dan Tes")
    cp = st.radio("Jenis nyeri dada apa yang Anda alami saat ini atau pernah alami?", 
                      options=list(cp_labels.keys()), format_func=lambda x: f" {cp_labels[x]}")
    trestbps = st.slider("Silakan masukkan tekanan darah Anda saat istirahat yang terakhir kali diukur (dalam mmHg):", 
                             min_value=0, max_value=250)
    chol = st.slider("Berapa kadar kolesterol total Anda yang terakhir kali diukur (dalam mg/dL)?", 
                         min_value=0, max_value=650, value=100)
    fbs = st.radio("Apakah kadar gula darah pasien saat puasa > 120mg?", 
                       options=list(integer_labels.keys()), format_func=lambda x: f" {integer_labels[x]}")
    restecg = st.radio("Apa hasil dari elektrokardiogram (EKG) Anda saat istirahat?", 
                           options=list(restecg_labels.keys()), format_func=lambda x: f" {restecg_labels[x]}")
    thalach = st.slider("Berapa frekuensi denyut jantung maksimum yang Anda capai selama tes stres terakhir (dalam denyut per menit)?", 
                            min_value=0, max_value=250)
    exang = st.radio("Selama aktivitas fisik intensif atau tes stres, apakah Anda pernah mengalami nyeri dada (angina)?", 
                         options=list(integer_labels.keys()), format_func=lambda x: f" {integer_labels[x]}")
        
    # Input Data - Bagian 3: Penilaian dan Hasil Tes
    st.header("Penilaian dan Hasil Tes")
    oldpeak = st.slider("Nilai oldpeak Anda menunjukkan seberapa banyak segmen ST menurun selama tes stres. Mohon masukkan nilai oldpeak Anda (dalam mm):", 
                            min_value=0, max_value=6, value=0)
    slope = st.radio("Bagaimana kemiringan segmen ST Anda selama tes stres?", 
                         options=list(slope_labels.keys()), format_func=lambda x: f" {slope_labels[x]}")
    ca = st.selectbox("Berapa jumlah pembuluh darah utama yang tampak jelas selama fluoroskopi?", 
                          options=(0, 1, 2, 3))
    thal = st.radio("Apa hasil dari tes stres thallium Anda?", 
                        options=list(thal_labels.keys()), format_func=lambda x: f" {thal_labels[x]}")

    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

    
# Urutkan input_df sesuai urutan fitur yang digunakan saat pelatihan
training_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                     'exang', 'oldpeak', 'slope', 'ca', 'thal']
input_df = input_df[training_columns]

if st.button('Prediksi Yuk!'):
    with open("best_model.pkl", "rb") as file:
         loaded_model = pickle.load(file)
    
    prediction = loaded_model.predict(input_df)
    if prediction==0:
          result = """<div style="background-color: #ccffcc; padding: 10px; border-radius: 5px;">
                    <strong>Kamu tidak terkena penyakit jantung.</strong><br>
                    Selamat! Hasil pemeriksaan menunjukkan bahwa jantung Anda dalam kondisi baik.<br>
                    Ini adalah hasil dari pilihan gaya hidup sehat yang telah Anda buat.<br>
                    Teruslah menjaga kebiasaan baik ini untuk menjaga kesehatan jantung Anda di masa depan.<br>
                    Untuk mengurangi risiko penyakit jantung dan meningkatkan kesehatan jantung secara keseluruhan, lakukan langkah - langkah pencegahan berikut:<br>
                    <strong>1. Pola Makan Sehat</strong><br>
                       - <strong>Konsumsi Sayur dan Buah</strong>: Pilih makanan yang kaya serat, vitamin, dan mineral.<br>
                       - <strong>Kurangi Lemak Jenuh dan Trans</strong>: Batasi konsumsi makanan yang tinggi lemak jenuh dan trans, seperti makanan cepat saji dan produk olahan.<br>
                       - <strong>Pilih Lemak Sehat</strong>: Konsumsi lemak sehat dari sumber seperti alpukat, kacang-kacangan, dan ikan berlemak (misalnya salmon).<br>
                       - <strong>Kurangi Garam dan Gula</strong>: Batasi asupan garam dan gula untuk menjaga tekanan darah dan berat badan.<br><br>
                    <strong>2. Olahraga Teratur</strong><br>
                       - <strong>Aktivitas Fisik</strong>: Lakukan aktivitas fisik setidaknya 150 menit per minggu, seperti jalan cepat, berlari, atau berenang.<br>
                       - <strong>Latihan Kekuatan</strong>: Sertakan latihan kekuatan untuk meningkatkan kesehatan jantung dan otot.<br><br>
                    <strong>3. Menjaga Berat Badan Ideal</strong><br>
                        Pertahankan berat badan dalam rentang sehat untuk mengurangi risiko tekanan darah tinggi, diabetes, dan gangguan kolesterol.<br><br>
                    <strong>4. Hindari Merokok</strong><br>
                        Jika Anda merokok, berhenti dapat membantu menurunkan risiko penyakit jantung dan memperbaiki kesehatan secara keseluruhan.<br><br>
                    <strong>5. Batasi Konsumsi Alkohol</strong><br>
                        Jika Anda minum alkohol, lakukan dengan moderasi. Batasnya adalah satu botol per hari untuk wanita dan dua botol per hari untuk pria.<br><br>
                    <strong>6. Kelola Stres</strong><br>
                       - <strong>Teknik Relaksasi</strong>: Gunakan teknik relaksasi seperti meditasi, yoga, atau latihan pernapasan untuk mengurangi stres.<br>
                       - <strong>Aktivitas yang Menyenangkan</strong>: Lakukan aktivitas yang Anda nikmati untuk meningkatkan kesehatan mental dan emosional.<br><br>
                    <strong>7. Rutin Pemeriksaan Kesehatan</strong><br>
                       - <strong>Cek Kesehatan Berkala</strong>: Lakukan pemeriksaan kesehatan rutin untuk memantau tekanan darah, kadar kolesterol, dan gula darah.<br>
                       - <strong>Konsultasi dengan Dokter</strong>: Diskusikan faktor risiko dan langkah-langkah pencegahan yang tepat dengan dokter.<br><br>
                    <strong>8. Tidur yang Cukup</strong><br>
                        Pastikan Anda mendapatkan 7-9 jam tidur berkualitas setiap malam untuk mendukung kesehatan jantung.
                </div>"""
    else:
          result = """<div style="background-color: #ffcccc; padding: 10px; border-radius: 5px;">
                    <strong>Kamu terkena penyakit jantung.</strong><br>
                    Meskipun hasil tes menunjukkan adanya masalah, ini adalah awal dari perjalanan Anda menuju kesehatan yang lebih baik.<br>
                    Berikut adalah beberapa rekomendasi medis yang umumnya dianjurkan untuk mengelola penyakit jantung:<br>
                    <strong>1. Evaluasi Medis Mendalam</strong><br>
                       - <strong>Pemeriksaan Kesehatan</strong>: Segera lakukan evaluasi medis menyeluruh jika mengalami gejala seperti nyeri dada, sesak napas, atau gejala lain yang mencurigakan.<br>
                       - <strong>Tes Diagnostik</strong>: Tes seperti elektrokardiogram (EKG), ekokardiogram, atau angiografi koroner mungkin diperlukan untuk menilai kondisi jantung.<br><br>
                    <strong>2. Pengobatan</strong><br>
                       - <strong>Obat-obatan</strong>: Ikuti resep obat yang diberikan oleh dokter. Ini mungkin termasuk obat untuk menurunkan tekanan darah, kolesterol, atau obat pengencer darah.<br>
                       - <strong>Pengaturan Dosis</strong>: Pastikan untuk mengikuti dosis dan jadwal yang ditentukan. Jangan berhenti atau mengubah dosis obat tanpa persetujuan dokter.<br><br>
                    <strong>3. Rehabilitasi Jantung</strong><br>
                        Ikuti program rehabilitasi jantung jika dianjurkan. Program ini sering mencakup latihan fisik yang dipantau, pendidikan tentang gaya hidup sehat, dan dukungan psikologis.<br><br>
                    <strong>4. Perubahan Gaya Hidup</strong><br>
                       - <strong>Diet Khusus</strong>: Ikuti diet yang direkomendasikan oleh ahli gizi atau dokter, seperti diet rendah garam, lemak jenuh, dan kolesterol.<br>
                       - <strong>Aktivitas Fisik</strong>: Lakukan olahraga yang dianjurkan oleh dokter. Biasanya, olahraga teratur yang tidak membebani jantung sangat dianjurkan, seperti jalan kaki atau bersepeda.<br><br>
                    <strong>5. Manajemen Faktor Risiko</strong><br>
                       - <strong>Pengelolaan Tekanan Darah</strong>: Monitor dan kontrol tekanan darah dengan pengobatan dan perubahan gaya hidup.<br>
                       - <strong>Pengelolaan Diabetes</strong>: Jika Anda menderita diabetes, kontrol gula darah dengan diet, obat-obatan, dan olahraga sesuai petunjuk dokter.<br>
                       - <strong>Berhenti Merokok</strong>: Jika Anda merokok, carilah bantuan untuk berhenti. Merokok dapat memperburuk kondisi jantung.<br><br>
                    <strong>6. Pemeriksaan Rutin</strong><br>
                       - <strong>Kunjungan Berkala</strong>: Jadwalkan kunjungan rutin dengan dokter untuk memantau kondisi jantung dan mengevaluasi efektivitas pengobatan.<br>
                       - <strong>Tes Berkala</strong>: Lakukan tes yang diperlukan, seperti tes kolesterol dan tes fungsi jantung, untuk memastikan pengelolaan penyakit jantung yang efektif.<br><br>
                    <strong>7. Konseling dan Dukungan</strong><br>
                       - <strong>Dukungan Psikologis</strong>: Pertimbangkan konseling atau terapi jika Anda mengalami stres, kecemasan, atau depresi terkait penyakit jantung.<br>
                       - <strong>Kelompok Dukungan</strong>: Bergabunglah dengan kelompok dukungan penyakit jantung untuk mendapatkan informasi dan dukungan emosional tambahan.<br><br>
                    <strong>8. Pemantauan Gejala</strong><br>
                        Catat dan laporkan gejala baru atau perubahan kondisi kepada dokter. Ini membantu dalam penyesuaian pengobatan atau perawatan jika diperlukan.
                </div>"""

    st.subheader("Prediksi: ")
    with st.spinner('Tunggu sebentar...'):
        time.sleep(4)
    st.markdown(result, unsafe_allow_html=True)
