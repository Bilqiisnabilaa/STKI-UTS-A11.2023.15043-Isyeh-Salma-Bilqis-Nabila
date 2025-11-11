#  Mini Search Engine – UTS Sistem Temu Kembali Informasi (STKI)

Proyek ini merupakan implementasi **Mini Search Engine** untuk memenuhi tugas **UTS Mata Kuliah Sistem Temu Kembali Informasi (STKI)**.  
Aplikasi ini dirancang untuk meniru cara kerja sistem pencarian informasi sederhana menggunakan **Boolean Retrieval** dan **Vector Space Model (VSM)** dengan pendekatan **TF-IDF** dan **Cosine Similarity**.

---

##  Deskripsi Proyek

### Tujuan Utama
1. Mengolah sekumpulan dokumen teks dan menyiapkannya untuk pencarian (*preprocessing*).  
2. Mengimplementasikan dua model pencarian: **Boolean** dan **VSM (TF-IDF)**.  
3. Mengukur kualitas hasil pencarian menggunakan metrik:
   - **Precision@K**
   - **MAP@K**
   - **nDCG@K**
4. Menyediakan sistem pencarian berbasis **Command Line Interface (CLI)** yang interaktif dan mudah digunakan.  

Proyek ini merupakan hasil implementasi **Sub-CPMK 10.1.1–10.1.4**, yang mencakup:
- Konsep dasar STKI  
- Preprocessing teks  
- Pembentukan model pencarian  
- Evaluasi performa sistem  

---

##  Struktur Folder
2. Struktur Folder
STKI-UTS-A11.2023.15043-ISYEH-SALMA-BILQIS-NABILA/
│
├── app/
│   ├── main.py
|
├── data/                        # Dokumen mentah (.txt) 
│   ├── Pengenalan.txt
│   ├── Boolean Model.txt
│   ├── Vector Space Model.txt
│   ├── Evaluasi.txt
│   ├── Dokumen Preprocessing.txt
│   ├── Naive Bayes.txt
│   ├── Search Engine Concept.txt
|
├── data/processed/             # Hasil preprocessing (otomatis dibuat jika menjalankan preprocess.py)           
│   ├── Pengenalan.txt
│   ├── Boolean Model.txt
|   ├── Vector Space Model.txt
│   ├── distribusi_panjang_dokumen.png
│   ├── Evaluasi.txt
│   ├── Dokumen Preprocessing.txt
│   ├── Naive Bayes.txt
│   ├── Search Engine Concept.txt
│
├── processed/                  # Hasil preprocessing (otomatis dibuat jika menjalankan main.py)
│   ├── CLEAN_Pengenalan.txt
│   ├── CLEAN_Boolean Model.txt
│   ├── CLEAN_Vector Space Model.txt
│   ├── CLEAN_Evaluasi.txt
│   ├── CLEAN_Dokumen Preprocessing.txt
│   ├── CLEAN_Naive Bayes.txt
│   ├── CLEAN_Search Engine Concept.txt
|
├── notebooks/                         
│   ├── UTS_STKI_A11.2023.15043_ISYEH SALMA BILQIS NABILA.ipynb
│
├── reports/                         
│   ├── LAPORAN UTS-STKI-A11.2023.15043-Isyeh Salma Bilqis Nabila.pdf
│   ├── SOAL 01-UTS-STKI-A11.2023.15043-Isyeh Salma Bilqis Nabila.pdf
│   ├── readme.md                # Dokumentasi proyek
|
├── src/                         # Source code utama
│   ├── preprocess.py
│   ├── boolean_ir.py
│   ├── vsm_ir.py
│   ├── search.py
│   ├── eval.py
│   └── main.py
│
├── requirements.txt             # Dependensi Python

c. Instal seluruh dependensi
pip install -r requirements.txt

d. Tambahan NLTK
modul stopword dari NLTK:
import nltk
nltk.download('stopwords')
nltk.download('punkt')

3. Menjalankan Program
Jalankan menu utama
python src/main.py

Program akan menampilkan menu interaktif di terminal seperti berikut:
=== UTS STKI - MAIN MENU ===
1) Preprocess documents
2) Build indices
3) Boolean query (interactive)
4) Build VSM (TF-IDF) and run example query
5) Interactive VSM search (top-K)
6) Run evaluation examples (Precision/Recall/F1/nDCG)
0) Exit

Penjelasan Menu:
Menu	Fungsi
1	Membersihkan dan memproses dokumen mentah ke bentuk token
2	Membangun indeks Boolean dan inverted index
3	Menjalankan pencarian Boolean interaktif
4	Membentuk model VSM (TF-IDF) dan menjalankan contoh query
5	Pencarian interaktif berbasis VSM (top-k ranking)
6	Menjalankan evaluasi metrik (Precision, MAP, nDCG)
0	Keluar dari program

4. Evaluasi Sistem
Hasil evaluasi dilakukan dengan membandingkan hasil pencarian top-k terhadap gold set (dokumen relevan yang ditentukan manual).

Metrik yang digunakan:
Precision@K — proporsi dokumen relevan dari total hasil top-k.
MAP@K — rata-rata presisi dari seluruh posisi relevan dalam top-k.
nDCG@K — mengukur urutan relevansi hasil pencarian.

Contoh hasil:
Query: informasi and sistem
+--------+-----------------------------+----------+----------------------------------------------+
| Rank   | Doc ID                      | Cosine   | Snippet                                      |
+--------+-----------------------------+----------+----------------------------------------------+
| 1      | Vector Space Model.txt      | 0.0723   | sistem temu kembali informasi berbasis ...   |
| 2      | Pengenalan.txt              | 0.0614   | pengenalan konsep sistem informasi dan ...   |
+--------+-----------------------------+----------+----------------------------------------------+
Precision@5: 0.40, MAP@5: 0.45, nDCG@5: 0.58

5. Visualisasi Preprocessing
Script preprocess.py melakukan:
Case folding (mengubah huruf menjadi kecil semua)
Tokenisasi (memisah kata)
Stopword removal
Stemming sederhana
Pembersihan angka dan tanda baca

Hasil:
Setiap file di folder data/processed merupakan hasil bersih dari preprocessing.
Program juga menampilkan 10 token paling sering muncul di setiap dokumen.
Grafik distribusi panjang dokumen divisualisasikan dengan Matplotlib.

6. Asumsi dan Catatan Teknis
Dataset berisi 7 dokumen teks buatan sendiri (topik STKI dan IR).
Bahasa campuran (Indonesia dan Inggris) diperbolehkan sesuai ketentuan.
Preprocessing sederhana tanpa stemming lanjutan (Sastrawi opsional).
Semua path relatif terhadap direktori proyek (data/ dan src/).
Evaluasi menggunakan top-k = 5.

7. Evaluasi Capaian Sub-CPMK
Sub-CPMK	Capaian Pembelajaran
10.1.1	Menjelaskan konsep dasar Sistem Temu Kembali Informasi (STKI)
10.1.2	Menerapkan tahapan preprocessing teks secara sistematis
10.1.3	Mengimplementasikan model Boolean dan VSM dengan cosine similarity
10.1.4	Mendesain dan mengevaluasi performa mini search engine sederhana

8. Reprodusibilitas
Agar eksperimen dapat direproduksi:
Gunakan Python 3.10+

Install dependensi:
pip install -r requirements.txt

Jalankan secara berurutan:
python src/preprocess.py
python src/main.py

Untuk evaluasi langsung:
python src/eval.py

9. Informasi Penulis
Nama: Isyeh Salma Bilqis Nabila
NIM: A11.2023.15043
Program Studi: Teknik Informatika
Mata Kuliah: Sistem Temu Kembali Informasi (STKI)
Universitas: Universitas Dian Nuswantoro (UDINUS)
