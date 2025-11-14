# Mini Search Engine -- UTS Sistem Temu Kembali Informasi (STKI)

Proyek ini merupakan implementasi **Mini Search Engine** untuk memenuhi
tugas **UTS Mata Kuliah Sistem Temu Kembali Informasi (STKI)**.\
Aplikasi ini meniru cara kerja sistem pencarian sederhana menggunakan:

-   **Boolean Retrieval**
-   **Vector Space Model (VSM)** berbasis **TF-IDF + Cosine Similarity**

------------------------------------------------------------------------

##  1. Deskripsi Proyek

Tujuan utama proyek:

-   Melakukan preprocessing pada sekumpulan dokumen teks.
-   Mengimplementasikan dua model pencarian: **Boolean** dan **VSM**.
-   Mengukur kualitas hasil pencarian menggunakan:\
    **Precision@K, MAP@K, nDCG@K**.
-   Menyediakan **CLI (Command Line Interface)** yang interaktif dan
    mudah digunakan.

Proyek ini merupakan implementasi **Sub-CPMK 10.1.1 -- 10.1.4**,
mencakup konsep STKI, preprocessing, model IR, dan evaluasi performa.

------------------------------------------------------------------------

##  2. Struktur Folder

    STKI-UTS-A11.2023.15043-ISYEH-SALMA-BILQIS-NABILA/
    │
    ├── app/
    │   └── main.py
    │
    ├── data/
    │   ├── Pengenalan.txt
    │   ├── Boolean Model.txt
    │   ├── Vector Space Model.txt
    │   ├── Evaluasi.txt
    │   ├── Dokumen Preprocessing.txt
    │   ├── Naive Bayes.txt
    │   └── Search Engine Concept.txt
    │
    ├── data/processed/
    │   ├── (hasil preprocessing otomatis)
    │   └── distribusi_panjang_dokumen.png
    │
    ├── processed/(hasil preprocessing otomatis)
    │   ├── CLEAN_Pengenalan.txt
    │   ├── CLEAN_Boolean Model.txt
    │   ├── CLEAN_Vector Space Model.txt
    │   ├── CLEAN_Evaluasi.txt
    │   ├── CLEAN_Dokumen Preprocessing.txt
    │   ├── CLEAN_Naive Bayes.txt
    │   └── CLEAN_Search Engine Concept.txt
    │
    ├── notebooks/
    │   └── UTS_STKI_A11.2023.15043_ISYEH SALMA BILQIS NABILA.ipynb
    │
    ├── reports/
    │   ├── LAPORAN UTS-STKI-A11.2023.15043-Isyeh Salma Bilqis Nabila.pdf
    │   ├── SOAL 01-UTS-STKI-A11.2023.15043-Isyeh Salma Bilqis Nabila.pdf
    │   └── readme.md
    │
    ├── src/
    │   ├── preprocess.py
    │   ├── boolean_ir.py
    │   ├── vsm_ir.py
    │   ├── search.py
    │   └── eval.py
    │
    └── requirements.txt

------------------------------------------------------------------------

##  3. Instalasi

### a. Clone Repository

``` bash
git clone <URL-repo-anda>
cd STKI-UTS-A11.2023.15043-ISYEH-SALMA-BILQIS-NABILA
```

### b. Install dependencies

``` bash
pip install -r requirements.txt
```

### c. Install tambahan NLTK

``` python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

------------------------------------------------------------------------

##  4. Menjalankan Program

### Menjalankan menu utama

``` bash
python src/main.py
```

Program akan menampilkan menu:

    === UTS STKI - MAIN MENU ===
    1) Preprocess documents
    2) Build indices
    3) Boolean query (interactive)
    4) Build VSM (TF-IDF) and run example query
    5) Interactive VSM search (top-K)
    6) Run evaluation examples (Precision/Recall/F1/nDCG)
    0) Exit

### Penjelasan Menu

  Menu    Fungsi
  ------- -------------------------------------------
  **1**   Melakukan preprocessing dokumen
  **2**   Membuat indeks Boolean & inverted index
  **3**   Boolean search interaktif
  **4**   Build model TF-IDF dan contoh query
  **5**   VSM search interaktif (top-k ranking)
  **6**   Evaluasi menggunakan Precision, MAP, nDCG
  **0**   Keluar

------------------------------------------------------------------------

##  5. Evaluasi Sistem

Evaluasi dilakukan dengan membandingkan hasil pencarian top-k dengan
**gold set** (dokumen relevan).

### Metrik yang digunakan:

-   **Precision@K**
-   **MAP@K**
-   **nDCG@K**

### Contoh Hasil:

Query: `informasi and sistem`

  ------------------------------------------------------------------------
  Rank     Doc ID                                 Cosine      Snippet
  -------- -------------------------------------- ----------- ------------
  1        Vector Space Model.txt                 0.0723      sistem temu
                                                              kembali
                                                              informasi
                                                              berbasis ...

  2        Pengenalan.txt                         0.0614      pengenalan
                                                              konsep
                                                              sistem
                                                              informasi
                                                              dan ...
  ------------------------------------------------------------------------

**Precision@5: 0.40 · MAP@5: 0.45 · nDCG@5: 0.58**

------------------------------------------------------------------------

##  6. Visualisasi & Preprocessing

`preprocess.py` melakukan:

-   Case folding\
-   Tokenisasi\
-   Stopword removal\
-   Stemming sederhana\
-   Pembersihan angka & tanda baca

Hasil preprocessing tersimpan di folder:

-   `data/processed/`
-   `processed/`

Script juga menampilkan:

-   10 token terbanyak di tiap dokumen
-   Grafik distribusi panjang dokumen (`.png`)

------------------------------------------------------------------------

##  7. Evaluasi Capaian Sub-CPMK

  Sub-CPMK     Capaian
  ------------ ---------------------------------------------
  **10.1.1**   Menjelaskan konsep dasar STKI
  **10.1.2**   Menerapkan preprocessing teks
  **10.1.3**   Implementasi Boolean & VSM (Cosine)
  **10.1.4**   Mendesain & mengevaluasi Mini Search Engine

------------------------------------------------------------------------

##  8. Reprodusibilitas

Gunakan **Python 3.10+**.

### Jalankan preprocessing

``` bash
python src/preprocess.py
```

### Jalankan program utama

``` bash
python src/main.py
```

### Evaluasi langsung

``` bash
python src/eval.py
```

------------------------------------------------------------------------

##  9. Informasi Mahasiswa

**Nama:** Isyeh Salma Bilqis Nabila\
**NIM:** A11.2023.15043\
**Program Studi:** Teknik Informatika\
**Mata Kuliah:** Sistem Temu Kembali Informasi (STKI)\
**Universitas:** Universitas Dian Nuswantoro (UDINUS)
