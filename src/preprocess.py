import os
import re
from collections import Counter
import matplotlib.pyplot as plt


# KONFIGURASI

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

STOPWORDS = {
    'di','ke','yang','dan','dari','pada','dengan','atau','ini','itu','untuk',
    'sebagai','adalah','oleh','juga','hingga','tersebut','karena','maka',
    'saat','kita','kami','anda','ia','mereka','sebuah','para','akan','dapat'
}


# FUNGSI DASAR

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)             # hapus angka
    text = re.sub(r'[^a-z\s]', ' ', text)        # hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()     # hapus spasi berlebih
    return text

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def simple_stem(token):
    # stemming sederhana bahasa Indonesia
    for suf in ['lah','kah','nya','kan','i','an','ku','mu','s']:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[:-len(suf)]
    return token

def stem_tokens(tokens):
    return [simple_stem(t) for t in tokens]


# PROSES SEMUA DOKUMEN

def preprocess_all_docs():
    print("=== MEMULAI PREPROCESSING ===")
    doc_lengths = {}
    all_docs = {}

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(DATA_DIR, fname)
        with open(path, encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        cleaned = clean_text(raw_text)
        tokens = tokenize(cleaned)
        tokens = remove_stopwords(tokens)
        tokens = stem_tokens(tokens)

        all_docs[fname] = tokens
        doc_lengths[fname] = len(tokens)

        # simpan hasil
        out_path = os.path.join(PROCESSED_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))

        # tampilkan 10 token paling sering
        counter = Counter(tokens)
        top10 = counter.most_common(10)
        print(f"\n {fname}")
        print(f"Jumlah token: {len(tokens)}")
        print("10 token paling sering:")
        for tok, freq in top10:
            print(f"  {tok:15s} : {freq}")

    # tampilkan dan simpan grafik panjang dokumen
    plt.figure(figsize=(8,5))
    plt.bar(doc_lengths.keys(), doc_lengths.values())
    plt.xticks(rotation=45, ha='right')
    plt.title("Distribusi Panjang Dokumen (Setelah Preprocessing)")
    plt.ylabel("Jumlah Token")
    plt.tight_layout()

    # simpan ke folder processed
    graph_path = os.path.join(PROCESSED_DIR, "distribusi_panjang_dokumen.png")
    plt.savefig(graph_path)
    print(f"\n📊 Grafik disimpan ke: {graph_path}")

    plt.show()


    print("\n=== PREPROCESSING SELESAI ===")
    return all_docs


# EKSEKUSI UTAMA

if __name__ == "__main__":
    preprocess_all_docs()
