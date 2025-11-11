import os
import re
import math
import numpy as np
from collections import Counter, defaultdict
from tabulate import tabulate  # pip install tabulate

#  PATH SETUP 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")

#  UTILITAS 
def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

def load_processed_docs(path=DATA_PROCESSED_DIR):
    docs = {}
    if not os.path.exists(path):
        print(f"Folder {path} tidak ditemukan!")
        return docs
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read().split()
    return docs

# PREPROCESSING 
def run_preprocessing_and_save():
    ensure_dirs()
    data_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
    if not data_files:
        print("Tidak ada file di folder data/.")
        return
    count = 0
    for fn in data_files:
        in_path = os.path.join(DATA_DIR, fn)
        out_path = os.path.join(DATA_PROCESSED_DIR, f"CLEAN_{fn}")
        with open(in_path, "r", encoding="utf-8") as f:
            tokens = f.read().lower().split()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(" ".join(tokens))
        count += 1
    print(f" Preprocessing selesai. {count} file disimpan ke processed/")

#  BUILD INDICES 
def build_indices_from_processed():
    ensure_dirs()
    docs = load_processed_docs()
    if not docs:
        print("Tidak ada dokumen processed. Jalankan preprocessing dulu.")
        return None, None, None
    all_terms = set()
    for toks in docs.values():
        all_terms.update(toks)
    vocabulary = sorted(list(all_terms))
    inverted = defaultdict(set)
    for doc_id, toks in docs.items():
        for t in toks:
            inverted[t].add(doc_id)
    print(f"Loaded {len(docs)} documents, vocab size: {len(vocabulary)}")
    return docs, vocabulary, inverted

#  BOOLEAN RETRIEVE 
def boolean_retrieve(query, inverted_index, all_doc_ids):
    query = query.upper()
    tokens = re.findall(r'\b\w+\b|AND|OR|NOT', query)
    if not tokens:
        return []

    result_set = None
    current_op = None
    negate_next = False

    for tok in tokens:
        if tok in ("AND", "OR"):
            current_op = tok
        elif tok == "NOT":
            negate_next = True
        else:
            term = tok.lower()
            docs_with_term = inverted_index.get(term, set())
            if negate_next:
                docs_with_term = set(all_doc_ids) - docs_with_term
                negate_next = False
            if result_set is None:
                result_set = docs_with_term
            else:
                if current_op == "AND":
                    result_set &= docs_with_term
                elif current_op == "OR":
                    result_set |= docs_with_term
                else:
                    result_set &= docs_with_term
    return list(result_set) if result_set else []

# BOOLEAN CLI 
def boolean_query_cli(inverted_index, all_doc_ids, documents):
    if inverted_index is None:
        print(" Boolean retrieval tidak tersedia.")
        return
    print("\nMasukkan query Boolean ('AND', 'OR', 'NOT'). Ketik 'exit' untuk kembali.")
    while True:
        q = input("Boolean query> ").strip()
        if q.lower() in ("exit", "quit", "back"):
            break
        try:
            res = boolean_retrieve(q, inverted_index, all_doc_ids)
            if not res:
                print("Tidak ada dokumen yang cocok.")
                continue
            print("\nHasil dokumen:", res)
            print(f"Jumlah dokumen cocok: {len(res)} dari {len(all_doc_ids)} "
                  f"({len(res)/len(all_doc_ids)*100:.2f}%)")
            query_terms = [t.lower() for t in re.findall(r'\b\w+\b', q)]
            print("\nJumlah kemunculan query per dokumen:")
            for doc_id in res:
                tokens = [t.lower() for t in documents[doc_id]]
                count = sum(tokens.count(term) for term in query_terms)
                print(f"  {doc_id}: {count} kali")
        except Exception as e:
            print("Error:", e)

#  TF-IDF / VSM 
def compute_tf_idf(documents):
    all_terms = sorted({t for toks in documents.values() for t in toks})
    term_to_idx = {t:i for i,t in enumerate(all_terms)}
    N = len(documents)
    df = defaultdict(int)
    for toks in documents.values():
        for t in set(toks):
            df[t] += 1
    idf_vector = np.array([math.log10(N/df[t]) for t in all_terms])

    tfidf_matrix = np.zeros((N, len(all_terms)))
    doc_ids = list(documents.keys())
    for i, doc_id in enumerate(doc_ids):
        tf = Counter(documents[doc_id])
        max_tf = max(tf.values())
        for t, cnt in tf.items():
            idx = term_to_idx[t]
            tfidf_matrix[i, idx] = (cnt / max_tf) * idf_vector[idx]
    return tfidf_matrix, idf_vector, term_to_idx, doc_ids

def query_to_tfidf_vector(query, term_to_idx, idf_vector):
    tokens = query.lower().split()
    vec = np.zeros(len(term_to_idx))
    tf = Counter(tokens)
    for t, cnt in tf.items():
        if t in term_to_idx:
            vec[term_to_idx[t]] = cnt * idf_vector[term_to_idx[t]]
    return vec

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm > 0 else 0.0

def rank_documents(query_vec, tfidf_matrix, doc_ids):
    scores = [cosine_similarity(query_vec, tfidf_matrix[i]) for i in range(len(doc_ids))]
    ranking = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return ranking

def get_snippet(doc_tokens, n=120):
    txt = " ".join(doc_tokens)
    return txt[:n]+"..." if len(txt)>n else txt


def interactive_vsm_search(tfidf_matrix, idf_vector, term_to_idx, doc_ids, documents):
    print("\nMasukkan query VSM. Ketik 'exit' untuk kembali.")
    while True:
        q = input("VSM query> ").strip()
        if q.lower() in ("exit","quit","back"):
            break
        qvec = query_to_tfidf_vector(q, term_to_idx, idf_vector)
        ranking = rank_documents(qvec, tfidf_matrix, doc_ids)
        top5 = ranking[:5]
        for doc, score in top5:
            snippet = get_snippet(documents[doc])
            print(f"{doc} | score={score:.4f} | {snippet}")

#  EVALUATION 
def run_evaluation(docs, tfidf_matrix, idf_vector, term_to_idx, doc_ids):
    gold_queries = {
        "informasi sistem": set(d for d in doc_ids if "Pengenalan" in d or "Vector Space Model" in d),
        "model boolean": set(d for d in doc_ids if "Boolean Model" in d),
        "preprocessing dokumen": set(d for d in doc_ids if "Dokumen Preprocessing" in d),
    }
    k = 5
    total_p, total_ap, total_ndcg = 0,0,0
    for q,gold in gold_queries.items():
        print(f"\nQuery: {q}")
        qvec = query_to_tfidf_vector(q, term_to_idx, idf_vector)
        ranking = rank_documents(qvec, tfidf_matrix, doc_ids)
        table = []
        for rank, (doc, score) in enumerate(ranking[:k],1):
            table.append([rank, doc, round(score,4), get_snippet(docs[doc])])
        print(tabulate(table, headers=["Rank","Doc ID","Cosine","Snippet"], tablefmt="grid"))
        # Metrik
        p = sum(1 for d,_ in ranking[:k] if d in gold)/k
        ap = sum(sum(1 for j in range(1,i+1) if ranking[j-1][0] in gold)/i for i in range(1,k+1) if ranking[i-1][0] in gold)/len(gold) if gold else 0
        ndcg = sum((1/math.log2(i+1) if ranking[i-1][0] in gold else 0) for i in range(1,k+1))/sum(1/math.log2(i+1) for i in range(1,min(k,len(gold))+1)) if gold else 0
        print(f"Precision@{k}: {p:.2f}, MAP@{k}: {ap:.2f}, nDCG@{k}: {ndcg:.2f}")
        total_p += p
        total_ap += ap
        total_ndcg += ndcg
    n = len(gold_queries)
    print(f"\nRata-rata Precision@{k}: {total_p/n:.2f}, MAP@{k}: {total_ap/n:.2f}, nDCG@{k}: {total_ndcg/n:.2f}")

# MAIN MENU 
def main_menu():
    ensure_dirs()
    docs, vocabulary, inverted = None, None, None
    tfidf_matrix, idf_vector, term_to_idx, doc_ids = None, None, None, None

    while True:
        print("\n=== UTS STKI - MAIN MENU ===")
        print("1) Preprocess documents")
        print("2) Build indices")
        print("3) Boolean query (interactive)")
        print("4) Build VSM (TF-IDF) and run example query")
        print("5) Interactive VSM search (top-K)")
        print("6) Run evaluation examples (Precision/Recall/F1/nDCG)")
        print("0) Exit")
        choice = input("Pilih nomor: ").strip()

        if choice=="1":
            run_preprocessing_and_save()
        elif choice=="2":
            docs, vocabulary, inverted = build_indices_from_processed()
            if docs:
                print(" Indeks siap digunakan.")
        elif choice=="3":
            if not docs or not inverted:
                print("Jalankan Build indices dulu (menu 2).")
                continue
            boolean_query_cli(inverted, list(docs.keys()), docs)
        elif choice=="4":
            if not docs:
                print("Jalankan Build indices dulu (menu 2).")
                continue
            tfidf_matrix, idf_vector, term_to_idx, doc_ids = compute_tf_idf(docs)
            print(" TF-IDF matrix siap. Contoh query:")
            q = "informasi sistem"
            qvec = query_to_tfidf_vector(q, term_to_idx, idf_vector)
            ranking = rank_documents(qvec, tfidf_matrix, doc_ids)
            for doc, score in ranking[:5]:
                print(f"{doc} | score={score:.4f} | {get_snippet(docs[doc])}")
        elif choice=="5":
            if tfidf_matrix is None:
                print("Bangun TF-IDF dulu (menu 4).")
                continue
            interactive_vsm_search(tfidf_matrix, idf_vector, term_to_idx, doc_ids, docs)
        elif choice=="6":
            if tfidf_matrix is None:
                print("Bangun TF-IDF dulu (menu 4).")
                continue
            run_evaluation(docs, tfidf_matrix, idf_vector, term_to_idx, doc_ids)
        elif choice=="0":
            print("Keluar. Terimakasih.")
            break
        else:
            print("Pilihan tidak dikenali. Coba lagi.")

#  ENTRY POINT 
if __name__=="__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nDihentikan pengguna. Keluar.")
