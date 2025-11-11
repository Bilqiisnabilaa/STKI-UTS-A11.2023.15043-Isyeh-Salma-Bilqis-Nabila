import os
import re
import math
from collections import Counter, defaultdict
from tabulate import tabulate  # pip install tabulate

#  LOAD DOKUMEN 
def load_processed_docs(path="data/processed"):
    docs = {}
    if not os.path.exists(path):
        print(f"Folder {path} tidak ditemukan!")
        return docs
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read().split()
    return docs

#  HITUNG TF-IDF 
def compute_tf_idf(docs):
    N = len(docs)
    df = defaultdict(int)
    for tokens in docs.values():
        for term in set(tokens):
            df[term] += 1

    idf = {term: math.log10(N / df[term]) for term in df}

    tfidf_docs = {}
    for doc_id, tokens in docs.items():
        tf = Counter(tokens)
        max_tf = max(tf.values())
        tfidf_docs[doc_id] = {term: (tf[term] / max_tf) * idf[term] for term in tf}

    return tfidf_docs, idf

#  VECTORIZE QUERY 
def vectorize_query(query, idf):
    tokens = re.findall(r"\b\w+\b", query.lower())
    tf = Counter(tokens)
    query_vec = {term: (tf[term] * idf.get(term, 0)) for term in tf}
    return query_vec

#  COSINE SIMILARITY 
def cosine_similarity(vec1, vec2):
    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[t] * vec2[t] for t in common_terms)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

#  RETRIEVE & RANK 
def retrieve(query, tfidf_docs, idf, top_k=5):
    query_vec = vectorize_query(query, idf)
    scores = {doc: cosine_similarity(query_vec, vec) for doc, vec in tfidf_docs.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

#  EVALUASI METRIK 
def precision_at_k(results, gold, k):
    retrieved = [doc for doc, _ in results[:k]]
    rel = sum(1 for d in retrieved if d in gold)
    return rel / k

def average_precision(results, gold, k):
    relevant = 0
    sum_prec = 0.0
    for i, (doc, _) in enumerate(results[:k], start=1):
        if doc in gold:
            relevant += 1
            sum_prec += relevant / i
    return sum_prec / len(gold) if gold else 0.0

def ndcg_at_k(results, gold, k):
    dcg = 0.0
    for i, (doc, _) in enumerate(results[:k], start=1):
        rel_i = 1 if doc in gold else 0
        dcg += rel_i / math.log2(i + 1)

    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal > 0 else 0.0

#  UTILITY: snippet 120 karakter 
def get_snippet(doc_tokens, char_len=120):
    text = " ".join(doc_tokens)
    return text[:char_len] + "..." if len(text) > char_len else text

#  MAIN PROGRAM 
if __name__ == "__main__":
    docs = load_processed_docs("data/processed")
    if not docs:
        exit("Tidak ada dokumen yang terbaca di folder 'data/processed'.")

    print(f"Jumlah dokumen terbaca: {len(docs)}")

    tfidf_docs, idf = compute_tf_idf(docs)

    #  GOLD SET (Task-C) 
    file_list = set(docs.keys())
    queries = {
        "informasi and sistem": set(f for f in file_list if "Pengenalan" in f or "Vector Space Model" in f),
        "dokumen or query": set(f for f in file_list if "Preprocessing" in f or "Boolean Model" in f),
        "(informasi or sistem) and not evaluasi": set(f for f in file_list if "Pengenalan" in f or "Vector Space Model" in f),
    }

    k = 5
    total_p, total_ap, total_ndcg = 0, 0, 0

    for q, gold in queries.items():
        print(f"\nQUERY: {q}")
        results = retrieve(q, tfidf_docs, idf, top_k=k)

        if not results:
            print("Tidak ada dokumen yang cocok.")
            continue

        # Buat tabel rapih
        table_data = []
        for rank, (doc, score) in enumerate(results, 1):
            snippet = get_snippet(docs[doc], char_len=120)
            table_data.append([rank, doc, round(score, 4), snippet])

        print(tabulate(table_data, headers=["Rank", "Doc ID", "Cosine", "Snippet"], tablefmt="grid"))

        # Evaluasi metrik
        p = precision_at_k(results, gold, k)
        ap = average_precision(results, gold, k)
        ndcg = ndcg_at_k(results, gold, k)

        total_p += p
        total_ap += ap
        total_ndcg += ndcg

        print(f"Precision@{k}: {p:.2f}, MAP@{k}: {ap:.2f}, nDCG@{k}: {ndcg:.2f}")
        print("-" * 120)

    n = len(queries)
    print(f"\nRata-rata Precision@{k}: {total_p / n:.2f}")
    print(f"Rata-rata MAP@{k}: {total_ap / n:.2f}")
    print(f"Rata-rata nDCG@{k}: {total_ndcg / n:.2f}")
    print("Evaluasi lengkap selesai.")
