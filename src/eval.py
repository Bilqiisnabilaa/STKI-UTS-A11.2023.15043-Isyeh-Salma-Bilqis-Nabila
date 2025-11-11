import os
import re
import math
from collections import Counter, defaultdict


#  Load dokumen hasil preprocessing

def load_processed_docs(path="data/processed"):
    docs = {}
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read().split()
    return docs


#  Hitung TF-IDF

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


#  Representasi query sebagai vektor TF-IDF

def vectorize_query(query, idf):
    tokens = re.findall(r"\b\w+\b", query.lower())
    tf = Counter(tokens)
    query_vec = {term: (tf[term] * idf.get(term, 0)) for term in tf}
    return query_vec

#   Cosine similarity

def cosine_similarity(vec1, vec2):
    common_terms = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[t] * vec2[t] for t in common_terms)
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


#   Retrieve top-k dokumen

def retrieve(query, tfidf_docs, idf, top_k=5):
    query_vec = vectorize_query(query, idf)
    scores = {doc: cosine_similarity(query_vec, vec) for doc, vec in tfidf_docs.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

#  Evaluasi metrik

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


#   Main

if __name__ == "__main__":
    data_path = "data/processed"
    docs = load_processed_docs(data_path)
    print(f"Load dokumen dari: {data_path}")
    print(f"Dokumen terbaca: {list(docs.keys())}")
    print(f"Jumlah dokumen: {len(docs)}\n")

    tfidf_docs, idf = compute_tf_idf(docs)

    
    #  Gold set disesuaikan dengan nama file yang ada
    
    queries = {
        "informasi and sistem": {"Pengenalan.txt", "Vector Space Model.txt"},
        "dokumen or query": {"Dokumen Preprocessing.txt", "Boolean Model.txt"},
        "(informasi or sistem) and not evaluasi": {"Pengenalan.txt", "Vector Space Model.txt"},
    }

    k = 5
    print("Query                                              | Precision@K |   MAP@K |  nDCG@K")
    print("-" * 80)

    total_p, total_ap, total_ndcg = 0, 0, 0
    n = len(queries)

    for q, gold in queries.items():
        results = retrieve(q, tfidf_docs, idf, top_k=k)
        p = precision_at_k(results, gold, k)
        ap = average_precision(results, gold, k)
        ndcg = ndcg_at_k(results, gold, k)

        total_p += p
        total_ap += ap
        total_ndcg += ndcg

        print(f"{q:50} | {p:10.2f} | {ap:7.2f} | {ndcg:7.2f}")

    print("-" * 80)
    print(f"{'Rata-rata':50} | {total_p/n:10.2f} | {total_ap/n:7.2f} | {total_ndcg/n:7.2f}")
