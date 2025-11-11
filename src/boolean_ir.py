from collections import defaultdict
from pathlib import Path
import re

# Build inverted index
def build_inverted_index(processed_docs):
    """
    Membangun inverted index dari kumpulan dokumen.
    Parameter:
        documents: dict {doc_id: [token1, token2, ...]}
    """
    inverted = defaultdict(set)
    for doc_id, tokens in processed_docs.items():
        for t in tokens:
            inverted[t].add(doc_id)
    return inverted
def build_incidence_matrix(documents, vocabulary):
    """
    Membangun incidence matrix (dokumen x term)
    """
    import numpy as np
    doc_ids = sorted(documents.keys())
    term_to_idx = {t: i for i, t in enumerate(vocabulary)}
    mat = np.zeros((len(doc_ids), len(vocabulary)), dtype=int)
    for i, doc_id in enumerate(doc_ids):
        for t in documents[doc_id]:
            if t in term_to_idx:
                mat[i, term_to_idx[t]] = 1
    return mat, doc_ids

def boolean_retrieve(query, inverted_index, all_doc_ids, stemmer=None, stop_words=None):
    """
    Menjalankan Boolean retrieval untuk query seperti:
        "term1 AND term2", "term1 OR term2", "NOT term1", "(term1 AND term2) OR term3"
    """
    all_docs = set(all_doc_ids)
    tokens = re.findall(r'\w+|AND|OR|NOT|\(|\)', query.upper())

    def get_docs(term):
        term = term.lower()
        if stop_words and term in stop_words:
            return set()
        if stemmer:
            term = stemmer(term)
        return inverted_index.get(term, set())

    def eval_expr(tokens):
        stack = []
        op_stack = []

        def apply_op():
            if not op_stack or len(stack) < 2:
                return
            b = stack.pop()
            a = stack.pop()
            op = op_stack.pop()
            if op == "AND":
                stack.append(a & b)
            elif op == "OR":
                stack.append(a | b)

        i = 0
        negate_next = False
        while i < len(tokens):
            tok = tokens[i]
            if tok == "(":
                sub, j = eval_expr(tokens[i + 1:])
                stack.append(sub)
                i += j + 1
            elif tok == ")":
                break
            elif tok in ("AND", "OR"):
                while op_stack and op_stack[-1] == "AND" and tok == "OR":
                    apply_op()
                op_stack.append(tok)
            elif tok == "NOT":
                negate_next = True
            else:
                docs = get_docs(tok)
                if negate_next:
                    docs = all_docs - docs
                    negate_next = False
                stack.append(docs)
            i += 1

        while op_stack:
            apply_op()
        return stack[-1] if stack else set(), i

    result, _ = eval_expr(tokens)
    return sorted(result)

# EVALUASI 
def calculate_precision_recall(retrieved, relevant):
    """
    Hitung Precision, Recall, dan F1 untuk hasil Boolean retrieval.
    """
    retrieved = set(retrieved)
    relevant = set(relevant)
    tp = len(retrieved & relevant)
    fp = len(retrieved - relevant)
    fn = len(relevant - retrieved)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

#  Boolean evaluator 
def eval_boolean_query(query: str, inverted_index: dict):
    """Evaluasi query Boolean (AND, OR, NOT)."""
    q = query.lower()
    q = q.replace(" and ", " AND ").replace(" or ", " OR ").replace(" not ", " NOT ")
    tokens = q.split()
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    output = []
    stack = []

    # Konversi ke Reverse Polish Notation (RPN)
    for tok in tokens:
        if tok in ("AND", "OR", "NOT"):
            while stack and prec.get(stack[-1], 0) >= prec[tok]:
                output.append(stack.pop())
            stack.append(tok)
        else:
            output.append(tok)
    while stack:
        output.append(stack.pop())

    # Evaluasi postfix
    eval_stack = []
    all_docs = set()
    for s in inverted_index.values():
        all_docs |= set(s)

    for tok in output:
        if tok == "NOT":
            a = eval_stack.pop()
            eval_stack.append(all_docs - a)
        elif tok == "AND":
            b = eval_stack.pop()
            a = eval_stack.pop()
            eval_stack.append(a & b)
        elif tok == "OR":
            b = eval_stack.pop()
            a = eval_stack.pop()
            eval_stack.append(a | b)
        else:
            eval_stack.append(set(inverted_index.get(tok, set())))

    return eval_stack[-1] if eval_stack else set()


#  Hitung skor akurasi per dokumen 
def compute_accuracy(query: str, inverted_index: dict, result_docs: set):
    q_terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query)
               if t.lower() not in ("and", "or", "not")]
    if not q_terms:
        return {}

    scores = {}
    for doc in result_docs:
        matches = sum(1 for term in q_terms if doc in inverted_index.get(term, []))
        scores[doc] = round(matches / len(q_terms), 2)
    return scores


#  Utility untuk menampilkan nama dokumen lebih rapi 
def pretty_name(filename):
    name = filename.replace(".txt", "").replace("_", " ").title()
    return name


#  Demo interaktif 
if __name__ == "__main__":
    print("=== Boolean Retrieval Model with Accuracy ===")

    processed_path = Path("E:/STKI-UTS-A11.2023.15043-ISYEH SALMA BILQIS NABILA/data/processed")
    if not processed_path.exists():
        print("Folder data/processed tidak ditemukan. Pastikan sudah ada hasil preprocessing di sana.")
        exit()

    # Muat hasil preprocessing
    print(f"Memuat dokumen dari: {processed_path} ...")
    processed_docs = {}
    for file in processed_path.glob("*.txt"):
        text = file.read_text(encoding="utf-8").split()
        processed_docs[file.name] = text
    print(f"{len(processed_docs)} dokumen dimuat.")

    # inverted index
    inverted = build_inverted_index(processed_docs)
    print(f"Inverted index berhasil dibuat ({len(inverted)} term unik).")
    print("Ketik query seperti: sistem AND temu, atau NOT sistem")
    print("Ketik 'exit' untuk keluar.\n")

    # Loop interaktif
    while True:
        query = input("Query Boolean > ").strip()
        if not query or query.lower() == "exit":
            print("Selesai.")
            break

        result_docs = eval_boolean_query(query, inverted)
        scores = compute_accuracy(query, inverted, result_docs)

        if not result_docs:
            print("Tidak ditemukan dokumen yang cocok.\n")
            continue

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        print(f"\nDitemukan {len(sorted_results)} dokumen:")
        for doc, score in sorted_results:
            print(f" - {pretty_name(doc)} (akurasi: {score:.2f})")
        print()
