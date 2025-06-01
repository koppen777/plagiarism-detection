import json
from pathlib import Path

def load_pairs(pairs_file):
    """
    Load suspicious-source document pairs from TSV file.
    Returns a list of (susp_doc_id, src_doc_id).
    """
    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                susp_id, src_id = line.strip().split("\t")
                pairs.append((susp_id, src_id))
    return pairs

# def load_pairs(pairs_file):
#     pairs = []
#     with open(pairs_file, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split()  # 不传参数，默认按任意空白符拆分
#             if len(parts) != 2:
#                 print(f"Warning: skipping malformed line: {line.strip()}")
#                 continue
#             susp_id, src_id = parts
#             pairs.append((susp_id, src_id))
#     return pairs


def load_docs(doc_dir):
    """
    Load all documents from a directory.
    Returns a dict: {doc_id: doc_text}
    Assumes files are named as <doc_id>.txt
    """
    docs = {}
    for path in Path(doc_dir).glob("*.txt"):
        doc_id = path.stem
        with open(path, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    return docs

def detect_matches(pairs, susp_docs, src_docs):
    """
    Placeholder for the actual detection logic.
    For demo, just returns empty list of matches.
    You should replace this function with your embedding + FAISS logic.
    """
    matches = []
    # Example of output format:
    # matches.append({
    #     "susp_id": "doc1",
    #     "src_id": "doc2",
    #     "susp_start": 0,
    #     "susp_end": 10,
    #     "src_start": 5,
    #     "src_end": 15,
    #     "score": 0.85
    # })
    return matches

def write_output(matches, output_path):
    """
    Write detection matches to a JSONL file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for match in matches:
            f.write(json.dumps(match, ensure_ascii=False) + "\n")
