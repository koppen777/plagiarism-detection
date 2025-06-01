import os
import re
import glob
import xml.etree.ElementTree as ET
import time
import warnings
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import multiprocessing

WINDOW_SIZE = 6
STRIDE = 2
SIM_THRESH = 0.83
MERGE_GAP = 30
TOP_K = 12

def simple_sent_tokenize(text: str):
    text = re.sub(r'\s+', ' ', text.strip())
    return [s for s in re.split(r'(?<=[.!?])\s+', text) if s]

def load_and_split(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()
    return txt, simple_sent_tokenize(txt)

def sliding_window(sentences, win=WINDOW_SIZE, stride=STRIDE):
    return [(i, " ".join(sentences[i:i+win]))
            for i in range(0, len(sentences) - win + 1, stride)]

def get_sentence_offsets(text, sentences):
    offsets, cur = [], 0
    for s in sentences:
        idx = text.find(s, cur)
        idx = idx if idx != -1 else cur
        offsets.append((idx, len(s)))
        cur = idx + len(s)
    return offsets

def merge_spans(spans, max_gap=MERGE_GAP):
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

def save_xml(susp_name, src_name, feats, out_path):
    root = ET.Element('document', {'reference': susp_name})
    for f in feats:
        ET.SubElement(root, 'feature', {
            'name': 'detected-plagiarism',
            'this_offset': str(f['this_offset']),
            'this_length': str(f['this_length']),
            'source_reference': src_name,
            'source_offset': str(f['source_offset']),
            'source_length': str(f['source_length'])
        })
    ET.indent(ET.ElementTree(root), space="  ")
    ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)

def run_plagiarism_detection(pairs_file, susp_dir, src_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = SentenceTransformer("intfloat/e5-base-v2", device='cuda', use_auth_token=False)

    def encode(texts):
        return model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    print("🔄 Computing source document embeddings (this runs once per session)...")
    src_doc_names, src_doc_vecs = [], []

    for src_path in tqdm(sorted(glob.glob(os.path.join(src_dir, '*.txt'))), desc="SRC"):
        name = Path(src_path).name
        text, _ = load_and_split(src_path)
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        src_doc_names.append(name)
        src_doc_vecs.append(vec.astype(np.float32))
    src_doc_vecs = np.stack(src_doc_vecs)
    print(f"✅ {len(src_doc_names)} source docs encoded.")

    def faiss_search(query_vecs, db_vecs, top_k=5):
        dim = db_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(query_vecs)
        faiss.normalize_L2(db_vecs)
        index.add(db_vecs)
        sims, indices = index.search(query_vecs, top_k)
        return sims, indices

    stats = {"matched": 0, "empty": 0}

    def process_pair(susp_name, src_name):
        print("process_pair", susp_name, src_name)
        if not src_name.endswith('.txt'):
            src_name += '.txt'

        susp_path = os.path.join(susp_dir, susp_name)
        src_path  = os.path.join(src_dir,  src_name)
        if not (os.path.exists(susp_path) and os.path.exists(src_path)):
            print("❗ Missing:", susp_path, src_path)
            return

        susp_text, susp_sents = load_and_split(susp_path)
        src_text,  src_sents  = load_and_split(src_path)
        susp_off = get_sentence_offsets(susp_text, susp_sents)
        src_off  = get_sentence_offsets(src_text,  src_sents)

        susp_chunks = sliding_window(susp_sents)
        src_chunks  = sliding_window(src_sents)

        susp_vecs = encode([c[1] for c in susp_chunks]).astype(np.float32)
        src_vecs  = encode([c[1] for c in src_chunks]).astype(np.float32)

        sims, idxs = faiss_search(susp_vecs, src_vecs, top_k=5)

        spans = []
        for i, (s_idx, _) in enumerate(susp_chunks):
            for score, j in zip(sims[i], idxs[i]):
                if score >= SIM_THRESH:
                    r_idx, _ = src_chunks[j]
                    s_start = susp_off[s_idx][0]
                    s_end   = susp_off[s_idx + WINDOW_SIZE - 1][0] + susp_off[s_idx + WINDOW_SIZE - 1][1]
                    r_start = src_off[r_idx][0]
                    r_end   = src_off[r_idx + WINDOW_SIZE - 1][0] + src_off[r_idx + WINDOW_SIZE - 1][1]
                    spans.append((s_start, s_end, r_start, r_end))

        merged = merge_spans([(s, e) for (s, e, _, _) in spans])
        det_features = []
        for m_start, m_end in merged:
            src_start = min(r_start for (s, e, r_start, r_end) in spans if s >= m_start and e <= m_end)
            src_end   = max(r_end   for (s, e, r_start, r_end) in spans if s >= m_start and e <= m_end)
            det_features.append({
                'this_offset': m_start,
                'this_length': m_end - m_start,
                'source_offset': src_start,
                'source_length': src_end - src_start
            })

        out_xml = os.path.join(output_dir, f"{Path(susp_name).stem}-{Path(src_name).stem}.xml")
        if det_features:
            save_xml(susp_name, src_name, det_features, out_xml)
            stats["matched"] += 1
        else:
            stats["empty"] += 1
            print(f"⚠️ No match: {susp_name} - {src_name}")

    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pairs = [tuple(line.strip().split()) for line in open(pairs_file, encoding='utf-8')]

    start = time.time()

    for src, susp in tqdm(pairs, "Run plagiarism detection"):
        process_pair(src, susp)

    print(f"\n✅ Done in {time.time() - start:.2f}s | matched={stats['matched']} empty={stats['empty']}")

