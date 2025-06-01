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

    model = SentenceTransformer("/mnt/d/Tools/Models/intfloat-e5-base-v2", device='cuda', use_auth_token=False)

    def encode(texts):
        return model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    print("🔄 Computing source document embeddings (this runs once per session)...")
    src_doc_names, src_doc_vecs = [], []

    # MAX_SRC = 10
    # for i, src_path in enumerate(tqdm(sorted(glob.glob(os.path.join(src_dir, '*.txt'))), desc="SRC")):
    #     if i >= MAX_SRC:
    #         break
    #     name = Path(src_path).name
    #     text, _ = load_and_split(src_path)
    #     vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    #     src_doc_names.append(name)
    #     src_doc_vecs.append(vec.astype(np.float32))

    for src_path in tqdm(sorted(glob.glob(os.path.join(src_dir, '*.txt'))), desc="SRC"):
        name = Path(src_path).name
        text, _ = load_and_split(src_path)
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        src_doc_names.append(name)
        src_doc_vecs.append(vec.astype(np.float32))
    src_doc_vecs = np.stack(src_doc_vecs)
    print(f"✅ {len(src_doc_names)} source docs encoded.")


    def get_top_k_sources(susp_path, k=TOP_K):
        text, _ = load_and_split(susp_path)
        susp_vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(src_doc_vecs, susp_vec)
        idxs = np.argsort(sims)[::-1][:k]
        return [src_doc_names[i] for i in idxs]

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

    def process_susp_filtered(susp_name):
        for src_name in get_top_k_sources(os.path.join(susp_dir, susp_name)):
            process_pair(susp_name, src_name)

    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pairs = [tuple(line.strip().split()) for line in open(pairs_file, encoding='utf-8')]



    susp_files = sorted({s if s.endswith('.txt') else f"{s}.txt" for s, _ in pairs})

    NUM_CORES = min(2, multiprocessing.cpu_count())
    start = time.time()

    with tqdm_joblib(tqdm(total=len(susp_files), desc="📄 Suspicious docs", unit="doc")):
        Parallel(n_jobs=NUM_CORES)(
            delayed(process_susp_filtered)(susp_name)
            for susp_name in susp_files
        )

    print(f"\n✅ Done in {time.time() - start:.2f}s | matched={stats['matched']} empty={stats['empty']}")

# scripts/detection_pipeline.py
# import os, re, glob, time, warnings, multiprocessing, xml.etree.ElementTree as ET
# from pathlib import Path
# from typing import List, Tuple
# import numpy as np
# import faiss
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from joblib import Parallel, delayed
# from tqdm_joblib import tqdm_joblib
#
# # ---------------- hyper-params ----------------
# WINDOW_SIZE = 6
# STRIDE      = 2
# SIM_THRESH  = 0.83
# MERGE_GAP   = 30
# TOP_K       = 12
# # ---------------------------------------------
#
# # ---------- utils ----------
# def simple_sent_tokenize(text: str) -> List[str]:
#     text = re.sub(r'\s+', ' ', text.strip())
#     return [s for s in re.split(r'(?<=[.!?])\s+', text) if s]
#
# def load_and_split(path: str):
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         txt = f.read()
#     return txt, simple_sent_tokenize(txt)
#
# def sliding_window(sents, win=WINDOW_SIZE, stride=STRIDE):
#     return [(i, " ".join(sents[i:i+win]))
#             for i in range(0, len(sents) - win + 1, stride)]
#
# def get_sentence_offsets(text, sents):
#     off, cur = [], 0
#     for s in sents:
#         idx = text.find(s, cur)
#         idx = idx if idx != -1 else cur
#         off.append((idx, len(s)))
#         cur = idx + len(s)
#     return off
#
# def merge_spans(spans, max_gap=MERGE_GAP):
#     if not spans:
#         return []
#     spans = sorted(spans, key=lambda x: x[0])
#     merged = [spans[0]]
#     for s, e in spans[1:]:
#         ps, pe = merged[-1]
#         if s <= pe + max_gap:
#             merged[-1] = (ps, max(pe, e))
#         else:
#             merged.append((s, e))
#     return merged
#
# def save_xml(susp, src, feats, out_path):
#     root = ET.Element("document", {"reference": susp})
#     for f in feats:
#         ET.SubElement(root, "feature", {
#             "name": "detected-plagiarism",
#             "this_offset": str(f["this_offset"]),
#             "this_length": str(f["this_length"]),
#             "source_reference": src,
#             "source_offset": str(f["source_offset"]),
#             "source_length": str(f["source_length"]),
#         })
#     ET.indent(ET.ElementTree(root), space="  ")
#     ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
# # -----------------------------
#
# # ----------- embedding helpers -----------
# def load_src_doc_vecs(src_dir: str, model, max_docs=None, debug=False):
#     names, vecs = [], []
#     paths = sorted(glob.glob(os.path.join(src_dir, "*.txt")))
#     if max_docs:
#         paths = paths[:max_docs]
#
#     for p in tqdm(paths, desc="SRC"):
#         name = Path(p).name
#         if debug:
#             vec = np.random.randn(model.get_sentence_embedding_dimension()).astype(np.float32)
#         else:
#             text, _ = load_and_split(p)
#             vec = model.encode(
#                 [text], convert_to_numpy=True, normalize_embeddings=True
#             )[0].astype(np.float32)
#         names.append(name)
#         vecs.append(vec)
#     return names, np.stack(vecs) if vecs else np.array([])
#
# def get_top_k_sources(
#     susp_path: str, src_names: List[str], src_vecs: np.ndarray, model, k=TOP_K
# ):
#     text, _   = load_and_split(susp_path)
#     susp_vec  = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
#     sims      = np.dot(src_vecs, susp_vec)
#     idxs      = np.argsort(sims)[::-1][:k]
#     return [src_names[i] for i in idxs]
# # ------------------------------------------
#
# def faiss_search(q, db, k=5):
#     dim = db.shape[1]
#     index = faiss.IndexFlatIP(dim)
#     faiss.normalize_L2(q)
#     faiss.normalize_L2(db)
#     index.add(db)
#     return index.search(q, k)  # sims, idxs
#
# def process_pair(
#     susp_name: str,
#     src_name: str,
#     susp_dir: str,
#     src_dir: str,
#     output_dir: str,
#     model,
#     encode,
# ):
#     if not src_name.endswith(".txt"):
#         src_name += ".txt"
#
#     susp_path = os.path.join(susp_dir, susp_name)
#     src_path  = os.path.join(src_dir, src_name)
#     if not (os.path.exists(susp_path) and os.path.exists(src_path)):
#         return False
#
#     susp_text, susp_sents = load_and_split(susp_path)
#     src_text,  src_sents  = load_and_split(src_path)
#     susp_off = get_sentence_offsets(susp_text, susp_sents)
#     src_off  = get_sentence_offsets(src_text,  src_sents)
#
#     susp_chunks = sliding_window(susp_sents)
#     src_chunks  = sliding_window(src_sents)
#
#     susp_vecs = encode([c[1] for c in susp_chunks]).astype(np.float32)
#     src_vecs  = encode([c[1] for c in src_chunks]).astype(np.float32)
#
#     sims, idxs = faiss_search(susp_vecs, src_vecs)
#
#     spans = []
#     for i, (s_idx, _) in enumerate(susp_chunks):
#         for score, j in zip(sims[i], idxs[i]):
#             if score >= SIM_THRESH:
#                 r_idx, _ = src_chunks[j]
#                 s_start = susp_off[s_idx][0]
#                 s_end   = susp_off[s_idx + WINDOW_SIZE - 1][0] + susp_off[s_idx + WINDOW_SIZE - 1][1]
#                 r_start = src_off[r_idx][0]
#                 r_end   = src_off[r_idx + WINDOW_SIZE - 1][0] + src_off[r_idx + WINDOW_SIZE - 1][1]
#                 spans.append((s_start, s_end, r_start, r_end))
#
#     merged = merge_spans([(s, e) for (s, e, _, _) in spans])
#     feats  = []
#     for ms, me in merged:
#         ss = min(r_s for (s, e, r_s, r_e) in spans if s >= ms and e <= me)
#         se = max(r_e for (s, e, r_s, r_e) in spans if s >= ms and e <= me)
#         feats.append({
#             "this_offset": ms,
#             "this_length": me - ms,
#             "source_offset": ss,
#             "source_length": se - ss,
#         })
#
#     if feats:
#         out_xml = os.path.join(output_dir, f"{Path(susp_name).stem}-{Path(src_name).stem}.xml")
#         save_xml(susp_name, src_name, feats, out_xml)
#         return True
#     return False
# # ------------------------------------------
#
# def run_plagiarism_detection(
#     pairs_file: str,
#     susp_dir: str,
#     src_dir: str,
#     output_dir: str,
#     max_src_docs=None,
#     debug=False,
# ):
#     """主入口：符合主办方 API 要求"""
#     os.makedirs(output_dir, exist_ok=True)
#
#     # model  = SentenceTransformer("intfloat/e5-base-v2", device="cuda")
#     model  = SentenceTransformer("/mnt/d/Tools/Models/intfloat-e5-base-v2", device="cuda")
#     encode = lambda t: model.encode(t, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
#
#     # 1) 预编码源文档
#     src_doc_names, src_doc_vecs = load_src_doc_vecs(src_dir, model, max_src_docs, debug)
#
#     # 2) 读取 pairs
#     pairs = [tuple(l.strip().split()) for l in open(pairs_file, encoding="utf-8")]
#     susp_files = sorted({s if s.endswith(".txt") else f"{s}.txt" for s, _ in pairs})
#
#     # 3) 并行处理
#     warnings.filterwarnings("ignore")
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#
#     stats = {"matched": 0, "empty": 0}
#     def _proc(susp_name):
#         hit_any = False
#         for src_name in get_top_k_sources(
#             os.path.join(susp_dir, susp_name), src_doc_names, src_doc_vecs, model
#         ):
#             hit_any |= process_pair(
#                 susp_name, src_name, susp_dir, src_dir, output_dir, model, encode
#             )
#         stats["matched" if hit_any else "empty"] += 1
#
#     num_cores = min(2, multiprocessing.cpu_count())
#     start = time.time()
#     with tqdm_joblib(tqdm(total=len(susp_files), desc="Suspicious", unit="doc")):
#         Parallel(n_jobs=num_cores)(delayed(_proc)(sn) for sn in susp_files)
#     elapsed = time.time() - start
#     print(f"Finished in {elapsed:.1f}s | matched={stats['matched']} empty={stats['empty']}")


