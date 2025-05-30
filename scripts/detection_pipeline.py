import os, re, glob, xml.etree.ElementTree as ET
import numpy as np
import spacy
from pathlib import Path
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

# %% [markdown]
# 定义文件路径和超参数（根据实际情况修改）

# %%

# 路径设置（根据实际情况修改）
# 文本对数据（susp, src, pairs.txt）所在目录
DATA_DIR = '/kaggle/input/plagiarism-detection-data/pan25-generated-plagiarism-detection-validation/02_validation/02_validation/'

# 标注 XML 文件所在目录
TRUTH_DIR = '/kaggle/input/plagiarism-detection-data/pan25-generated-plagiarism-detection-validation/02_validation/02_validation_truth'

# 子文件夹路径
SUSP_DIR = os.path.join(DATA_DIR, 'susp')
SRC_DIR  = os.path.join(DATA_DIR, 'src')
PAIRS_FILE = os.path.join(DATA_DIR, 'pairs')

OUTPUT_DIR = '/kaggle/working/validation_pred_xml'  # 输出结果保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)


EMB_DIR = "/kaggle/input/your-dataset-name/validation_doc_emb_cache"



WINDOW_SIZE = 6       # 每段包含句子数
STRIDE      = 2
SIM_THRESH  = 0.83    # 余弦相似度阈值
MERGE_GAP   = 30      # 合并检测段时，字符间隔 ≤ MERGE_GAP 视为同一段
TOP_K = 12

# 向量缓存目录（你可以存为 dataset 下次复用）
VEC_CACHE_DIR = "/kaggle/working/vec_cache"
os.makedirs(VEC_CACHE_DIR, exist_ok=True)

# %%
# 4. 工具函数
# ========================================
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

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T

# def merge_spans(spans):
#     """合并相邻/重叠/间隔很小的字符区间列表"""
#     if not spans:
#         return []
#     spans.sort(key=lambda x: x[0])
#     merged = [spans[0]]
#     for start, end in spans[1:]:
#         last_start, last_end = merged[-1]
#         if start - last_end <= MERGE_GAP:
#             merged[-1] = (last_start, max(last_end, end))
#         else:
#             merged.append((start, end))
#     return merged


def merge_spans(spans, max_gap=50):
    """
    合并重叠或接近的 span 段（用于合并抄袭检测结果）
    spans: List of (start, end)
    max_gap: 两个 span 之间允许的最大间隙（字符）
    return: merged spans
    """
    if not spans:
        return []

    # 排序
    spans = sorted(spans, key=lambda x: x[0])
    merged = [spans[0]]

    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]

        # 如果当前段和上一个段重叠或间隔不超过 max_gap，则合并
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


def load_cached_doc_embedding(doc_id):
    path = os.path.join(EMB_DIR, f"{doc_id}.npy")
    return np.load(path)


# %% [markdown]
# **这是用自己的模型去跑****

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-base-v2", device='cuda')  # 自动用 GPU

def encode(texts):
    return model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

def get_or_encode_vectors(doc_path, sent_list, cache_dir, prefix):
    doc_id = Path(doc_path).stem
    out_path = os.path.join(cache_dir, f"{prefix}_{doc_id}.npy")

    if os.path.exists(out_path):
        return np.load(out_path)

    chunks = sliding_window(sent_list)
    texts = [chunk[1] for chunk in chunks]
    
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(out_path, vecs)
    return vecs


# %%
import faiss

# ---------- 1) 先定义 Faiss 搜索函数（函数级、顶格） ----------
def faiss_search(query_vecs, db_vecs, top_k=5):
    dim = db_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)          # 内积 = 余弦（需归一化）
    faiss.normalize_L2(query_vecs)
    faiss.normalize_L2(db_vecs)
    index.add(db_vecs)
    sims, indices = index.search(query_vecs, top_k)
    return sims, indices                    # shape: (n_query, top_k)

stats = {"matched": 0, "empty": 0}
# ---------- 2) 单对文档处理函数 ----------
def process_pair(susp_name, src_name):
     # 新增部分：确保源文件名称包含 .txt 扩展名
    if not src_name.endswith('.txt'):
        src_name = src_name + '.txt'
        
    susp_path = os.path.join(SUSP_DIR, susp_name)
    src_path  = os.path.join(SRC_DIR,  src_name)



    if not (os.path.exists(susp_path) and os.path.exists(src_path)):
        print("❗ 路径不存在:", susp_path, src_path)
        return

    # 分句与偏移
    susp_text, susp_sents = load_and_split(susp_path)
    src_text,  src_sents  = load_and_split(src_path)
    susp_off = get_sentence_offsets(susp_text, susp_sents)
    src_off  = get_sentence_offsets(src_text,  src_sents)

    # 滑动窗口（用于偏移定位）
    susp_chunks = sliding_window(susp_sents)
    src_chunks  = sliding_window(src_sents)

    # 嵌入（优先从缓存读取）
    susp_vecs = get_or_encode_vectors(susp_path, susp_sents, VEC_CACHE_DIR, prefix='susp')
    src_vecs  = get_or_encode_vectors(src_path,  src_sents,  VEC_CACHE_DIR, prefix='src')

    # Faiss 查询候选匹配窗口
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

    # 合并 + 构建 features
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

    # 保存 XML
    out_xml = os.path.join(OUTPUT_DIR, f"{Path(susp_name).stem}-{Path(src_name).stem}.xml")
    if det_features:
        save_xml(susp_name, src_name, det_features, out_xml)
        stats["matched"] += 1
    else:
        stats["empty"] += 1
        print(f"⚠️ 无匹配结果，未生成 XML: {susp_name} - {src_name}")


def process_susp_filtered(susp_name):
    susp_path = os.path.join(SUSP_DIR, susp_name)
    top_src_names = get_top_k_sources(susp_path, k=5)

    for src_name in top_src_names:
        process_pair(susp_name, src_name)  # 你已有的主函数不变


# %%
import xml.etree.ElementTree as ET

def save_detected_xml(susp_name, src_name, features, output_path):
    """
    将检测到的抄袭片段信息保存为指定格式的 XML。
    features 为一系列 dict，每个包含 this_offset, this_length, source_reference, source_offset, source_length。
    """
    # 创建 XML 根
    root = ET.Element('document', {'reference': susp_name})
    for feat in features:
        # 添加 feature 元素
        ET.SubElement(root, 'feature', {
            'name': 'detected-plagiarism',
            'this_offset': feat['this_offset'],
            'this_length': feat['this_length'],
            'source_reference': feat['source_reference'],
            'source_offset': feat['source_offset'],
            'source_length': feat['source_length']
        })
    tree = ET.ElementTree(root)
    # 缩进（Python 3.9+支持，降低格式混乱）
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"已保存预测结果：{output_path}")

# 保存当前示例对的结果
    output_file = os.path.join(OUTPUT_DIR, f"{susp_name.replace('.txt','')}-{src_name.replace('.txt','')}.xml")
    save_detected_xml(susp_name, src_name, detected_features, output_file)



# %% [markdown]
# 2️⃣ 子集筛选 + 近似候选源文档过滤

# %%
def get_doc_embedding(path):
    text, _ = load_and_split(path)
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]


# %% [markdown]


# %%
# 获取所有源文档的整篇嵌入
src_doc_paths = [os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR) if f.endswith('.txt')]
src_doc_names = [os.path.basename(p) for p in src_doc_paths]
src_doc_vecs = []
for p in tqdm(src_doc_paths, desc="Embedding src docs", ncols=100):
    text, _ = load_and_split(p)
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    src_doc_vecs.append(vec)

# %% [markdown]
# ✅ 推荐保存格式：.npz 压缩向量字典
# ✅ 保存到 .npz 文件（当前 Notebook 最后执行一次）：

# %%
import numpy as np

# 建立 ID → 向量的映射（dict）
src_doc_dict = {
    doc_name.replace(".txt", ""): vec
    for doc_name, vec in zip(src_doc_names, src_doc_vecs)
}

# 保存为压缩文件
np.savez_compressed("validation_src_doc_vecs_e5.npz", **src_doc_dict)


# %% [markdown]
# ✅ 下次使用时只需几行代码：

# %%
# 指定你上传的数据集路径
VEC_PATH = "/kaggle/input/plagiarism-detection-data/validation_src_doc_vecs_e5.npz"
SUSP_VEC_DIR ="/kaggle/input/plagiarism-detection-data/susp_cache/kaggle/working/susp_emb_cache_e5"
# 加载所有嵌入
src_vec_npz = np.load(VEC_PATH)

# 获取单篇向量（例如：document123.txt）
# vec = src_vec_npz["document123"]
src_doc_names = list(src_vec_npz.keys())
src_doc_vecs = [src_vec_npz[k] for k in src_doc_names]


# %% [markdown]
# go on

# %%
# def get_top_k_sources(susp_path, k=5):
#     # susp_vec = get_doc_embedding(susp_path)  # shape: (384,)
#     susp_vec = np.load(os.path.join(SUSP_VEC_DIR, susp_name.replace(".txt", ".npy")))
#     sims = np.dot(src_doc_vecs, susp_vec)    # shape: (num_src,)
#     topk_idx = sims.argsort()[-k:][::-1].tolist()
#     # top_src_names = [src_doc_names[int(i) if not isinstance(i, list) else int(i[0])] for i in topk_idx]
#     # return top_src_names
#     # sims = np.dot(src_doc_vecs, susp_vec)
#     idxs = np.argsort(sims)[::-1][:top_k]
#     return [src_doc_names[i] for i in idxs]


# def get_top_k_sources(susp_name, k=10):
#     susp_vec_path = os.path.join(SUSP_VEC_DIR, susp_name.replace(".txt", ".npy"))
#     if not os.path.exists(susp_vec_path):
#         print(f"❌ Susp 向量缺失: {susp_name}")
#         return []

#     susp_vec = np.load(susp_vec_path)  # shape: (768,)
    
#     sims = np.dot(src_doc_vecs, susp_vec)
#     idxs = np.argsort(sims)[::-1][:top_k]

#     return [src_doc_names[i] for i in idxs]

def get_top_k_sources(susp_name, k):
    susp_basename = os.path.basename(susp_name)
    susp_vec_path = os.path.join(SUSP_VEC_DIR, susp_basename.replace(".txt", ".npy"))

    if not os.path.exists(susp_vec_path):
        print(f"❌ Susp 向量缺失: {susp_basename}")
        return []

    susp_vec = np.load(susp_vec_path)
    sims = np.dot(src_doc_vecs, susp_vec)
    idxs = np.argsort(sims)[::-1][:k]
    return [src_doc_names[i] for i in idxs]




# %%


# %%
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np, os
from tqdm import tqdm

model = SentenceTransformer("intfloat/e5-base-v2")
model.to("cuda")


out_dir = "susp_emb_cache_e5"
os.makedirs(out_dir, exist_ok=True)

for fname in tqdm(os.listdir(SUSP_DIR)):
    if not fname.endswith(".txt"): continue
    path = os.path.join(SUSP_DIR, fname)
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    emb = model.encode([text], convert_to_numpy=True,
                       normalize_embeddings=True)[0]
    np.save(os.path.join(out_dir, fname.replace(".txt", ".npy")), emb)


# %%
import os

# 替换为实际路径
vec_file = "/kaggle/input/plagiarism-detection-data/susp_cache/kaggle/working/susp_emb_cache_e5/suspicious-document010240.npy"
print(os.path.exists(vec_file))  # 应该是 True

# %%
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import multiprocessing, time, os, warnings

import os
import warnings


# 屏蔽所有 Python 警告
warnings.filterwarnings("ignore")


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 消除 tokenizer 并行警告
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # 关闭 TensorFlow 初始化错误输出

# 环境配置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
NUM_CORES = min(2, multiprocessing.cpu_count())

# 加载文档对
pairs = [tuple(line.strip().split()) for line in open(PAIRS_FILE, encoding='utf-8')]

with open(PAIRS_FILE, encoding='utf-8') as f:
    for line in f:
        susp_name, src_name = line.strip().split()
        # 自动补全 .txt（如果缺失）
        if not susp_name.endswith(".txt"):
            susp_name += ".txt"
        if not src_name.endswith(".txt"):
            src_name += ".txt"
        pairs.append((susp_name, src_name))

susp_files = sorted(set([s for s, _ in pairs]))

# 包装函数
def safe_process_pair(susp_name, src_name):
    try:
        process_pair(susp_name, src_name)
    except Exception as e:
        print(f"❌ 错误: {susp_name} - {src_name}，{e}")

NUM_CORES = 2  # Kaggle 最多 2 核
start = time.time()

with tqdm_joblib(tqdm(total=len(pairs), desc="📄 并行处理文档对", unit="pair")):
    # 并行：
    Parallel(n_jobs=2)(
        delayed(process_susp_filtered)(susp_name)
        for susp_name in susp_files
    )


print(f"\n✅ 全部完成！总耗时 {time.time() - start:.2f} 秒")
print(f"✅ 总处理数：{len(pairs)}，成功命中：{stats['matched']}，无匹配：{stats['empty']}")


# %%
import glob
print("实际输出文件数量：", len(glob.glob(os.path.join(OUTPUT_DIR, "*.xml"))))


# %%
print("示例路径：", os.path.join(SUSP_DIR, "suspicious-document020468.txt"))
print("存在吗？", os.path.exists(os.path.join(SUSP_DIR, "suspicious-document020468.txt")))


# %% [markdown]
# 模型训练与评估

# %%
# 示例：压缩 /kaggle/working/output_folder 为 output.zip
# !zip -r output.zip /kaggle/working/validation_pred_xml
