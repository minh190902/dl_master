"""Tạo embedding matrix gọn (chỉ vocab UIT-VSFC) từ PhoW2V 100d.

Lưu ra data/phow2v_uitvsfc_100d.npy + vocab list, để notebook nạp nhanh thay vì đọc
file gốc 1.1GB. Tokenizer ở đây phải khớp với tokenizer trong notebook.
Chạy:  python build_embedding_matrix.py
"""
import os, json, numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

DATA_DIR = os.path.join("data", "UIT-VSFC")
EMB_SRC = os.path.join("data", "phow2v", "word2vec_vi_syllables_100dims.txt")
OUT_NPY = os.path.join("data", "phow2v_uitvsfc_100d.npy")
OUT_VOCAB = os.path.join("data", "phow2v_uitvsfc_vocab.json")
EMBED_DIM = 100

def read(split, name):
    with open(os.path.join(DATA_DIR, split, name), encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

train_texts = read("train", "sents.txt")

# Tokenizer KHỚP với notebook: full vocab (không num_words), oov, filters=""
tok = Tokenizer(oov_token="<OOV>", filters="")
tok.fit_on_texts(train_texts)
word_index = tok.word_index
vocab_size = len(word_index) + 1
print(f"Vocab size: {vocab_size}")

# Đọc PhoW2V, chỉ giữ vector cho từ trong vocab
emb_index = {}
with open(EMB_SRC, encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip().split(" ")
        if len(parts) != EMBED_DIM + 1:
            continue
        if parts[0] in word_index:
            emb_index[parts[0]] = np.asarray(parts[1:], dtype="float32")

rng = np.random.default_rng(42)
M = rng.normal(0, 0.1, (vocab_size, EMBED_DIM)).astype("float32")
hit = 0
for w, i in word_index.items():
    v = emb_index.get(w)
    if v is not None:
        M[i] = v; hit += 1
coverage = 100 * hit / len(word_index)
print(f"Coverage: {hit}/{len(word_index)} = {coverage:.1f}%")

np.save(OUT_NPY, M)
with open(OUT_VOCAB, "w", encoding="utf-8") as f:
    json.dump({"word_index": word_index, "coverage": coverage, "dim": EMBED_DIM}, f, ensure_ascii=False)
print(f"Saved {OUT_NPY} ({M.shape}, {os.path.getsize(OUT_NPY)//1024} KB) + vocab json")
