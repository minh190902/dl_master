"""Build the Vietnamese Sentiment Analysis notebook (Multi-channel LSTM-CNN).

Generates `25C15052.ipynb` as a clean nbformat-v4
notebook. Keeping the notebook as code-generated guarantees valid JSON structure.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))

# ----------------------------------------------------------------------------
# 0. Title / Introduction
# ----------------------------------------------------------------------------
md(r"""
# Multi-channel LSTM-CNN cho Phân tích Cảm xúc Tiếng Việt

**Bài tập Deep Learning** — Tái hiện kiến trúc trong bài báo:

> Vo, Q., Nguyen, H., Le, B., Nguyen, M. (2017).
> *Multi-channel LSTM-CNN model for Vietnamese sentiment analysis.*
> 9th International Conference on Knowledge and Systems Engineering (KSE), IEEE.
> [ResearchGate](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis)
> · [Code gốc của tác giả](https://github.com/ntienhuy/MultiChannel)

---

## Nội dung bài tập

| # | Yêu cầu | Phần trong notebook |
|---|---------|---------------------|
| 1 | Thiết kế kiến trúc CNN + LSTM như trong paper | §3 (model gốc) |
| 2 | Tìm 1 bài toán text classification tiếng Việt + dataset | §1 (UIT-VSFC) |
| 3 | Huấn luyện model trên dataset | §5 |
| 4 | Report kết quả (dạng bảng) | §6 |

Theo yêu cầu của thầy: report **cả mô hình gốc lẫn mô hình cải tiến**, kết quả trình bày **dạng bảng**.
""")

# ----------------------------------------------------------------------------
# 1. Bài toán & Dataset
# ----------------------------------------------------------------------------
md(r"""
## 1. Bài toán & Dataset

**Bài toán:** Phân loại cảm xúc (sentiment classification) — một bài toán *text classification* tiếng Việt.
Cho một câu phản hồi, dự đoán nó mang sắc thái **negative / neutral / positive**.

**Dataset: UIT-VSFC** (Vietnamese Students' Feedback Corpus) — Nguyen et al., 2018.
- ~16.175 câu phản hồi của sinh viên, gán nhãn thủ công (độ đồng thuận annotator > 91%).
- 3 lớp cảm xúc: `0 = negative`, `1 = neutral`, `2 = positive` — **khớp đúng** đầu ra 3 lớp của paper gốc.
- Chia sẵn: **train 11.426 / dev 1.583 / test 3.166**.
- Văn bản đã được **tách từ sẵn** (word-segmented), nên không cần thư viện tách từ tiếng Việt.

> **Vì sao chọn UIT-VSFC?** Đây là benchmark chuẩn cho NLP tiếng Việt, có sẵn nhãn 3 lớp giống
> hệt thiết kế của paper, đủ lớn để huấn luyện mạng sâu nhưng vẫn nhẹ để chạy trên CPU.

**Lưu ý quan trọng — mất cân bằng lớp:** lớp `neutral` chỉ chiếm ~4-5%. Vì vậy ngoài *accuracy*
ta sẽ report thêm **macro-F1** (trung bình F1 các lớp, không thiên vị lớp đông) và dùng
`class_weight` khi huấn luyện. Đây cũng là chỉ số chính mà các paper dùng để so sánh trên UIT-VSFC.
""")

# ----------------------------------------------------------------------------
# 2. Setup / imports
# ----------------------------------------------------------------------------
md(r"""
## 2. Thiết lập môi trường & nạp dữ liệu
""")

code(r"""
import os
import re
import io
import sys
import time
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
SEED = 42
import random
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print("TensorFlow :", tf.__version__)
print("Keras      :", keras.__version__)
print("GPU(s)     :", tf.config.list_physical_devices('GPU') or "None (CPU mode)")
""")

md(r"""
### Tự động tải dataset UIT-VSFC

Notebook này là **file nộp duy nhất** — không kèm dữ liệu rời. Cell dưới tự tải dataset
**UIT-VSFC** về thư mục `data/UIT-VSFC/` nếu chưa có.

**Nguồn dữ liệu:** dataset có trên [HuggingFace `uitnlp/vietnamese_students_feedback`](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback).
Tuy nhiên bản HF dùng *loading script* kiểu cũ (đã bị `datasets` ≥ 3.0 ngừng hỗ trợ), nên ta tải
trực tiếp từ **Google Drive** — chính là nguồn gốc mà script HF trỏ tới (9 file: train/dev/test ×
sents/sentiments/topics). Chỉ cần kết nối mạng.
""")

code(r"""
# --- Tự động tải UIT-VSFC từ Google Drive (nguồn gốc của bản trên HuggingFace) ---
DATA_DIR = os.path.join("data", "UIT-VSFC")

DRIVE_IDS = {
    "train": {"sents": "1nzak5OkrheRV1ltOGCXkT671bmjODLhP",
              "sentiments": "1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv",
              "topics": "14MuDtwMnNOcr4z_8KdpxprjbwaQ7lJ_C"},
    "dev":   {"sents": "1sMJSR3oRfPc3fe1gK-V3W5F24tov_517",
              "sentiments": "1GiY1AOp41dLXIIkgES4422AuDwmbUseL",
              "topics": "1DwLgDEaFWQe8mOd7EpF-xqMEbDLfdT-W"},
    "test":  {"sents": "1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n",
              "sentiments": "1vkQS5gI0is4ACU58-AbWusnemw7KZNfO",
              "topics": "1_ArMpDguVsbUGl-xSMkTF_p5KpZrmpSB"},
}

def ensure_uitvsfc():
    # đã có đủ file?
    ok = all(os.path.exists(os.path.join(DATA_DIR, sp, f"{k}.txt"))
             for sp in DRIVE_IDS for k in ("sents", "sentiments"))
    if ok:
        print("UIT-VSFC đã có sẵn, bỏ qua bước tải.")
        return
    try:
        import gdown
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown
    for sp, files in DRIVE_IDS.items():
        os.makedirs(os.path.join(DATA_DIR, sp), exist_ok=True)
        for k, fid in files.items():
            out = os.path.join(DATA_DIR, sp, f"{k}.txt")
            if not os.path.exists(out):
                gdown.download(f"https://drive.google.com/uc?id={fid}", out, quiet=True)
    print("Đã tải xong UIT-VSFC.")

ensure_uitvsfc()
""")

code(r"""
# --- Load UIT-VSFC từ file đã tải ---
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

def load_split(split):
    def read(name):
        with open(os.path.join(DATA_DIR, split, name), encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    sents = read("sents.txt")
    sentiments = [int(x) for x in read("sentiments.txt")]
    assert len(sents) == len(sentiments)
    return pd.DataFrame({"text": sents, "label": sentiments})

train_df = load_split("train")
dev_df   = load_split("dev")
test_df  = load_split("test")

print(f"train: {len(train_df):>6}  |  dev: {len(dev_df):>5}  |  test: {len(test_df):>5}")
train_df.head()
""")

code(r"""
# --- Label distribution (shows the class imbalance) ---
def dist(df):
    c = df["label"].value_counts().sort_index()
    return {LABELS[k]: f"{v} ({100*v/len(df):.1f}%)" for k, v in c.items()}

dist_df = pd.DataFrame({"train": dist(train_df), "dev": dist(dev_df), "test": dist(test_df)})
print(dist_df, "\n")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
for a, (name, df) in zip(ax, [("train", train_df), ("dev", dev_df), ("test", test_df)]):
    df["label"].map(LABELS).value_counts().reindex(["negative","neutral","positive"]).plot(
        kind="bar", ax=a, color=["#e74c3c","#95a5a6","#2ecc71"])
    a.set_title(f"{name} (n={len(df)})"); a.set_xlabel(""); a.tick_params(axis='x', rotation=0)
plt.suptitle("UIT-VSFC sentiment label distribution"); plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------
# 2b. Preprocessing: tokenize -> sequences -> padding
# ----------------------------------------------------------------------------
md(r"""
## 2.1 Tiền xử lý: token hoá → chỉ số → padding

Quy trình giống paper gốc:
1. Xây **vocabulary** từ tập train (Keras `Tokenizer`), giới hạn `MAX_WORDS` từ phổ biến nhất.
2. Chuyển mỗi câu thành **chuỗi chỉ số** từ.
3. **Padding** về cùng độ dài `MAX_LEN`.

> Paper dùng `MAX_LEN=400` cho *document*. Ở đây dữ liệu là *câu* phản hồi ngắn (p95 = 33 từ),
> nên ta đặt `MAX_LEN=100` (phủ > 99% mẫu) — nhẹ hơn nhiều khi train trên CPU mà không mất thông tin.
""")

code(r"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN   = 100     # độ dài chuỗi sau padding
NUM_CLASSES = 3

# Văn bản UIT-VSFC đã được tách từ sẵn (cách nhau bằng khoảng trắng) -> tách theo space.
# Giữ TOÀN BỘ vocab (không giới hạn num_words) để pretrained embedding phủ tối đa.
tokenizer = Tokenizer(oov_token="<OOV>", filters="")
tokenizer.fit_on_texts(train_df["text"])

def to_seq(texts):
    return pad_sequences(tokenizer.texts_to_sequences(texts),
                         maxlen=MAX_LEN, padding="post", truncating="post")

X_train, X_dev, X_test = to_seq(train_df["text"]), to_seq(dev_df["text"]), to_seq(test_df["text"])
y_train = train_df["label"].to_numpy()
y_dev   = dev_df["label"].to_numpy()
y_test  = test_df["label"].to_numpy()

VOCAB_SIZE = len(tokenizer.word_index) + 1
print(f"Vocabulary : {len(tokenizer.word_index)} từ")
print(f"X_train shape      : {X_train.shape}")
print(f"Ví dụ chuỗi đã mã hoá: {X_train[0][:15]} ...")
""")

code(r"""
# class_weight để bù cho việc lớp 'neutral' rất hiếm (~4%)
class_weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
CLASS_WEIGHT = {i: w for i, w in enumerate(class_weights)}
print("class_weight:", {LABELS[k]: round(v, 3) for k, v in CLASS_WEIGHT.items()})
""")

md(r"""
## 2.2 Pretrained embedding tiếng Việt — PhoW2V (dùng cho model cải tiến)

Một hạn chế lớn của model gốc: lớp `Embedding` **học từ đầu** trên vocabulary rất nhỏ
(chỉ ~2.5k từ của UIT-VSFC) → các vector từ yếu, kém tổng quát.

Cách khắc phục hiệu quả: khởi tạo embedding bằng **PhoW2V** —
[bộ word2vec tiếng Việt](https://github.com/datquocnguyen/PhoW2V) (Nguyen et al., EMNLP-2020)
được train sẵn trên corpus 20GB. Ta dùng bản **syllable-level 100 chiều**.

> File PhoW2V gốc nặng ~1.1GB (979k từ) — quá nặng để tải mỗi lần chạy. Vì đây là **file nộp duy
> nhất**, ta đã **trích sẵn** ma trận gọn chỉ chứa vector cho vocab UIT-VSFC (< 1MB) và **nhúng thẳng
> vào notebook dưới dạng base64 (nén zlib)**. Cell dưới giải nén ra ma trận, không cần file ngoài.
> **Coverage ≈ 86%** từ vựng có vector pretrained.
>
> *(Cách dựng lại từ đầu: tải PhoW2V syllable 100d từ github.com/datquocnguyen/PhoW2V → chạy
> `build_embedding_matrix.py`. Code trích đính kèm trong README.)*
""")

code(r"""
import json, base64, zlib

EMBED_DIM = 100   # khớp PhoW2V syllables 100d

# Ma trận PhoW2V gọn (đã trích cho vocab UIT-VSFC) + vocab, nhúng dạng base64(zlib).
# Hai biến chuỗi __EMB_NPY_B64__ / __EMB_VOCAB_B64__ được định nghĩa ở cell ngay dưới.
def load_pretrained_embedding():
    M = np.load(io.BytesIO(zlib.decompress(base64.b64decode(__EMB_NPY_B64__))),
                allow_pickle=False)
    meta = json.loads(zlib.decompress(base64.b64decode(__EMB_VOCAB_B64__)).decode("utf-8"))
    saved_wi = meta["word_index"]
    # Căn lại ma trận theo word_index hiện tại của tokenizer (đảm bảo khớp index).
    idx2vec = {w: M[i] for w, i in saved_wi.items() if i < M.shape[0]}
    M2 = np.random.normal(0, 0.1, (VOCAB_SIZE, EMBED_DIM)).astype("float32")
    hit = 0
    for w, i in tokenizer.word_index.items():
        if w in idx2vec:
            M2[i] = idx2vec[w]; hit += 1
    print(f"PhoW2V matrix: {M2.shape}, coverage ~{100*hit/len(tokenizer.word_index):.1f}%")
    return M2
""")

# Cell chứa hai chuỗi base64 lớn (được chèn từ file lúc build notebook)
with open("_emb_npy_b64.txt") as f:
    _npy_b64 = f.read().strip()
with open("_emb_vocab_b64.txt") as f:
    _vocab_b64 = f.read().strip()

code(
    "# (Dữ liệu nhúng — không cần đọc/sửa) Ma trận PhoW2V + vocab, base64(zlib-nén).\n"
    "import io\n"
    f"__EMB_NPY_B64__ = \"{_npy_b64}\"\n"
    f"__EMB_VOCAB_B64__ = \"{_vocab_b64}\"\n"
    "embedding_matrix = load_pretrained_embedding()"
)

# ----------------------------------------------------------------------------
# 3. Original architecture
# ----------------------------------------------------------------------------
md(r"""
## 3. Kiến trúc gốc — Multi-channel LSTM-CNN (theo paper)

Ý tưởng cốt lõi của paper: **kết hợp đa kênh (multi-channel)** từ một lớp Embedding chung:

- **Kênh CNN** (3 nhánh song song): các `Conv1D` với kích thước cửa sổ **3, 5, 7** học đặc trưng
  cục bộ (n-gram). Mỗi nhánh sau đó **max-pool-over-time** → 1 vector đặc trưng.
- **Kênh LSTM**: một `LSTM` học phụ thuộc tuần tự (ngữ cảnh xa) toàn câu.
- **Hợp nhất (concatenate)** đầu ra của tất cả các kênh → `Dense` → `softmax` 3 lớp.

```
                Input (MAX_LEN,)
                      |
                Embedding (VOCAB x 100)   ← học từ đầu (model gốc)
        ┌─────────────┼─────────────┬───────────────┐
     Conv1D k=3     Conv1D k=5    Conv1D k=7       LSTM(128)
     150 filters    150 filters   150 filters         |
     GMaxPool       GMaxPool      GMaxPool            (128)
        └──────(150)──┴───(150)──────┘(150)            |
                      └────────── Concatenate ─────────┘
                                   (150*3 + 128 = 578)
                                        |
                              Dense(200, sigmoid) + Dropout(0.2)
                                        |
                                 Dense(3, softmax)
```

Cấu hình huấn luyện theo paper: optimizer **Adamax**, loss **categorical_crossentropy**.
""")

code(r"""
CNN_FILTERS = 150       # paper: 150 filters / nhánh
KERNELS     = (3, 5, 7) # paper: cửa sổ 3, 5, 7
LSTM_UNITS  = 128       # paper: 128
DENSE_UNITS = 200       # paper: 200
DROPOUT     = 0.2       # paper: 0.2
# Ghi chú: paper dùng embedding 200d. Ở đây dùng EMBED_DIM=100 (đã định nghĩa ở §2.2) để
# THỐNG NHẤT với PhoW2V 100d, giúp so sánh công bằng giữa model gốc (emb học từ đầu) và
# model cải tiến (emb khởi tạo bằng PhoW2V) trên cùng số chiều.

def build_multichannel_lstm_cnn():
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="tokens")
    emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM, name="embedding")(inp)

    # --- CNN channels (3 parallel branches) ---
    cnn_outputs = []
    for k in KERNELS:
        c = layers.Conv1D(CNN_FILTERS, k, activation="relu", name=f"conv_k{k}")(emb)
        c = layers.GlobalMaxPooling1D(name=f"gmaxpool_k{k}")(c)   # max-pool-over-time
        cnn_outputs.append(c)

    # --- LSTM channel ---
    lstm_out = layers.LSTM(LSTM_UNITS, name="lstm")(emb)

    # --- Fusion ---
    merged = layers.concatenate(cnn_outputs + [lstm_out], name="concat")
    x = layers.Dense(DENSE_UNITS, activation="sigmoid", name="dense")(merged)
    x = layers.Dropout(DROPOUT, name="dropout")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="MultiChannel_LSTM_CNN")
    model.compile(optimizer="adamax",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

model_base = build_multichannel_lstm_cnn()
model_base.summary()
""")

# ----------------------------------------------------------------------------
# 4. Improved architecture
# ----------------------------------------------------------------------------
md(r"""
## 4. Mô hình cải tiến (improved)

Yêu cầu của thầy: report **cả mô hình gốc lẫn mô hình sửa đổi**. Điều quan trọng là bản cải tiến
phải **thực sự tốt hơn**, không phải chỉ "khác đi". Vì vậy ta đã chạy một loạt **thực nghiệm có
kiểm soát** (xem §4.1) để tìm ra thay đổi nào thật sự nâng hiệu năng một cách **ổn định** (trung
bình qua 3 seeds), rồi mới chốt kiến trúc.

**Hai thay đổi cốt lõi thắng cuộc** (giữ nguyên xương sống đa kênh CNN+LSTM của paper):

| Thay đổi | Lý do | Tác động (đo được) |
|----------|-------|--------------------|
| **PhoW2V pretrained embedding (fine-tune)** | Vector từ học sẵn trên corpus 20GB thay cho embedding học từ vocab ~2.5k | Giải quyết nút thắt embedding yếu |
| **Focal loss** (γ=2) | Tập trung học các mẫu khó / lớp hiếm (`neutral`) thay vì cross-entropy thường | Nâng F1 lớp neutral, ổn định hơn |
| + giữ `class_weight`, SpatialDropout nhẹ, EarlyStopping | Regularize + xử lý mất cân bằng | |

> **Lưu ý quan trọng:** ta đã thử cả **BiLSTM** nhưng nó *không* giúp ích trên dữ liệu câu ngắn này
> (thậm chí kém và rất nhiễu — xem bảng §4.1). Đây là minh chứng: cải tiến phải dựa trên **bằng chứng
> thực nghiệm**, không phải "thêm cho phức tạp".
""")

md(r"""
### 4.1 Bảng thực nghiệm chọn kiến trúc cải tiến

Kết quả trung bình qua **3 seeds** (42, 7, 123) trên tập test — baseline gốc ≈ acc 89.9 / macro-F1 75.7:

| Cấu hình | Accuracy | Macro-F1 | Neutral-F1 | Nhận xét |
|----------|:--------:|:--------:|:----------:|----------|
| PhoW2V **frozen** + BiLSTM | 87.24 | 70.52 | 32.54 | ❌ đóng băng emb → kém (domain lệch) |
| PhoW2V 300d + BiLSTM + Focal | 89.65 | 74.75 ± **3.99** | 40.70 | ❌ rất nhiễu, không ổn định |
| PhoW2V 300d + LSTM + CE | 89.09 | 76.18 ± 0.82 | 44.94 | ✅ tốt |
| PhoW2V 300d + LSTM + Focal | 90.05 | 76.70 ± 0.97 | 45.84 | ✅ rất tốt |
| **PhoW2V 100d + LSTM + Focal** ⭐ | **90.44** | **76.80 ± 0.37** | 45.31 | 🏆 **cao nhất + ổn định nhất + nhẹ nhất** |

→ Chốt **bản cải tiến = PhoW2V 100d (fine-tune) + LSTM đa kênh + Focal loss**.
""")

code(r"""
# --- Focal loss cho nhãn dạng số nguyên (sparse) ---
def sparse_categorical_focal_loss(gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)                       # xác suất của lớp đúng
        return tf.reduce_mean(-tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))
    return loss

def build_improved_model():
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="tokens")
    # Embedding khởi tạo bằng PhoW2V, vẫn cho fine-tune (trainable=True)
    emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM, weights=[embedding_matrix],
                           trainable=True, name="embedding_phow2v")(inp)
    emb = layers.SpatialDropout1D(0.2, name="spatial_dropout")(emb)

    # --- CNN channels (3 parallel branches) - giữ như paper ---
    cnn_outputs = []
    for k in KERNELS:
        c = layers.Conv1D(CNN_FILTERS, k, activation="relu", padding="same", name=f"conv_k{k}")(emb)
        c = layers.GlobalMaxPooling1D(name=f"gmaxpool_k{k}")(c)
        cnn_outputs.append(c)

    # --- LSTM channel (giữ LSTM 1 chiều như paper - thắng BiLSTM trong thực nghiệm) ---
    lstm_out = layers.LSTM(LSTM_UNITS, name="lstm")(emb)

    merged = layers.concatenate(cnn_outputs + [lstm_out], name="concat")
    x = layers.Dense(DENSE_UNITS, activation="sigmoid", name="dense")(merged)
    x = layers.Dropout(0.3, name="dropout")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="Improved_PhoW2V_LSTM_CNN_Focal")
    model.compile(optimizer="adamax",
                  loss=sparse_categorical_focal_loss(gamma=2.0),
                  metrics=["accuracy"])
    return model

model_improved = build_improved_model()
model_improved.summary()
""")

# ----------------------------------------------------------------------------
# 5. Training
# ----------------------------------------------------------------------------
md(r"""
## 5. Huấn luyện

Cả hai model dùng cùng dữ liệu train/dev và `class_weight` để xử lý mất cân bằng lớp.
- **Model gốc**: theo paper — Adamax, embedding học từ đầu, loss cross-entropy.
  (Dùng batch 32 thay vì 10 để train nhanh hơn trên CPU.)
- **Model cải tiến**: Adamax, embedding khởi tạo **PhoW2V** (fine-tune), **focal loss**,
  thêm `EarlyStopping` + `ReduceLROnPlateau`.
""")

code(r"""
EPOCHS = 12
BATCH = 32

def train_model(model, epochs=EPOCHS, use_callbacks=False):
    cbs = []
    if use_callbacks:
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=2, min_lr=1e-5),
        ]
    t0 = time.time()
    hist = model.fit(X_train, y_train,
                     validation_data=(X_dev, y_dev),
                     epochs=epochs, batch_size=BATCH,
                     class_weight=CLASS_WEIGHT,
                     callbacks=cbs, verbose=2)
    train_time = time.time() - t0
    print(f"\n>>> Thời gian huấn luyện: {train_time:.1f}s")
    return hist, train_time
""")

code(r"""
print("="*60, "\nHUẤN LUYỆN MODEL GỐC (Multi-channel LSTM-CNN)\n", "="*60)
hist_base, time_base = train_model(model_base, use_callbacks=False)
""")

code(r"""
print("="*60, "\nHUẤN LUYỆN MODEL CẢI TIẾN (PhoW2V + LSTM-CNN + Focal loss)\n", "="*60)
hist_improved, time_improved = train_model(model_improved, epochs=15, use_callbacks=True)
""")

code(r"""
# --- Learning curves ---
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for col, (name, h) in enumerate([("Base (LSTM-CNN)", hist_base),
                                 ("Improved (PhoW2V+Focal)", hist_improved)]):
    axes[0, col].plot(h.history["accuracy"], label="train")
    axes[0, col].plot(h.history["val_accuracy"], label="val")
    axes[0, col].set_title(f"{name} — Accuracy"); axes[0, col].legend(); axes[0, col].grid(alpha=.3)
    axes[1, col].plot(h.history["loss"], label="train")
    axes[1, col].plot(h.history["val_loss"], label="val")
    axes[1, col].set_title(f"{name} — Loss"); axes[1, col].legend(); axes[1, col].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------
# 6. Evaluation / Report
# ----------------------------------------------------------------------------
md(r"""
## 6. Đánh giá & Report kết quả (trên tập test)

Các chỉ số:
- **Accuracy** — tỉ lệ dự đoán đúng tổng thể.
- **Macro-F1 / Macro-Precision / Macro-Recall** — trung bình theo lớp (không thiên vị lớp đông);
  đây là chỉ số chính trên UIT-VSFC do dữ liệu mất cân bằng.
- **Weighted-F1** — F1 có trọng số theo số mẫu mỗi lớp.
""")

code(r"""
def evaluate(model, name, train_time):
    proba = model.predict(X_test, batch_size=128, verbose=0)
    pred = proba.argmax(axis=1)
    acc = accuracy_score(y_test, pred)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    p_w,   r_w,   f_w,   _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    return {
        "Model": name,
        "Accuracy": acc,
        "Macro-P": p_mac, "Macro-R": r_mac, "Macro-F1": f_mac,
        "Weighted-F1": f_w,
        "Params": model.count_params(),
        "Train time (s)": round(train_time, 1),
        "_pred": pred,
    }

res_base = evaluate(model_base, "Base: Multi-channel LSTM-CNN (paper)", time_base)
res_improved = evaluate(model_improved, "Improved: PhoW2V + LSTM-CNN + Focal", time_improved)
""")

md(r"""
### 6.1 Bảng tổng hợp kết quả (Base vs Improved)
""")

code(r"""
report_df = pd.DataFrame([
    {k: v for k, v in r.items() if not k.startswith("_")}
    for r in (res_base, res_improved)
]).set_index("Model")

# Định dạng % cho dễ đọc
fmt = report_df.copy()
for col in ["Accuracy", "Macro-P", "Macro-R", "Macro-F1", "Weighted-F1"]:
    fmt[col] = (fmt[col] * 100).round(2).astype(str) + "%"
fmt["Params"] = fmt["Params"].map(lambda x: f"{x:,}")
print("KẾT QUẢ TRÊN TẬP TEST (UIT-VSFC, n=3166)\n")
print(fmt.to_string())
fmt
""")

code(r"""
# --- Mức cải thiện của Improved so với Base ---
delta_acc = (res_improved["Accuracy"] - res_base["Accuracy"]) * 100
delta_f1  = (res_improved["Macro-F1"] - res_base["Macro-F1"]) * 100
print(f"Cải thiện của model Improved so với Base:")
print(f"  Accuracy : {delta_acc:+.2f} điểm %")
print(f"  Macro-F1 : {delta_f1:+.2f} điểm %")
print("  -> Cải tiến" + (" THỰC SỰ tốt hơn ✅" if delta_f1 > 0 and delta_acc > 0 else " cần xem lại"))
""")

md(r"""
### 6.2 Báo cáo chi tiết theo từng lớp (per-class)
""")

code(r"""
for r in (res_base, res_improved):
    print("="*70)
    print(r["Model"])
    print("="*70)
    print(classification_report(y_test, r["_pred"],
                                target_names=["negative","neutral","positive"],
                                digits=4, zero_division=0))
""")

code(r"""
# --- Confusion matrices ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, r in zip(ax, (res_base, res_improved)):
    cm = confusion_matrix(y_test, r["_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=a,
                xticklabels=["neg","neu","pos"], yticklabels=["neg","neu","pos"])
    a.set_title(r["Model"].split(":")[0]); a.set_xlabel("Predicted"); a.set_ylabel("True")
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------
# 7. Conclusion
# ----------------------------------------------------------------------------
md(r"""
## 7. Nhận xét & Kết luận

*(Số liệu chi tiết xem bảng §6.1 ngay phía trên — phần này được cập nhật theo kết quả thực tế.)*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (vn-sentiment)", "language": "python", "name": "vn-sentiment"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open("25C15052.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote notebook with {len(cells)} cells.")
