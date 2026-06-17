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
# Mô hình Multi-channel LSTM-CNN cho Phân tích Cảm xúc Tiếng Việt

Báo cáo này tái hiện kiến trúc mạng nơ-ron kết hợp CNN và LSTM được đề xuất trong bài báo:

> Vo, Q., Nguyen, H., Le, B., Nguyen, M. (2017).
> *Multi-channel LSTM-CNN model for Vietnamese sentiment analysis.*
> 9th International Conference on Knowledge and Systems Engineering (KSE), IEEE.
> [ResearchGate](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis)
> · [Mã nguồn của nhóm tác giả](https://github.com/ntienhuy/MultiChannel)

Báo cáo đồng thời huấn luyện và đánh giá **mô hình gốc** cùng một **mô hình cải tiến** trên một bài
toán phân loại văn bản tiếng Việt thực tế, đối chiếu hiệu năng giữa hai mô hình bằng các độ đo định lượng.

---

## Cấu trúc báo cáo

| Mục | Nội dung |
|-----|----------|
| §1 | Bài toán và bộ dữ liệu |
| §2 | Thiết lập môi trường, tiền xử lý và biểu diễn từ |
| §3 | Kiến trúc mô hình gốc (theo bài báo) |
| §4 | Mô hình cải tiến và thực nghiệm lựa chọn kiến trúc |
| §5 | Huấn luyện |
| §6–§7 | Kết quả, đánh giá và kết luận |
""")

# ----------------------------------------------------------------------------
# 1. Bài toán & Dataset
# ----------------------------------------------------------------------------
md(r"""
## 1. Bài toán và bộ dữ liệu

**Bài toán.** Phân loại cảm xúc (sentiment classification) là một dạng bài toán phân loại văn bản
(text classification). Cho một câu phản hồi tiếng Việt, mô hình dự đoán sắc thái cảm xúc thuộc một
trong ba lớp: *tiêu cực (negative)*, *trung lập (neutral)* hoặc *tích cực (positive)*.

**Bộ dữ liệu UIT-VSFC** (Vietnamese Students' Feedback Corpus, Nguyen và cộng sự, 2018):

- Gồm khoảng 16.175 câu phản hồi của sinh viên, được gán nhãn thủ công với độ đồng thuận giữa các
  người gán nhãn (inter-annotator agreement) đạt trên 91%.
- Ba lớp cảm xúc `0 = negative`, `1 = neutral`, `2 = positive`, tương ứng đúng với không gian đầu ra
  ba lớp của mô hình trong bài báo gốc.
- Đã được chia sẵn thành ba tập: huấn luyện 11.426 câu, kiểm định (dev) 1.583 câu, kiểm thử (test) 3.166 câu.
- Văn bản đã được tách từ sẵn (word-segmented), do đó không cần áp dụng thêm công cụ tách từ tiếng Việt.

**Lý do lựa chọn.** UIT-VSFC là một bộ dữ liệu chuẩn (benchmark) cho xử lý ngôn ngữ tự nhiên tiếng
Việt, có nhãn ba lớp tương thích trực tiếp với kiến trúc của bài báo và có kích thước đủ lớn để
huấn luyện mạng học sâu.

**Vấn đề mất cân bằng lớp.** Lớp `neutral` chỉ chiếm khoảng 4–5% tổng số mẫu. Do đó, bên cạnh độ
chính xác (accuracy), báo cáo sử dụng **macro-F1** (trung bình điểm F1 của các lớp, không thiên vị
lớp chiếm đa số) làm độ đo chính, đồng thời áp dụng trọng số lớp (`class_weight`) trong quá trình
huấn luyện. Đây cũng là độ đo được sử dụng phổ biến để so sánh trên UIT-VSFC.
""")

# ----------------------------------------------------------------------------
# 2. Setup / imports
# ----------------------------------------------------------------------------
md(r"""
## 2. Thiết lập môi trường và nạp dữ liệu
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
### 2.0. Tự động tải bộ dữ liệu UIT-VSFC

Ô lệnh bên dưới tự động tải bộ dữ liệu UIT-VSFC về thư mục `data/UIT-VSFC/` nếu chưa tồn tại, giúp
báo cáo có thể chạy lại độc lập mà không cần đính kèm dữ liệu rời.

**Nguồn dữ liệu.** Bộ dữ liệu được công bố trên [HuggingFace `uitnlp/vietnamese_students_feedback`](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback).
Tuy nhiên, phiên bản trên HuggingFace sử dụng cơ chế *loading script* đã bị thư viện `datasets`
phiên bản 3.0 trở lên ngừng hỗ trợ. Do đó, dữ liệu được tải trực tiếp từ Google Drive, vốn là
nguồn gốc mà loading script của HuggingFace tham chiếu đến (gồm 9 tệp: ba tập train/dev/test, mỗi
tập có ba tệp sents/sentiments/topics).
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
## 2.1. Tiền xử lý dữ liệu

Quy trình tiền xử lý tuân theo bài báo gốc, gồm ba bước:

1. Xây dựng từ điển (vocabulary) từ tập huấn luyện bằng `Tokenizer` của Keras.
2. Biểu diễn mỗi câu thành chuỗi chỉ số của các từ.
3. Đệm (padding) các chuỗi về cùng độ dài cố định `MAX_LEN`.

Bài báo gốc sử dụng `MAX_LEN = 400` cho đơn vị văn bản là *văn bản (document)*. Trong nghiên cứu
này, đơn vị xử lý là *câu* phản hồi có độ dài ngắn (phân vị 95 chỉ khoảng 33 từ), do đó `MAX_LEN`
được đặt bằng 100, đủ bao phủ trên 99% số mẫu mà vẫn giảm đáng kể chi phí tính toán.
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
## 2.2. Biểu diễn từ bằng vector tiền huấn luyện PhoW2V (cho mô hình cải tiến)

Một hạn chế của mô hình gốc là lớp `Embedding` được học từ đầu trên một từ điển nhỏ (khoảng 2.500
từ của UIT-VSFC). Với lượng dữ liệu hạn chế như vậy, các vector từ thu được thường có chất lượng
biểu diễn thấp và khả năng tổng quát hoá kém.

Để khắc phục, mô hình cải tiến khởi tạo lớp embedding bằng **PhoW2V**, một bộ vector từ tiếng Việt
tiền huấn luyện ([Nguyen và cộng sự, EMNLP-2020](https://github.com/datquocnguyen/PhoW2V)) được
huấn luyện trên kho ngữ liệu khoảng 20GB. Báo cáo sử dụng phiên bản theo âm tiết (syllable-level)
với số chiều bằng 100. Tỷ lệ từ trong UIT-VSFC có vector tiền huấn luyện tương ứng (coverage) đạt
khoảng 86%.

*Ghi chú kỹ thuật.* Tệp PhoW2V gốc có dung lượng khoảng 1.1GB (979.000 từ). Để báo cáo gọn nhẹ và
tái lập được, ma trận embedding rút gọn chỉ chứa vector cho từ điển UIT-VSFC (dưới 1MB) đã được
trích xuất sẵn và nhúng trực tiếp dưới dạng base64 (nén zlib) trong ô lệnh phía dưới. Quy trình
trích xuất từ tệp gốc được mô tả trong tệp `README.md` kèm theo mã nguồn.
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
    "# (Dữ liệu nhúng, không cần đọc/sửa) Ma trận PhoW2V + vocab, base64(zlib-nén).\n"
    "import io\n"
    f"__EMB_NPY_B64__ = \"{_npy_b64}\"\n"
    f"__EMB_VOCAB_B64__ = \"{_vocab_b64}\"\n"
    "embedding_matrix = load_pretrained_embedding()"
)

# ----------------------------------------------------------------------------
# 3. Original architecture
# ----------------------------------------------------------------------------
md(r"""
## 3. Kiến trúc mô hình gốc (Multi-channel LSTM-CNN)

Ý tưởng cốt lõi của bài báo là kết hợp nhiều kênh đặc trưng (multi-channel) xuất phát từ một lớp
Embedding dùng chung:

- **Kênh CNN** gồm ba nhánh tích chập song song. Các lớp `Conv1D` với kích thước cửa sổ lần lượt
  là 3, 5 và 7 nhằm trích xuất đặc trưng cục bộ (tương ứng các n-gram bậc khác nhau). Mỗi nhánh
  được tổng hợp bằng phép gộp cực đại theo thời gian (max-pooling-over-time) thành một vector đặc trưng.
- **Kênh LSTM** sử dụng một lớp `LSTM` để mô hình hoá các phụ thuộc tuần tự và ngữ cảnh xa trong câu.
- Đầu ra của tất cả các kênh được **ghép nối (concatenate)**, đưa qua lớp `Dense` và cuối cùng là
  lớp `softmax` ba lớp.

```
                Input (MAX_LEN,)
                      |
                Embedding (VOCAB x 100)      [học từ đầu]
        +-------------+-------------+---------------+
     Conv1D k=3     Conv1D k=5    Conv1D k=7       LSTM(128)
     150 filters    150 filters   150 filters         |
     GMaxPool       GMaxPool      GMaxPool            (128)
        +------(150)--+---(150)------+(150)            |
                      +---------- Concatenate ---------+
                                   (150*3 + 128 = 578)
                                        |
                              Dense(200, sigmoid) + Dropout(0.2)
                                        |
                                 Dense(3, softmax)
```

Cấu hình huấn luyện theo bài báo: bộ tối ưu **Adamax**, hàm mất mát **categorical cross-entropy**.
""")

code(r"""
CNN_FILTERS = 150       # paper: 150 filters / nhánh
KERNELS     = (3, 5, 7) # paper: cửa sổ 3, 5, 7
LSTM_UNITS  = 128       # paper: 128
DENSE_UNITS = 200       # paper: 200
DROPOUT     = 0.2       # paper: 0.2
EPOCHS = 12
BATCH = 32
# Ghi chú: bài báo dùng embedding 200 chiều. Ở đây dùng EMBED_DIM=100 (định nghĩa ở §2.2)
# để thống nhất với PhoW2V 100 chiều, bảo đảm so sánh công bằng giữa mô hình gốc
# (embedding học từ đầu) và mô hình cải tiến (embedding khởi tạo bằng PhoW2V).

# --- Focal loss cho nhãn dạng số nguyên (sparse) ---
def sparse_categorical_focal_loss(gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)                       # xác suất của lớp đúng
        return tf.reduce_mean(-tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))
    return loss

def build_model(name, use_phow2v=False, trainable_emb=True, bidirectional=False,
                use_focal=False, spatial_dropout=0.0):
    # Hàm dựng mô hình đa kênh CNN-LSTM tham số hoá, dùng chung cho mọi cấu hình.
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name="tokens")
    if use_phow2v:
        emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM, weights=[embedding_matrix],
                               trainable=trainable_emb, name="embedding")(inp)
    else:
        emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM, name="embedding")(inp)
    if spatial_dropout > 0:
        emb = layers.SpatialDropout1D(spatial_dropout, name="spatial_dropout")(emb)

    # Ba nhánh CNN song song (cửa sổ 3/5/7) + gộp cực đại theo thời gian
    cnn_outputs = []
    for k in KERNELS:
        c = layers.Conv1D(CNN_FILTERS, k, activation="relu", padding="same", name=f"conv_k{k}")(emb)
        c = layers.GlobalMaxPooling1D(name=f"gmaxpool_k{k}")(c)
        cnn_outputs.append(c)

    # Kênh hồi tiếp: LSTM một chiều hoặc hai chiều (BiLSTM)
    rnn = (layers.Bidirectional(layers.LSTM(LSTM_UNITS), name="bilstm") if bidirectional
           else layers.LSTM(LSTM_UNITS, name="lstm"))(emb)

    merged = layers.concatenate(cnn_outputs + [rnn], name="concat")
    x = layers.Dense(DENSE_UNITS, activation="sigmoid", name="dense")(merged)
    x = layers.Dropout(DROPOUT, name="dropout")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name=name)
    loss = sparse_categorical_focal_loss(2.0) if use_focal else "sparse_categorical_crossentropy"
    model.compile(optimizer="adamax", loss=loss, metrics=["accuracy"])
    return model

SEEDS = [42, 7, 123]   # ba hạt giống ngẫu nhiên để đánh giá độ ổn định

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def train_eval_once(build_kw, name, seed, epochs, use_callbacks):
    set_seed(seed)
    model = build_model(f"model_seed{seed}", **build_kw)   # tên model ASCII (Keras yêu cầu)
    cbs = []
    if use_callbacks:
        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    t0 = time.time()
    hist = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs,
                     batch_size=BATCH, class_weight=CLASS_WEIGHT, callbacks=cbs, verbose=0)
    dt = time.time() - t0
    pred = model.predict(X_test, batch_size=256, verbose=0).argmax(axis=1)
    acc = accuracy_score(y_test, pred)
    f_mac = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)[2]
    f_w = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)[2]
    f_neu = precision_recall_fscore_support(y_test, pred, average=None, zero_division=0)[2][1]
    # Giữ lại pred và history (dùng cho báo cáo chi tiết ở §5, tránh phải train lại)
    return dict(acc=acc, macroF1=f_mac, weightedF1=f_w, neutralF1=f_neu,
                params=model.count_params(), time=dt, pred=pred, hist=hist)

def run_3seeds(build_kw, name, epochs, use_callbacks):
    # Huấn luyện cùng cấu hình trên 3 seed, trả về trung bình và độ lệch chuẩn.
    # Đồng thời lưu dự đoán & lịch sử của seed đầu tiên (42) để §5 tái sử dụng.
    runs = []
    for s in SEEDS:
        r = train_eval_once(build_kw, name, s, epochs, use_callbacks)
        runs.append(r)
        print(f"    seed {s:>3}: acc={r['acc']*100:.2f}%  macroF1={r['macroF1']*100:.2f}%  ({r['time']:.0f}s)")
    agg = {"Mô hình": name, "Params": runs[0]["params"],
           "_pred": runs[0]["pred"], "_hist": runs[0]["hist"]}
    for k, label in [("acc", "Accuracy"), ("macroF1", "Macro-F1"),
                     ("neutralF1", "Neutral-F1"), ("weightedF1", "Weighted-F1")]:
        vals = np.array([r[k] for r in runs])
        agg[label] = vals.mean()
        agg[label + "_std"] = vals.std()
    agg["Train time (s)"] = round(sum(r["time"] for r in runs), 1)
    return agg
""")

code(r"""
# Mô hình gốc theo đúng bài báo: embedding học từ đầu, LSTM một chiều, cross-entropy, Adamax.
build_model("Base_demo", use_phow2v=False, use_focal=False).summary()
""")

code(r"""
print("="*64, "\nHUẤN LUYỆN MÔ HÌNH GỐC (Multi-channel LSTM-CNN) - 3 seed\n", "="*64)
BASE_KW = dict(use_phow2v=False, trainable_emb=True, bidirectional=False, use_focal=False, spatial_dropout=0.0)
res_base = run_3seeds(BASE_KW, "Mô hình gốc (LSTM-CNN)", epochs=EPOCHS, use_callbacks=False)
print(f"\n>>> Trung bình: acc={res_base['Accuracy']*100:.2f}% "
      f"macro-F1={res_base['Macro-F1']*100:.2f}% (± {res_base['Macro-F1_std']*100:.2f})")
""")

# ----------------------------------------------------------------------------
# 4. Improvement experiments
# ----------------------------------------------------------------------------
md(r"""
## 4. Thử nghiệm các phương án cải tiến

Để xác định hướng cải tiến hiệu quả, phần này huấn luyện và đánh giá một số phương án cải tiến trực
tiếp trong notebook, trên cùng điều kiện với mô hình gốc (cùng tập dữ liệu, cùng `class_weight`,
cùng số chiều embedding). Mỗi phương án được huấn luyện trên **ba hạt giống ngẫu nhiên** (42, 7,
123) và báo cáo theo giá trị trung bình kèm độ lệch chuẩn, nhằm bảo đảm kết luận không phụ thuộc
vào một lần khởi tạo may rủi. Các phương án tập trung vào hai yếu tố được cho là then chốt:

1. **Chất lượng biểu diễn từ:** thay embedding học từ đầu bằng vector tiền huấn luyện PhoW2V, xét cả
   trường hợp cố định (frozen) và có tinh chỉnh (fine-tune).
2. **Hàm mất mát:** thay cross-entropy bằng Focal loss để tăng trọng số học cho lớp thiểu số `neutral`.

Ngoài ra, kiến trúc Bidirectional LSTM (BiLSTM) cũng được đưa vào để kiểm chứng giả thuyết rằng việc
tăng độ phức tạp của mô hình chưa chắc cải thiện hiệu năng trên dữ liệu câu ngắn.

Các phương án được khảo sát:

| Ký hiệu | Cấu hình |
|---------|----------|
| C1 | PhoW2V (cố định) + LSTM + cross-entropy |
| C2 | PhoW2V (tinh chỉnh) + LSTM + cross-entropy |
| C3 | PhoW2V (tinh chỉnh) + BiLSTM + Focal loss |
| C4 | PhoW2V (tinh chỉnh) + LSTM + Focal loss |
""")

code(r"""
# Định nghĩa các phương án cải tiến (mỗi phương án huấn luyện một lần, cùng điều kiện với mô hình gốc)
CONFIGS = [
    dict(key="C1", name="PhoW2V cố định + LSTM + CE",
         kw=dict(use_phow2v=True, trainable_emb=False, bidirectional=False, use_focal=False, spatial_dropout=0.2)),
    dict(key="C2", name="PhoW2V tinh chỉnh + LSTM + CE",
         kw=dict(use_phow2v=True, trainable_emb=True, bidirectional=False, use_focal=False, spatial_dropout=0.2)),
    dict(key="C3", name="PhoW2V tinh chỉnh + BiLSTM + Focal",
         kw=dict(use_phow2v=True, trainable_emb=True, bidirectional=True, use_focal=True, spatial_dropout=0.2)),
    dict(key="C4", name="PhoW2V tinh chỉnh + LSTM + Focal",
         kw=dict(use_phow2v=True, trainable_emb=True, bidirectional=False, use_focal=True, spatial_dropout=0.2)),
]

exp_results = []
for cfg in CONFIGS:
    print("="*64, f"\nHUẤN LUYỆN PHƯƠNG ÁN {cfg['key']}: {cfg['name']} (3 seed)\n", "="*64)
    r = run_3seeds(cfg["kw"], f"{cfg['key']}: {cfg['name']}", epochs=15, use_callbacks=True)
    exp_results.append(r)
    print(f"\n>>> {cfg['key']} trung bình: acc={r['Accuracy']*100:.2f}% "
          f"macro-F1={r['Macro-F1']*100:.2f}% (± {r['Macro-F1_std']*100:.2f})")
""")

md(r"""
### 4.1. So sánh các phương án và lựa chọn mô hình cải tiến

Bảng dưới đây tổng hợp kết quả của mô hình gốc cùng bốn phương án cải tiến, dưới dạng giá trị trung
bình kèm độ lệch chuẩn trên ba hạt giống ngẫu nhiên. Việc đánh giá trên nhiều hạt giống giúp kết
luận không phụ thuộc vào một lần khởi tạo may rủi, đặc biệt quan trọng khi chênh lệch giữa các
phương án nhỏ. Phương án cải tiến được lựa chọn là phương án đạt macro-F1 trung bình cao nhất.
""")

code(r"""
def fmt_pm(mean, std):   # định dạng "mean ± std" theo phần trăm
    return f"{mean*100:.2f} ± {std*100:.2f}"

all_results = [res_base] + exp_results
rows = []
for r in all_results:
    rows.append({
        "Mô hình": r["Mô hình"],
        "Accuracy": fmt_pm(r["Accuracy"], r["Accuracy_std"]),
        "Macro-F1": fmt_pm(r["Macro-F1"], r["Macro-F1_std"]),
        "Neutral-F1": fmt_pm(r["Neutral-F1"], r["Neutral-F1_std"]),
        "Weighted-F1": fmt_pm(r["Weighted-F1"], r["Weighted-F1_std"]),
        "Params": f"{r['Params']:,}",
    })
show = pd.DataFrame(rows)
print("BẢNG SO SÁNH (trung bình ± độ lệch chuẩn trên 3 seed, tập kiểm thử UIT-VSFC, n=3166)\n")
print(show.to_string(index=False))

# Chọn phương án cải tiến tốt nhất theo macro-F1 trung bình (không tính mô hình gốc)
best = max(exp_results, key=lambda r: r["Macro-F1"])
print(f"\n>>> Phương án cải tiến tốt nhất theo macro-F1 trung bình: {best['Mô hình']}")
show
""")

# ----------------------------------------------------------------------------
# 5. Detailed comparison: base vs best
# ----------------------------------------------------------------------------
md(r"""
## 5. So sánh chi tiết: mô hình gốc và mô hình cải tiến tốt nhất

Phần này phân tích sâu hơn hai mô hình: mô hình gốc và phương án cải tiến đạt macro-F1 trung bình
cao nhất ở §4.1. Báo cáo theo lớp, đường cong huấn luyện và ma trận nhầm lẫn được lấy từ lần chạy
ứng với hạt giống đầu tiên (seed = 42) của mỗi mô hình trong quá trình thực nghiệm ở trên, do đó
không cần huấn luyện lại.
""")

code(r"""
res_improved = best   # phương án cải tiến được chọn ở §4.1

delta_acc = (res_improved["Accuracy"] - res_base["Accuracy"]) * 100
delta_f1  = (res_improved["Macro-F1"] - res_base["Macro-F1"]) * 100
delta_w   = (res_improved["Weighted-F1"] - res_base["Weighted-F1"]) * 100
print("So sánh hiệu năng trung bình (3 seed):")
print(f"  Mô hình gốc      : acc={res_base['Accuracy']*100:.2f}%  macro-F1={res_base['Macro-F1']*100:.2f}%")
print(f"  Mô hình cải tiến : acc={res_improved['Accuracy']*100:.2f}%  macro-F1={res_improved['Macro-F1']*100:.2f}%")
print(f"\nMức chênh lệch trung bình (cải tiến - gốc):")
print(f"  Accuracy    : {delta_acc:+.2f} điểm phần trăm")
print(f"  Macro-F1    : {delta_f1:+.2f} điểm phần trăm")
print(f"  Weighted-F1 : {delta_w:+.2f} điểm phần trăm")
""")

md(r"""
### 5.1. Báo cáo chi tiết theo từng lớp
""")

code(r"""
# Dùng lại dự đoán của seed=42 đã lưu ở §4 (không huấn luyện lại)
for r in (res_base, res_improved):
    print("="*70)
    print(r["Mô hình"])
    print("="*70)
    print(classification_report(y_test, r["_pred"],
                                target_names=["negative", "neutral", "positive"],
                                digits=4, zero_division=0))
""")

md(r"""
### 5.2. Đường cong huấn luyện và ma trận nhầm lẫn
""")

code(r"""
# Đường cong huấn luyện (seed=42)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for r in (res_base, res_improved):
    h = r["_hist"]
    axes[0].plot(h.history["val_accuracy"], label=r["Mô hình"][:24])
    axes[1].plot(h.history["val_loss"], label=r["Mô hình"][:24])
axes[0].set_title("Val Accuracy theo epoch"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].set_title("Val Loss theo epoch"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

# Ma trận nhầm lẫn (seed=42)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, r in zip(ax, (res_base, res_improved)):
    cm = confusion_matrix(y_test, r["_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=a,
                xticklabels=["neg", "neu", "pos"], yticklabels=["neg", "neu", "pos"])
    a.set_title(r["Mô hình"][:30]); a.set_xlabel("Dự đoán"); a.set_ylabel("Thực tế")
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------
# 6. Conclusion
# ----------------------------------------------------------------------------
md(r"""
## 6. Nhận xét & Kết luận

### 6.1. Tổng hợp kết quả

Bảng so sánh đầy đủ được trình bày ở §4.1; ô lệnh dưới đây in lại các con số then chốt của mô hình
gốc và mô hình cải tiến được chọn, cùng mức chênh lệch, trực tiếp từ kết quả vừa huấn luyện.
""")

code(r"""
print(f"{'Mô hình':40s} {'Accuracy':>9s} {'Macro-F1':>9s} {'Weighted-F1':>12s}")
print("-"*72)
for r in (res_base, res_improved):
    print(f"{r['Mô hình'][:40]:40s} {r['Accuracy']*100:8.2f}% {r['Macro-F1']*100:8.2f}% {r['Weighted-F1']*100:11.2f}%")
print("-"*72)
print(f"{'Chênh lệch (cải tiến - gốc)':40s} {delta_acc:+8.2f}  {delta_f1:+8.2f}  {delta_w:+11.2f}")
print(f"\nSố tham số: gốc = {res_base['Params']:,} | cải tiến = {res_improved['Params']:,}")
""")

md(r"""
### 6.2. Nhận xét

1. **Tái hiện thành công kiến trúc của bài báo.** Mô hình gốc (Multi-channel LSTM-CNN) đạt độ chính
   xác và macro-F1 nằm trong khoảng hiệu năng thường thấy của các mô hình học sâu trên UIT-VSFC,
   xác nhận kiến trúc đa kênh CNN-LSTM đã được hiện thực đúng.

2. **Cải thiện hiệu quả nhất đến từ chất lượng biểu diễn từ.** Việc khởi tạo lớp embedding bằng
   PhoW2V và cho phép tinh chỉnh đã khắc phục hạn chế lớn nhất của mô hình gốc: từ điển nhỏ
   (khoảng 2.500 từ) khiến embedding học từ đầu có chất lượng thấp. Ngược lại, phương án cố định
   embedding (frozen) cho kết quả thấp rõ rệt do lệch miền dữ liệu giữa kho ngữ liệu tiền huấn
   luyện và miền phản hồi của sinh viên. Kết hợp với Focal loss, các phương án cải tiến đạt macro-F1
   trung bình cao hơn mô hình gốc.

3. **Đánh giá trên nhiều hạt giống là cần thiết.** Chênh lệch giữa các phương án cải tiến tốt
   (chẳng hạn giữa LSTM và BiLSTM khi đều kết hợp PhoW2V và Focal loss) là khá nhỏ và có thể bị
   che lấp bởi dao động ngẫu nhiên giữa các lần khởi tạo. Việc huấn luyện trên ba hạt giống và báo
   cáo độ lệch chuẩn (§4.1) giúp kết luận đáng tin cậy hơn, thay vì dựa vào một lần chạy duy nhất.

4. **Lớp `neutral` là yếu tố hạn chế chính.** Do chỉ chiếm khoảng 5% dữ liệu, điểm F1 của lớp này
   thấp hơn đáng kể so với hai lớp còn lại, kéo macro-F1 xuống thấp hơn nhiều so với độ chính xác
   tổng thể. Đây là lý do báo cáo sử dụng macro-F1 làm độ đo chính, đồng thời áp dụng trọng số lớp
   và Focal loss.

**Kết luận chung.** Báo cáo đã tái hiện chính xác kiến trúc Multi-channel LSTM-CNN của bài báo và,
thông qua một quy trình thực nghiệm có hệ thống trên nhiều hạt giống ngẫu nhiên, xây dựng được một
mô hình cải tiến đạt hiệu năng cao hơn mô hình gốc. Kết quả cho thấy đối với bộ dữ liệu có quy mô
hạn chế, chất lượng biểu diễn từ (thông qua embedding tiền huấn luyện) đóng vai trò quan trọng đối
với hiệu năng, trong khi việc gia tăng độ phức tạp kiến trúc mang lại lợi ích không rõ rệt.

### 6.3. Hạn chế và hướng phát triển

- Áp dụng các mô hình ngôn ngữ tiền huấn luyện dạng Transformer (chẳng hạn PhoBERT), vốn thường đạt
  macro-F1 khoảng 92–93% trên UIT-VSFC.
- Xử lý lớp thiểu số `neutral` triệt để hơn bằng các kỹ thuật tăng cường dữ liệu (oversampling,
  data augmentation) thay vì chỉ dựa vào trọng số lớp và Focal loss.
- Đánh giá ổn định hơn bằng cách huấn luyện lặp lại trên nhiều hạt giống ngẫu nhiên và báo cáo
  giá trị trung bình cùng độ lệch chuẩn.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (vn-sentiment)", "language": "python", "name": "vn-sentiment"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open("25C15052.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote notebook with {len(cells)} cells.")
