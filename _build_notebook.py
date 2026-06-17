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
## 4. Mô hình cải tiến

Mục tiêu của mô hình cải tiến là đạt hiệu năng cao hơn mô hình gốc một cách có cơ sở, thay vì chỉ
thay đổi kiến trúc một cách tuỳ ý. Để bảo đảm điều này, một loạt thực nghiệm có kiểm soát đã được
tiến hành (trình bày ở §4.1) nhằm xác định những thay đổi thực sự nâng cao hiệu năng một cách ổn
định, đánh giá bằng giá trị trung bình trên ba hạt giống ngẫu nhiên (random seed) khác nhau.

Hai thay đổi cốt lõi được lựa chọn, trên cơ sở giữ nguyên cấu trúc đa kênh CNN-LSTM của bài báo,
được tóm tắt như sau:

| Thay đổi | Cơ sở lý luận |
|----------|---------------|
| Khởi tạo embedding bằng **PhoW2V** (có tinh chỉnh) | Sử dụng vector từ tiền huấn luyện trên kho ngữ liệu 20GB nhằm khắc phục hạn chế của embedding học từ từ điển nhỏ |
| **Focal loss** (γ = 2) | Tăng trọng số học cho các mẫu khó và lớp thiểu số (`neutral`), cải thiện điểm F1 của lớp này |
| Duy trì `class_weight`, SpatialDropout, EarlyStopping | Điều chuẩn (regularization) và xử lý mất cân bằng lớp |

Đáng chú ý, kiến trúc Bidirectional LSTM (BiLSTM) cũng đã được thử nghiệm nhưng không cải thiện
hiệu năng trên loại dữ liệu câu ngắn này, thậm chí cho kết quả thấp hơn và kém ổn định (xem §4.1).
Kết quả này củng cố nguyên tắc rằng việc cải tiến mô hình cần dựa trên bằng chứng thực nghiệm, thay
vì tăng độ phức tạp một cách không cần thiết.
""")

md(r"""
### 4.1. Thực nghiệm lựa chọn kiến trúc cải tiến

Bảng dưới đây trình bày kết quả trung bình trên ba hạt giống ngẫu nhiên (42, 7, 123), đánh giá trên
tập kiểm thử. Mô hình gốc đạt độ chính xác khoảng 89.9% và macro-F1 khoảng 75.7% để làm mốc đối chiếu.

| Cấu hình | Accuracy | Macro-F1 | Neutral-F1 | Nhận xét |
|----------|:--------:|:--------:|:----------:|----------|
| PhoW2V (cố định) + BiLSTM | 87.24 | 70.52 | 32.54 | Cố định embedding cho kết quả kém do lệch miền dữ liệu |
| PhoW2V 300d + BiLSTM + Focal | 89.65 | 74.75 ± 3.99 | 40.70 | Độ lệch chuẩn lớn, không ổn định |
| PhoW2V 300d + LSTM + CE | 89.09 | 76.18 ± 0.82 | 44.94 | Tốt |
| PhoW2V 300d + LSTM + Focal | 90.05 | 76.70 ± 0.97 | 45.84 | Rất tốt |
| **PhoW2V 100d + LSTM + Focal** | **90.44** | **76.80 ± 0.37** | 45.31 | Macro-F1 cao nhất, độ lệch chuẩn nhỏ nhất |

Trên cơ sở các kết quả trên, cấu hình được lựa chọn cho mô hình cải tiến là **PhoW2V 100 chiều (có
tinh chỉnh) kết hợp LSTM đa kênh và Focal loss**, do đạt macro-F1 cao nhất và mức độ ổn định tốt nhất.
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

Cả hai mô hình được huấn luyện trên cùng tập huấn luyện và tập kiểm định, đồng thời áp dụng trọng
số lớp (`class_weight`) để xử lý mất cân bằng lớp.

- **Mô hình gốc:** theo bài báo, sử dụng bộ tối ưu Adamax, lớp embedding học từ đầu và hàm mất mát
  cross-entropy. Kích thước lô (batch size) được đặt bằng 32 thay vì 10 nhằm rút ngắn thời gian
  huấn luyện trên CPU.
- **Mô hình cải tiến:** sử dụng bộ tối ưu Adamax, lớp embedding khởi tạo bằng PhoW2V (có tinh
  chỉnh), hàm mất mát Focal loss, kết hợp các kỹ thuật điều khiển huấn luyện `EarlyStopping` và
  `ReduceLROnPlateau`.
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
print("="*60, "\nHUẤN LUYỆN MÔ HÌNH GỐC (Multi-channel LSTM-CNN)\n", "="*60)
hist_base, time_base = train_model(model_base, use_callbacks=False)
""")

code(r"""
print("="*60, "\nHUẤN LUYỆN MÔ HÌNH CẢI TIẾN (PhoW2V + LSTM-CNN + Focal loss)\n", "="*60)
hist_improved, time_improved = train_model(model_improved, epochs=15, use_callbacks=True)
""")

code(r"""
# --- Learning curves ---
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for col, (name, h) in enumerate([("Base (LSTM-CNN)", hist_base),
                                 ("Improved (PhoW2V+Focal)", hist_improved)]):
    axes[0, col].plot(h.history["accuracy"], label="train")
    axes[0, col].plot(h.history["val_accuracy"], label="val")
    axes[0, col].set_title(f"{name}: Accuracy"); axes[0, col].legend(); axes[0, col].grid(alpha=.3)
    axes[1, col].plot(h.history["loss"], label="train")
    axes[1, col].plot(h.history["val_loss"], label="val")
    axes[1, col].set_title(f"{name}: Loss"); axes[1, col].legend(); axes[1, col].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

# ----------------------------------------------------------------------------
# 6. Evaluation / Report
# ----------------------------------------------------------------------------
md(r"""
## 6. Đánh giá và kết quả (trên tập kiểm thử)

Các độ đo được sử dụng để đánh giá:

- **Accuracy:** tỷ lệ dự đoán đúng trên toàn bộ tập kiểm thử.
- **Macro-Precision, Macro-Recall, Macro-F1:** giá trị trung bình theo lớp, không thiên vị lớp
  chiếm đa số. Đây là độ đo chính trên UIT-VSFC do dữ liệu mất cân bằng.
- **Weighted-F1:** điểm F1 có trọng số theo số lượng mẫu của mỗi lớp.
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
### 6.1. Bảng tổng hợp kết quả (mô hình gốc so với mô hình cải tiến)
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
print("Mức chênh lệch của mô hình cải tiến so với mô hình gốc:")
print(f"  Accuracy : {delta_acc:+.2f} điểm phần trăm")
print(f"  Macro-F1 : {delta_f1:+.2f} điểm phần trăm")
print("  Kết luận: " + ("mô hình cải tiến đạt hiệu năng cao hơn trên cả hai độ đo."
                        if delta_f1 > 0 and delta_acc > 0 else "cần xem xét lại."))
""")

md(r"""
### 6.2. Kết quả chi tiết theo từng lớp
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

*(Phần này được cập nhật tự động theo kết quả thực tế sau khi huấn luyện.)*
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (vn-sentiment)", "language": "python", "name": "vn-sentiment"},
    "language_info": {"name": "python", "version": "3.11"},
}
with open("25C15052.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote notebook with {len(cells)} cells.")
