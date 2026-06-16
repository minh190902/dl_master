"""So sánh nhiều chiến lược cải tiến model Multi-channel LSTM-CNN trên UIT-VSFC.

Mỗi cấu hình chạy với cùng split/seed. In bảng kết quả cuối để chọn bản thắng.
"""
import os, time, json, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

SEED = 42
import random; random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR = os.path.join("data", "UIT-VSFC")
EMB_PATH = os.path.join("data", "phow2v", "word2vec_vi_syllables_100dims.txt")

def load_split(split):
    def read(name):
        with open(os.path.join(DATA_DIR, split, name), encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    return pd.DataFrame({"text": read("sents.txt"),
                         "label": [int(x) for x in read("sentiments.txt")]})

train_df, dev_df, test_df = load_split("train"), load_split("dev"), load_split("test")

MAX_LEN, NUM_CLASSES, EMBED_DIM = 100, 3, 100
# Không giới hạn vocab chặt -> giữ nhiều từ để pretrained embedding phủ tối đa
tok = Tokenizer(oov_token="<OOV>", filters="")
tok.fit_on_texts(train_df["text"])
to_seq = lambda t: pad_sequences(tok.texts_to_sequences(t), maxlen=MAX_LEN, padding="post", truncating="post")
X_train, X_dev, X_test = to_seq(train_df["text"]), to_seq(dev_df["text"]), to_seq(test_df["text"])
y_train, y_dev, y_test = (train_df["label"].to_numpy(), dev_df["label"].to_numpy(), test_df["label"].to_numpy())
word_index = tok.word_index
VOCAB_SIZE = len(word_index) + 1
cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
CLASS_WEIGHT = {i: w for i, w in enumerate(cw)}
print(f"Vocab size: {VOCAB_SIZE}")

# ---- Load PhoW2V & build embedding matrix ----
print("Loading PhoW2V (this reads a 1.1GB text file, ~1-2 min)...")
t0 = time.time()
emb_index = {}
with open(EMB_PATH, encoding="utf-8") as f:
    header = f.readline()  # "<n> <dim>"
    for line in f:
        parts = line.rstrip().split(" ")
        if len(parts) != EMBED_DIM + 1:
            continue
        word = parts[0]
        if word in word_index:           # chỉ giữ từ có trong vocab -> tiết kiệm RAM
            emb_index[word] = np.asarray(parts[1:], dtype="float32")
print(f"  loaded {len(emb_index)} matching vectors in {time.time()-t0:.0f}s")

emb_matrix = np.random.normal(0, 0.1, (VOCAB_SIZE, EMBED_DIM)).astype("float32")
hit = 0
for word, i in word_index.items():
    v = emb_index.get(word)
    if v is not None:
        emb_matrix[i] = v; hit += 1
coverage = 100 * hit / len(word_index)
print(f"Coverage: {hit}/{len(word_index)} = {coverage:.1f}% từ có pretrained vector")

# ---------------------------------------------------------------------------
KERNELS, CNN_FILTERS, LSTM_UNITS, DENSE_UNITS = (3,5,7), 150, 128, 200

def build(emb_init=None, trainable_emb=True, bilstm=False, spatial_do=0.0,
          dense_act="sigmoid", batchnorm=False, dropout=0.2, opt="adamax"):
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32")
    if emb_init is not None:
        emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM, weights=[emb_init],
                               trainable=trainable_emb)(inp)
    else:
        emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    if spatial_do > 0:
        emb = layers.SpatialDropout1D(spatial_do)(emb)
    cnn = []
    for k in KERNELS:
        c = layers.Conv1D(CNN_FILTERS, k, activation="relu", padding="same")(emb)
        cnn.append(layers.GlobalMaxPooling1D()(c))
    rnn = (layers.Bidirectional(layers.LSTM(LSTM_UNITS)) if bilstm
           else layers.LSTM(LSTM_UNITS))(emb)
    merged = layers.concatenate(cnn + [rnn])
    x = layers.Dense(DENSE_UNITS, activation=dense_act)(merged)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def run(name, model, epochs=15, use_es=True):
    cbs = []
    if use_es:
        cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
               keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    t0 = time.time()
    model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs,
              batch_size=32, class_weight=CLASS_WEIGHT, callbacks=cbs, verbose=0)
    dt = time.time() - t0
    pred = model.predict(X_test, batch_size=256, verbose=0).argmax(1)
    acc = accuracy_score(y_test, pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    fn = precision_recall_fscore_support(y_test, pred, average=None, zero_division=0)[2]
    print(f"[{name:42s}] acc={acc*100:5.2f}  macroF1={f*100:5.2f}  neutralF1={fn[1]*100:5.2f}  ({dt:.0f}s)")
    return {"name": name, "acc": acc, "macroF1": f, "neutralF1": fn[1], "time": dt}

results = []
# E0: baseline đã biết (LSTM, emb scratch, adamax) – để đối chiếu
results.append(run("E0 baseline (scratch emb, LSTM)",
                   build(emb_init=None, bilstm=False, dense_act="sigmoid", dropout=0.2, opt="adamax"),
                   epochs=12, use_es=False))
# E1: pretrained emb (frozen) + BiLSTM
results.append(run("E1 PhoW2V frozen + BiLSTM",
                   build(emb_init=emb_matrix, trainable_emb=False, bilstm=True,
                         spatial_do=0.3, dense_act="relu", batchnorm=True, dropout=0.5, opt=keras.optimizers.Adam(1e-3))))
# E2: pretrained emb (fine-tuned) + BiLSTM
results.append(run("E2 PhoW2V finetune + BiLSTM",
                   build(emb_init=emb_matrix, trainable_emb=True, bilstm=True,
                         spatial_do=0.3, dense_act="relu", batchnorm=True, dropout=0.5, opt=keras.optimizers.Adam(1e-3))))
# E3: pretrained emb (fine-tuned) + LSTM (giống paper hơn, chỉ đổi embedding)
results.append(run("E3 PhoW2V finetune + LSTM (paper-style)",
                   build(emb_init=emb_matrix, trainable_emb=True, bilstm=False,
                         spatial_do=0.2, dense_act="sigmoid", dropout=0.3, opt="adamax")))

print("\n" + "="*70)
print("BẢNG TỔNG HỢP (sắp theo macro-F1)")
print("="*70)
rdf = pd.DataFrame(results).sort_values("macroF1", ascending=False)
for col in ("acc","macroF1","neutralF1"):
    rdf[col] = (rdf[col]*100).round(2)
print(rdf.to_string(index=False))
with open("_experiment_results.json", "w", encoding="utf-8") as f:
    json.dump({"coverage": coverage, "vocab": VOCAB_SIZE, "results": results}, f, ensure_ascii=False, indent=2, default=float)
print("\nSaved _experiment_results.json")
