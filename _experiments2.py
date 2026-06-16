"""Vòng 2: focal loss + PhoW2V 300d, và multi-seed để xác nhận cải tiến thật.

So sánh trên cùng split. Mục tiêu: tìm cấu hình vượt baseline macro-F1 một cách ỔN ĐỊNH
(trung bình nhiều seed), không phải may rủi 1 lần.
"""
import os, time, json, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA_DIR = os.path.join("data", "UIT-VSFC")
EMB_100 = os.path.join("data", "phow2v", "word2vec_vi_syllables_100dims.txt")
EMB_300 = os.path.join("data", "phow2v", "word2vec_vi_syllables_300dims.txt")

def load_split(split):
    def read(name):
        with open(os.path.join(DATA_DIR, split, name), encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    return pd.DataFrame({"text": read("sents.txt"),
                         "label": [int(x) for x in read("sentiments.txt")]})

train_df, dev_df, test_df = load_split("train"), load_split("dev"), load_split("test")
MAX_LEN, NUM_CLASSES = 100, 3
tok = Tokenizer(oov_token="<OOV>", filters="")
tok.fit_on_texts(train_df["text"])
to_seq = lambda t: pad_sequences(tok.texts_to_sequences(t), maxlen=MAX_LEN, padding="post", truncating="post")
X_train, X_dev, X_test = to_seq(train_df["text"]), to_seq(dev_df["text"]), to_seq(test_df["text"])
y_train, y_dev, y_test = (train_df["label"].to_numpy(), dev_df["label"].to_numpy(), test_df["label"].to_numpy())
word_index = tok.word_index
VOCAB_SIZE = len(word_index) + 1
cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train)
CLASS_WEIGHT = {i: w for i, w in enumerate(cw)}

def build_emb_matrix(path, dim):
    print(f"  loading {path} ...", flush=True)
    idx = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            if parts[0] in word_index:
                idx[parts[0]] = np.asarray(parts[1:], dtype="float32")
    M = np.random.normal(0, 0.1, (VOCAB_SIZE, dim)).astype("float32")
    hit = 0
    for w, i in word_index.items():
        v = idx.get(w)
        if v is not None: M[i] = v; hit += 1
    print(f"  coverage {dim}d: {100*hit/len(word_index):.1f}%", flush=True)
    return M

EMB_M100 = build_emb_matrix(EMB_100, 100)
EMB_M300 = build_emb_matrix(EMB_300, 300)

KERNELS, CNN_FILTERS, LSTM_UNITS, DENSE_UNITS = (3,5,7), 150, 128, 200

# --- Focal loss for sparse labels ---
def sparse_categorical_focal_loss(gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        idx = tf.stack([tf.range(tf.shape(y_true)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)
        return tf.reduce_mean(-tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))
    return loss

def build(emb_init, dim, bilstm=False, loss="sparse_categorical_crossentropy",
          spatial_do=0.2, dense_act="sigmoid", dropout=0.3, opt_name="adamax"):
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32")
    emb = layers.Embedding(VOCAB_SIZE, dim, weights=[emb_init], trainable=True)(inp)
    if spatial_do > 0:
        emb = layers.SpatialDropout1D(spatial_do)(emb)
    cnn = []
    for k in KERNELS:
        c = layers.Conv1D(CNN_FILTERS, k, activation="relu", padding="same")(emb)
        cnn.append(layers.GlobalMaxPooling1D()(c))
    rnn = (layers.Bidirectional(layers.LSTM(LSTM_UNITS)) if bilstm else layers.LSTM(LSTM_UNITS))(emb)
    x = layers.concatenate(cnn + [rnn])
    x = layers.Dense(DENSE_UNITS, activation=dense_act)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    m = keras.Model(inp, out)
    opt = keras.optimizers.Adam(1e-3) if opt_name == "adam" else "adamax"
    m.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return m

def run_once(builder, seed, use_cw=True):
    import random; random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    m = builder()
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    m.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=15, batch_size=32,
          class_weight=(CLASS_WEIGHT if use_cw else None), callbacks=cbs, verbose=0)
    pred = m.predict(X_test, batch_size=256, verbose=0).argmax(1)
    acc = accuracy_score(y_test, pred)
    f = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)[2]
    fn = precision_recall_fscore_support(y_test, pred, average=None, zero_division=0)[2][1]
    return acc, f, fn

CONFIGS = {
    "C1 300d + LSTM + CE":        lambda: build(EMB_M300, 300, bilstm=False, loss="sparse_categorical_crossentropy", dense_act="sigmoid"),
    "C2 300d + LSTM + Focal":     lambda: build(EMB_M300, 300, bilstm=False, loss=sparse_categorical_focal_loss(2.0), dense_act="sigmoid"),
    "C3 100d + LSTM + Focal":     lambda: build(EMB_M100, 100, bilstm=False, loss=sparse_categorical_focal_loss(2.0), dense_act="sigmoid"),
    "C4 300d + BiLSTM + Focal":   lambda: build(EMB_M300, 300, bilstm=True,  loss=sparse_categorical_focal_loss(2.0), dense_act="relu", dropout=0.5, spatial_do=0.3, opt_name="adam"),
}

SEEDS = [42, 7, 123]
rows = []
for name, builder in CONFIGS.items():
    accs, fs, fns = [], [], []
    t0 = time.time()
    for s in SEEDS:
        a, f, fn = run_once(builder, s)
        accs.append(a); fs.append(f); fns.append(fn)
    rows.append({"config": name,
                 "acc_mean": np.mean(accs)*100, "macroF1_mean": np.mean(fs)*100, "macroF1_std": np.std(fs)*100,
                 "neutralF1_mean": np.mean(fns)*100, "time": time.time()-t0})
    print(f"[{name:30s}] acc={np.mean(accs)*100:5.2f}  macroF1={np.mean(fs)*100:5.2f}±{np.std(fs)*100:.2f}  neutralF1={np.mean(fns)*100:5.2f}  ({time.time()-t0:.0f}s)", flush=True)

print("\n" + "="*78)
print("VÒNG 2 — trung bình 3 seeds (baseline gốc ~ acc 89.9 / macroF1 75.7)")
print("="*78)
rdf = pd.DataFrame(rows).sort_values("macroF1_mean", ascending=False)
print(rdf.round(2).to_string(index=False))
with open("_experiment2_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2, default=float)
print("\nSaved _experiment2_results.json")
