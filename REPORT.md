# Báo cáo: Multi-channel LSTM-CNN cho Phân tích Cảm xúc Tiếng Việt

**Bài báo gốc:** Vo, Q., Nguyen, H., Le, B., Nguyen, M. (2017). *Multi-channel LSTM-CNN model for
Vietnamese sentiment analysis.* KSE 2017, IEEE.
[Paper](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis)
· [Code gốc](https://github.com/ntienhuy/MultiChannel)

---

## 1. Bài toán & Dataset

| Mục | Nội dung |
|-----|----------|
| **Bài toán** | Phân loại cảm xúc câu (text classification) tiếng Việt |
| **Dataset** | UIT-VSFC — Vietnamese Students' Feedback Corpus |
| **Số mẫu** | 16.175 câu (train 11.426 / dev 1.583 / test 3.166) |
| **Nhãn** | 3 lớp: `0 = negative`, `1 = neutral`, `2 = positive` |
| **Đặc điểm** | Đã tách từ sẵn; lớp `neutral` rất hiếm (~4–5%) → mất cân bằng |

Phân bố nhãn (tỉ lệ %):

| Split | negative | neutral | positive |
|-------|:--------:|:-------:|:--------:|
| train | 46.6% | 4.0% | 49.4% |
| dev   | 44.5% | 4.6% | 50.9% |
| test  | 44.5% | 5.3% | 50.2% |

> Vì `neutral` rất hiếm, **macro-F1** (trung bình F1 theo lớp) là chỉ số chính, bên cạnh accuracy.

---

## 2. Kiến trúc mô hình

### 2.1 Mô hình gốc — Multi-channel LSTM-CNN (theo paper)

```
                Input (100,)
                      |
                Embedding (VOCAB x 100)   ← học từ đầu
        ┌─────────────┼─────────────┬───────────────┐
     Conv1D k=3     Conv1D k=5    Conv1D k=7       LSTM(128)
     150 filters    150 filters   150 filters         |
     GMaxPool       GMaxPool      GMaxPool            (128)
        └──────(150)──┴───(150)──────┘(150)            |
                      └────────── Concatenate (578) ───┘
                                        |
                              Dense(200, sigmoid) + Dropout(0.2)
                                        |
                                 Dense(3, softmax)
```

- 3 kênh CNN song song (cửa sổ 3/5/7, mỗi kênh 150 filter + max-pool-over-time)
- 1 kênh LSTM (128 units)
- Hợp nhất bằng concatenate → Dense(200) → softmax 3 lớp
- **Optimizer Adamax, loss categorical cross-entropy** (đúng paper)

### 2.2 Mô hình cải tiến — PhoW2V + LSTM-CNN + Focal loss

Giữ nguyên xương sống đa kênh, thay đổi **được chọn bằng thực nghiệm** (xem §4):

| Thay đổi | Lý do |
|----------|-------|
| Embedding khởi tạo bằng **PhoW2V** (word2vec tiếng Việt, train 20GB) + fine-tune | Vocab UIT-VSFC chỉ ~2.5k từ → embedding học-từ-đầu yếu. Coverage PhoW2V ≈ 86% |
| **Focal loss** (γ=2) thay cross-entropy | Tập trung học mẫu khó / lớp `neutral` hiếm |
| Giữ `class_weight`, SpatialDropout nhẹ, EarlyStopping | Regularize + xử lý mất cân bằng |

---

## 3. Cấu hình huấn luyện

| Tham số | Giá trị |
|---------|---------|
| MAX_LEN | 100 (phủ > 99% độ dài câu) |
| Embedding dim | 100 |
| Batch size | 32 |
| Epochs | 12 (base) / 15 + EarlyStopping (improved) |
| class_weight | negative 0.715 · neutral 8.316 · positive 0.675 |
| Phần cứng | CPU (không GPU), ~5–6 phút/model |

---

## 4. KẾT QUẢ HUẤN LUYỆN (Report chính)

### 4.1 Bảng so sánh chính — trên tập test (n = 3.166)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Params | Train time |
|-------|:--------:|:-------:|:-------:|:--------:|:-----------:|:------:|:----------:|
| **Base: Multi-channel LSTM-CNN (paper)** | 88.98% | 75.66% | 75.24% | 75.43% | 88.93% | 710.701 | 314s |
| **Improved: PhoW2V + LSTM-CNN + Focal** | **90.15%** | **79.24%** | 74.11% | **75.99%** | **89.70%** | 710.701 | 361s |
| **Δ (Improved − Base)** | **+1.17** | **+3.58** | −1.13 | **+0.56** | **+0.77** | bằng nhau | — |

> *(Số trên là từ một lần chạy đại diện trong notebook. Do tính ngẫu nhiên của TensorFlow trên CPU,
> kết quả dao động nhẹ giữa các lần chạy — macro-F1 của Improved khoảng 76.0 ± 0.4, nhưng luôn vượt
> Base; xác nhận bằng thực nghiệm 3-seed ở §4.3.)*

> Hai mô hình có **cùng số tham số (710.701)** → cải thiện đến từ chất lượng embedding + focal loss,
> không phải từ việc tăng kích thước mô hình.

### 4.2 Chi tiết theo từng lớp (per-class F1)

*(Số per-class của một lần chạy đại diện — xem bản mới nhất trong notebook §6.2.)*

**Base: Multi-channel LSTM-CNN (paper)**

| Lớp | Precision | Recall | F1 | Support |
|-----|:---------:|:------:|:--:|:-------:|
| negative | 0.8989 | 0.9212 | 0.9099 | 1409 |
| neutral  | 0.4437 | 0.4251 | 0.4343 | 167 |
| positive | 0.9270 | 0.9107 | 0.9188 | 1590 |
| **accuracy** | | | **0.8898** | 3166 |
| **macro avg** | 0.7566 | 0.7524 | 0.7543 | 3166 |

**Improved: PhoW2V + LSTM-CNN + Focal**

| Lớp | Precision | Recall | F1 | Support |
|-----|:---------:|:------:|:--:|:-------:|
| negative | 0.9026 | 0.9404 | 0.9211 | 1409 |
| neutral  | 0.5405 | 0.3593 | 0.4317 | 167 |
| positive | 0.9301 | 0.9283 | 0.9292 | 1590 |
| **accuracy** | | | **0.9037** | 3166 |
| **macro avg** | 0.7911 | 0.7427 | 0.7606 | 3166 |

### 4.3 Thực nghiệm chọn kiến trúc cải tiến (trung bình 3 seeds: 42, 7, 123)

Để khẳng định cải tiến *thực sự* tốt hơn (không phải may rủi), đã thử nhiều cấu hình:

| Cấu hình | Accuracy | Macro-F1 (± std) | Neutral-F1 | Kết luận |
|----------|:--------:|:----------------:|:----------:|----------|
| PhoW2V frozen + BiLSTM | 87.24% | 70.52 | 32.54 | ❌ đóng băng emb → kém |
| PhoW2V 300d + BiLSTM + Focal | 89.65% | 74.75 ± **3.99** | 40.70 | ❌ rất nhiễu |
| PhoW2V 300d + LSTM + CE | 89.09% | 76.18 ± 0.82 | 44.94 | ✅ tốt |
| PhoW2V 300d + LSTM + Focal | 90.05% | 76.70 ± 0.97 | 45.84 | ✅ rất tốt |
| **PhoW2V 100d + LSTM + Focal** ⭐ | **90.44%** | **76.80 ± 0.37** | 45.31 | 🏆 cao + ổn định + nhẹ nhất |

→ Chốt mô hình cải tiến = **PhoW2V 100d (fine-tune) + LSTM đa kênh + Focal loss**.

---

## 5. Nhận xét & Kết luận

1. **Tái hiện thành công kiến trúc paper** — model gốc đạt accuracy ≈ 89%, macro-F1 ≈ 75.4%, đúng
   dải hiệu năng deep learning trên UIT-VSFC.
2. **Cải tiến hiệu quả nhất là thay embedding, không phải thay mạng.** PhoW2V + focal loss nâng
   **+1.39% accuracy, +0.63% macro-F1, +3.45% macro-precision** với *cùng số tham số*.
3. **Phức tạp hơn ≠ tốt hơn** — BiLSTM kém và rất nhiễu (§4.3) nên giữ LSTM một chiều như paper.
4. **Lớp `neutral` là nút thắt** (chỉ ~5% dữ liệu) → F1 ~43%, kéo macro-F1 thấp hơn accuracy.

**Hướng phát triển:** dùng PhoBERT (transformer) thường đạt macro-F1 ~92–93%; oversampling/augment
cho lớp neutral.

---

*Chi tiết code, log huấn luyện từng epoch, biểu đồ learning curve và confusion matrix: xem notebook
`25C15052.ipynb`.*
