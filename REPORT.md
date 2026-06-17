# Báo cáo: Mô hình Multi-channel LSTM-CNN cho Phân tích Cảm xúc Tiếng Việt

**Bài báo gốc:** Vo, Q., Nguyen, H., Le, B., Nguyen, M. (2017). *Multi-channel LSTM-CNN model for
Vietnamese sentiment analysis.* KSE 2017, IEEE.
[Paper](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis)
· [Mã nguồn nhóm tác giả](https://github.com/ntienhuy/MultiChannel)

> Toàn bộ kết quả trong báo cáo này được sinh trực tiếp khi chạy notebook `25C15052.ipynb`. Mỗi cấu
> hình được huấn luyện trên **ba hạt giống ngẫu nhiên** (42, 7, 123) và báo cáo theo trung bình kèm
> độ lệch chuẩn, bảo đảm kết luận không phụ thuộc vào một lần khởi tạo may rủi.

## 1. Bài toán và bộ dữ liệu

| Mục | Nội dung |
|-----|----------|
| Bài toán | Phân loại cảm xúc câu (text classification) tiếng Việt, 3 lớp |
| Bộ dữ liệu | UIT-VSFC (Vietnamese Students' Feedback Corpus) |
| Số mẫu | 16.175 câu (train 11.426 / dev 1.583 / test 3.166) |
| Nhãn | 0 = negative, 1 = neutral, 2 = positive |
| Đặc điểm | Đã tách từ sẵn; lớp `neutral` rất hiếm (~5%), gây mất cân bằng |

Do mất cân bằng lớp, **macro-F1** được dùng làm độ đo chính (bên cạnh accuracy), kết hợp trọng số
lớp (`class_weight`) và Focal loss khi huấn luyện.

## 2. Kiến trúc mô hình

**Mô hình gốc (theo bài báo):** một lớp Embedding dùng chung, ba nhánh CNN song song (Conv1D cửa
sổ 3/5/7, mỗi nhánh 150 filter, max-pooling-over-time) kết hợp một kênh LSTM (128 đơn vị); các kênh
được ghép nối rồi đưa qua Dense(200, sigmoid) và softmax 3 lớp. Bộ tối ưu Adamax, hàm mất mát
cross-entropy. Embedding học từ đầu.

**Các phương án cải tiến (khảo sát):** giữ xương sống đa kênh, thay đổi hai yếu tố then chốt là
chất lượng biểu diễn từ (dùng PhoW2V tiền huấn luyện) và hàm mất mát (Focal loss).

## 3. Kết quả huấn luyện

### 3.1. Bảng so sánh tất cả phương án (trung bình ± độ lệch chuẩn, 3 seed, tập kiểm thử)

| Mô hình | Accuracy | Macro-F1 | Neutral-F1 | Weighted-F1 | Params |
|---------|:--------:|:--------:|:----------:|:-----------:|:------:|
| Mô hình gốc (LSTM-CNN) | 88.98 ± 0.30 | 75.74 ± 0.26 | 44.02 ± 0.58 | 89.12 ± 0.14 | 710.701 |
| C1: PhoW2V cố định + LSTM + CE | 82.96 ± 1.70 | 70.78 ± 1.34 | 37.98 ± 1.96 | 84.65 ± 1.11 | 710.701 |
| C2: PhoW2V tinh chỉnh + LSTM + CE | 87.77 ± 1.30 | 75.33 ± 0.94 | 43.82 ± 1.35 | 88.62 ± 0.76 | 710.701 |
| **C3: PhoW2V tinh chỉnh + BiLSTM + Focal** | **90.50 ± 0.04** | **76.97 ± 0.22** | 45.72 ± 0.66 | 90.14 ± 0.08 | 853.549 |
| C4: PhoW2V tinh chỉnh + LSTM + Focal | 90.45 ± 0.30 | 76.96 ± 0.65 | 45.73 ± 1.49 | 90.12 ± 0.29 | 710.701 |

Phương án cải tiến tốt nhất theo macro-F1 trung bình là **C3 (PhoW2V tinh chỉnh + BiLSTM + Focal)**.

### 3.2. So sánh mô hình gốc và mô hình cải tiến tốt nhất

| | Mô hình gốc | Mô hình cải tiến (C3) | Chênh lệch |
|---|:---:|:---:|:---:|
| Accuracy | 88.98% | 90.50% | **+1.53** |
| Macro-F1 | 75.74% | 76.97% | **+1.23** |
| Weighted-F1 | 89.12% | 90.14% | **+1.02** |

### 3.3. Chi tiết theo từng lớp (seed = 42)

**Mô hình gốc**

| Lớp | Precision | Recall | F1 | Support |
|-----|:---------:|:------:|:--:|:-------:|
| negative | 0.9012 | 0.9255 | 0.9132 | 1409 |
| neutral  | 0.4591 | 0.4371 | 0.4479 | 167 |
| positive | 0.9301 | 0.9126 | 0.9213 | 1590 |
| macro avg | 0.7635 | 0.7584 | 0.7608 | 3166 |

**Mô hình cải tiến (C3)**

| Lớp | Precision | Recall | F1 | Support |
|-----|:---------:|:------:|:--:|:-------:|
| negative | 0.9027 | 0.9411 | 0.9215 | 1409 |
| neutral  | 0.5526 | 0.3772 | 0.4484 | 167 |
| positive | 0.9318 | 0.9277 | 0.9297 | 1590 |
| macro avg | 0.7957 | 0.7487 | 0.7665 | 3166 |

## 4. Nhận xét và Kết luận

1. **Tái hiện thành công kiến trúc của bài báo.** Mô hình gốc đạt accuracy 88.98% và macro-F1
   75.74%, nằm trong khoảng hiệu năng thường thấy của các mô hình học sâu trên UIT-VSFC.

2. **Chất lượng biểu diễn từ là yếu tố quyết định.** Khởi tạo embedding bằng PhoW2V và cho phép
   tinh chỉnh, kết hợp Focal loss (C3, C4), nâng macro-F1 trung bình lên gần 77%. Ngược lại, cố
   định embedding (C1) làm giảm mạnh hiệu năng do lệch miền dữ liệu.

3. **Đánh giá nhiều hạt giống là cần thiết.** Hai phương án tốt nhất C3 (BiLSTM) và C4 (LSTM) gần
   như tương đương (macro-F1 76.97 so với 76.96), chênh lệch nằm trong độ lệch chuẩn. Điều này cho
   thấy việc tăng độ phức tạp bằng BiLSTM không mang lại lợi ích rõ rệt so với LSTM một chiều vốn
   nhẹ hơn (710.701 so với 853.549 tham số).

4. **Lớp `neutral` là yếu tố hạn chế chính** (chỉ ~5% dữ liệu), F1 khoảng 45% kéo macro-F1 xuống
   thấp hơn nhiều so với accuracy.

**Kết luận:** mô hình cải tiến vượt mô hình gốc một cách ổn định (+1.53% accuracy, +1.23% macro-F1
trung bình trên 3 seed). Với bộ dữ liệu quy mô hạn chế, chất lượng biểu diễn từ quan trọng hơn việc
gia tăng độ phức tạp kiến trúc.

**Hướng phát triển:** sử dụng mô hình Transformer tiền huấn luyện (PhoBERT); tăng cường dữ liệu cho
lớp `neutral`; mở rộng số hạt giống đánh giá.

---

*Chi tiết mã nguồn, log huấn luyện và biểu đồ: xem notebook `25C15052.ipynb`.*
