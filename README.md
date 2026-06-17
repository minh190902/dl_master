# Multi-channel LSTM-CNN cho Phân tích Cảm xúc Tiếng Việt

Tái hiện kiến trúc **Multi-channel LSTM-CNN** trong bài báo và huấn luyện trên một bài toán
*text classification* tiếng Việt.

> Vo, Q., Nguyen, H., Le, B., Nguyen, M. (2017). *Multi-channel LSTM-CNN model for Vietnamese
> sentiment analysis.* KSE 2017, IEEE.
> [Paper](https://www.researchgate.net/publication/321259272_Multi-channel_LSTM-CNN_model_for_Vietnamese_sentiment_analysis)
> · [Code gốc](https://github.com/ntienhuy/MultiChannel)

## Nội dung

| Yêu cầu bài tập | Đáp ứng |
|-----------------|---------|
| 1. Thiết kế kiến trúc CNN + LSTM như paper | Model gốc trong notebook (§3) |
| 2. Bài toán text classification + dataset tiếng Việt | Sentiment 3 lớp trên **UIT-VSFC** (§1) |
| 3. Huấn luyện model | §3-§4: huấn luyện mô hình gốc và các phương án cải tiến |
| 4. Report kết quả (dạng bảng) | §4.1, §5, §6 trong notebook + file **[REPORT.md](REPORT.md)** |

## Cấu trúc thư mục

```text
deep_learning/
├── 25C15052.ipynb              # Notebook chính = FILE NỘP (self-contained, Run All là chạy)
├── download_data.py            # Tải dataset UIT-VSFC từ Google Drive
├── build_embedding_matrix.py   # Trích PhoW2V gọn cho vocab UIT-VSFC
├── requirements.txt            # Dependencies
├── README.md
└── data/
    ├── UIT-VSFC/               # Dataset (tạo bởi download_data.py)
    │   ├── train/  {sents,sentiments,topics}.txt   (11.426 câu)
    │   ├── dev/    {sents,sentiments,topics}.txt   ( 1.583 câu)
    │   └── test/   {sents,sentiments,topics}.txt   ( 3.166 câu)
    ├── phow2v_uitvsfc_100d.npy        # Embedding gọn (<1MB, cho model cải tiến)
    └── phow2v_uitvsfc_vocab.json
```

## Bài toán & Dataset

- **Bài toán:** phân loại cảm xúc câu phản hồi → `negative / neutral / positive` (3 lớp).
- **Dataset:** UIT-VSFC (Vietnamese Students' Feedback Corpus), ~16.175 câu, đã tách từ sẵn.
- **Lưu ý:** lớp `neutral` chỉ ~4-5% → dùng `class_weight` + report **macro-F1** (không chỉ accuracy).

## Kiến trúc

**Mô hình gốc (theo bài báo):** Embedding(100), ba kênh CNN (Conv1D k=3/5/7, 150 filter,
GlobalMaxPool) kết hợp một kênh LSTM(128), ghép nối rồi Dense(200, sigmoid), Dropout(0.2),
Softmax(3). Bộ tối ưu Adamax.

**Các phương án cải tiến (khảo sát trong notebook):** giữ xương sống đa kênh CNN+LSTM, thay đổi
(1) biểu diễn từ bằng **PhoW2V** tiền huấn luyện (cố định hoặc tinh chỉnh) và (2) hàm mất mát bằng
**Focal loss**; có xét thêm **BiLSTM**. Mỗi phương án huấn luyện trên 3 hạt giống (42, 7, 123) và
so sánh theo trung bình kèm độ lệch chuẩn để chọn phương án tốt nhất.

> **Lưu ý dữ liệu PhoW2V:** tệp gốc ~1.1GB không kèm trong project; notebook nhúng sẵn ma trận rút
> gọn dưới dạng base64. Để tạo lại từ đầu: tải PhoW2V syllable 100d từ
> [github.com/datquocnguyen/PhoW2V](https://github.com/datquocnguyen/PhoW2V), giải nén vào
> `data/phow2v/`, rồi chạy `python build_embedding_matrix.py`.

## Cách chạy

```bash
# 1. Tạo môi trường (Python 3.11, TensorFlow 2.16 cần numpy < 2.0)
conda create -n vn-sentiment python=3.11 -y
conda activate vn-sentiment
pip install -r requirements.txt

# 2. Tải dataset
python download_data.py

# 3. Mở notebook và Run All
jupyter notebook 25C15052.ipynb
```

Notebook đã được chạy sẵn end-to-end (có lưu output + bảng kết quả). Huấn luyện 15 mô hình
(5 cấu hình × 3 seed) trên CPU mất khoảng 90 phút.

## Kết quả (trung bình ± độ lệch chuẩn, 3 seed, tập test UIT-VSFC, n = 3.166)

| Mô hình | Accuracy | Macro-F1 | Params |
|---------|:--------:|:--------:|:------:|
| Mô hình gốc (LSTM-CNN) | 88.98 ± 0.30 | 75.74 ± 0.26 | 710.701 |
| C1: PhoW2V cố định + LSTM + CE | 82.96 ± 1.70 | 70.78 ± 1.34 | 710.701 |
| C2: PhoW2V tinh chỉnh + LSTM + CE | 87.77 ± 1.30 | 75.33 ± 0.94 | 710.701 |
| **C3: PhoW2V tinh chỉnh + BiLSTM + Focal** | **90.50 ± 0.04** | **76.97 ± 0.22** | 853.549 |
| C4: PhoW2V tinh chỉnh + LSTM + Focal | 90.45 ± 0.30 | 76.96 ± 0.65 | 710.701 |

Mô hình cải tiến tốt nhất (C3) vượt mô hình gốc **+1.53% accuracy, +1.23% macro-F1**. Đáng chú ý,
C3 (BiLSTM) và C4 (LSTM) gần như tương đương, cho thấy lợi ích của biểu diễn từ tốt (PhoW2V) và
Focal loss lớn hơn lợi ích của việc tăng độ phức tạp kiến trúc. Lớp `neutral` (~5% dữ liệu) khó
nhất nên macro-F1 thấp hơn accuracy, đó là lý do dùng macro-F1 làm độ đo chính.
