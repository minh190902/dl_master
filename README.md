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
| 3. Huấn luyện model | §5 — train cả model gốc + cải tiến |
| 4. Report kết quả (dạng bảng) | §6 trong notebook + file **[REPORT.md](REPORT.md)** |

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

**Model gốc (theo paper):** Embedding(200) → [Conv1D k=3/5/7, 150 filters → GlobalMaxPool] × 3
kênh CNN + 1 kênh LSTM(128) → Concatenate → Dense(200, sigmoid) → Dropout(0.2) → Softmax(3).
Optimizer **Adamax**.

**Model cải tiến (chọn bằng thực nghiệm):** giữ xương sống đa kênh CNN+LSTM của paper, nhưng
(1) khởi tạo embedding bằng **PhoW2V** (word2vec tiếng Việt, train trên 20GB — fine-tune) và
(2) dùng **focal loss** để học tốt hơn lớp `neutral` hiếm. Đã chạy multi-seed (3 seeds) để xác nhận
cải tiến **ổn định vượt** bản gốc, đồng thời loại bỏ BiLSTM vì kém + nhiễu (xem §4.1 trong notebook).

> **Lưu ý dữ liệu PhoW2V:** file gốc ~1.1GB không kèm trong project. Notebook nạp từ
> `data/phow2v_uitvsfc_100d.npy` (< 1MB, đã trích sẵn). Để tạo lại từ đầu: tải PhoW2V syllable 100d
> từ [github.com/datquocnguyen/PhoW2V](https://github.com/datquocnguyen/PhoW2V), giải nén vào
> `data/phow2v/`, rồi chạy `python build_embedding_matrix.py`.

## Cách chạy

```bash
# 1. Tạo môi trường (Python 3.11 — TensorFlow 2.16 cần numpy < 2.0)
conda create -n vn-sentiment python=3.11 -y
conda activate vn-sentiment
pip install -r requirements.txt

# 2. Tải dataset
python download_data.py

# 3. Mở notebook và Run All
jupyter notebook 25C15052.ipynb
```

Notebook đã được chạy sẵn end-to-end (có lưu output + bảng kết quả). Train trên CPU mất ~15 phút.

## Kết quả (tập test UIT-VSFC, n = 3.166)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Params |
|-------|:--------:|:--------:|:-----------:|:------:|
| Base: Multi-channel LSTM-CNN (paper) | 88.98% | 75.43% | 88.93% | 710.701 |
| **Improved: PhoW2V + LSTM-CNN + Focal** | **90.37%** | **76.06%** | **89.93%** | 710.701 |
| **Δ cải thiện** | **+1.39** | **+0.63** | **+1.00** | (cùng params) |

Bản cải tiến vượt bản gốc với **cùng số tham số** → cải thiện đến từ embedding tốt hơn + focal loss,
được xác nhận ổn định qua thực nghiệm multi-seed (§4.1 trong notebook). Lớp `neutral` (~5% dữ liệu)
là khó nhất nên macro-F1 thấp hơn accuracy — đó là lý do report macro-F1 làm chỉ số chính.
